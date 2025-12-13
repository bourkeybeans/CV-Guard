import json
import os

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from werkzeug.utils import secure_filename

from agent_workflow import (
    CVClaimsAgent,
    ReliabilityScoringAgent,
    RepoVerificationAgent,
    client as agent_client,
    run_claimcheck,
)
from gitTool import get_github_repositories

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        cv_file = request.files.get("cv")
        cv_path = None

        if cv_file and cv_file.filename.endswith(".pdf"):
            filename = secure_filename(cv_file.filename)
            cv_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            cv_file.save(cv_path)

        linkedin_url = request.form.get("linkedin")

        github_username = request.form.get("github")

        if not cv_path:
            return jsonify({"error": "A PDF CV is required."}), 400

        try:
            # Run the three-agent pipeline end-to-end.
            result = run_claimcheck(
                cv_path=cv_path,
                github_username=github_username,
                linkedin_url=linkedin_url,
                github_token=request.form.get("github_token") or None,
                verbose=bool(request.form.get("verbose")),
            )
        except Exception as exc:  # pylint: disable=broad-except
            return jsonify({"error": str(exc)}), 500

        return jsonify(result)

    return render_template("index.html")


def _sse(event: str, data: dict) -> str:
    """Format data as an SSE event chunk."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.route("/stream", methods=["POST"])
def stream():
    cv_file = request.files.get("cv")
    cv_path = None

    if cv_file and cv_file.filename.endswith(".pdf"):
        filename = secure_filename(cv_file.filename)
        cv_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        cv_file.save(cv_path)

    linkedin_url = request.form.get("linkedin")
    github_username = request.form.get("github")
    github_token = request.form.get("github_token") or None
    verbose = bool(request.form.get("verbose"))

    if not cv_path:
        return jsonify({"error": "A PDF CV is required."}), 400

    @stream_with_context
    def generate():
        transcript = [] if verbose else None
        claims_agent = CVClaimsAgent(agent_client)
        verifier = RepoVerificationAgent(agent_client)
        scorer = ReliabilityScoringAgent(agent_client)

        try:
            yield _sse("status", {"message": "Starting ClaimCheck pipeline..."})

            claims_payload = claims_agent.gather(
                cv_path=cv_path,
                linkedin_url=linkedin_url,
                transcript=transcript,
            )
            yield _sse(
                "claims",
                {
                    "count": len(claims_payload["claims"]),
                    "summary": claims_payload.get("summary", ""),
                },
            )

            yield _sse("status", {"message": "Fetching GitHub repositories..."})
            repos = get_github_repositories(github_username, github_token)
            yield _sse("repos", {"count": len(repos)})

            verification = verifier.verify(
                claims_payload["claims"], repos, transcript=transcript
            )
            yield _sse("verification", verification)

            reliability = scorer.score(
                claims_payload["claims"], verification, transcript=transcript
            )
            yield _sse("reliability", reliability)

            if verbose:
                yield _sse("transcript", transcript)

            yield _sse(
                "done",
                {
                    "claims": claims_payload["claims"],
                    "claims_summary": claims_payload["summary"],
                    "structured_cv": claims_payload["structured_cv"],
                    "repos_checked": len(repos),
                    "verification": verification,
                    "reliability": reliability,
                    "transcript": transcript or [],
                },
            )
        except Exception as exc:  # pylint: disable=broad-except
            yield _sse("error", {"error": str(exc)})

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True)
