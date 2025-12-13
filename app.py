import json
import os
import time

from flask import Flask, Response, jsonify, redirect, render_template, request, session, stream_with_context
from werkzeug.utils import secure_filename

from agent_workflow import (
    CVClaimsAgent,
    LinkedInVerificationAgent,
    ReliabilityScoringAgent,
    RepoVerificationAgent,
    SummaryAgent,
    LINKEDIN_API_KEY,
    client as agent_client,
    extract_linkedin_username,
    fetch_linkedin_profile,
    run_claimcheck,
)
from gitTool import get_github_repositories

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle form submission - save to session and redirect
        cv_file = request.files.get("cv")
        session_id = str(int(time.time() * 1000))
        
        if cv_file and cv_file.filename.endswith(".pdf"):
            filename = secure_filename(cv_file.filename)
            # Save with session ID to avoid conflicts
            session_filename = f"{session_id}_{filename}"
            cv_path = os.path.join(app.config["UPLOAD_FOLDER"], session_filename)
            cv_file.save(cv_path)
        else:
            return jsonify({"error": "A PDF CV is required."}), 400

        # Store form data in Flask session
        session[f"form_data_{session_id}"] = {
            "cv_path": cv_path,
            "linkedin_url": request.form.get("linkedin", ""),
            "github_username": request.form.get("github", ""),
            "github_token": request.form.get("github_token") or None,
            "verbose": bool(request.form.get("verbose")),
        }

        return redirect(f"/discussion?session={session_id}")

    return render_template("index.html")


@app.route("/discussion", methods=["GET"])
def discussion():
    return render_template("discussion.html")


def _sse(event: str, data: dict) -> str:
    """Format data as an SSE event chunk."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.route("/stream", methods=["POST"])
def stream():
    # Try to get from session first, then fall back to form data
    session_id = request.form.get("session_id") or request.args.get("session")
    form_data = None
    
    if session_id:
        form_data = session.get(f"form_data_{session_id}")
    
    if form_data:
        cv_path = form_data["cv_path"]
        linkedin_url = form_data["linkedin_url"]
        github_username = form_data["github_username"]
        github_token = form_data["github_token"]
        verbose = form_data["verbose"]
    else:
        # Fall back to form data (for direct submissions)
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
    
    linkedin_api_key = LINKEDIN_API_KEY

    if not cv_path or not os.path.exists(cv_path):
        return jsonify({"error": "A PDF CV is required."}), 400

    @stream_with_context
    def generate():
        # Always maintain transcript for the chat interface
        transcript = []
        last_sent_index = 0
        
        def send_new_transcript_entries():
            """Send any new transcript entries that haven't been sent yet"""
            nonlocal last_sent_index
            new_entries = transcript[last_sent_index:]
            if new_entries:
                for entry in new_entries:
                    yield _sse("transcript", [entry])
                last_sent_index = len(transcript)
        
        claims_agent = CVClaimsAgent(agent_client)
        verifier = RepoVerificationAgent(agent_client)
        linkedin_verifier = LinkedInVerificationAgent(agent_client)
        scorer = ReliabilityScoringAgent(agent_client)
        summarizer = SummaryAgent(agent_client)

        try:
            yield _sse("status", {"message": "Starting ClaimCheck pipeline..."})

            claims_payload = claims_agent.gather(
                cv_path=cv_path,
                linkedin_url=linkedin_url,
                transcript=transcript,
            )
            # Send any new transcript entries
            for update in send_new_transcript_entries():
                yield update
            
            yield _sse(
                "claims",
                {
                    "count": len(claims_payload["claims"]),
                    "summary": claims_payload.get("summary", ""),
                },
            )

            linkedin_profile = None
            linkedin_verification = None
            linkedin_username = extract_linkedin_username(linkedin_url) if linkedin_url else None

            if linkedin_username and linkedin_api_key:
                yield _sse("status", {"message": f"Fetching LinkedIn profile for {linkedin_username}..."})
                status, profile_or_error = fetch_linkedin_profile(
                    linkedin_username, linkedin_api_key, transcript=transcript
                )
                # Send any new transcript entries
                for update in send_new_transcript_entries():
                    yield update
                    
                yield _sse(
                    "linkedin_profile",
                    {"status": status, "username": linkedin_username},
                )
                if status == "success" and isinstance(profile_or_error, dict):
                    linkedin_profile = profile_or_error
                else:
                    transcript.append(
                        {
                            "agent": "LinkedInScraper",
                            "message": f"LinkedIn fetch failed ({status}): {profile_or_error}",
                        }
                    )
                    for update in send_new_transcript_entries():
                        yield update
            elif linkedin_username and not linkedin_api_key:
                transcript.append(
                    {
                        "agent": "LinkedInScraper",
                        "message": "Skipping LinkedIn fetch (missing LINKD_API_KEY).",
                    }
                )
                for update in send_new_transcript_entries():
                    yield update

            yield _sse("status", {"message": "Fetching GitHub repositories..."})
            repos = get_github_repositories(github_username, github_token)
            yield _sse("repos", {"count": len(repos)})

            verification = verifier.verify(
                claims_payload["claims"], repos, transcript=transcript
            )
            # Send any new transcript entries
            for update in send_new_transcript_entries():
                yield update
                
            yield _sse("verification", verification)

            if linkedin_profile:
                linkedin_verification = linkedin_verifier.verify(
                    claims_payload["claims"],
                    linkedin_profile,
                    transcript=transcript,
                )
                # Send any new transcript entries
                for update in send_new_transcript_entries():
                    yield update
                    
                yield _sse("linkedin_verification", linkedin_verification)

            reliability = scorer.score(
                claims_payload["claims"],
                verification,
                linkedin_verification=linkedin_verification,
                transcript=transcript,
            )
            # Send any new transcript entries
            for update in send_new_transcript_entries():
                yield update
                
            yield _sse("reliability", reliability)

            summary = summarizer.summarize(
                claims_payload["claims"],
                verification,
                linkedin_verification,
                reliability,
                transcript,
            )
            yield _sse("summary", summary)

            # Send all transcript entries at the end for verbose mode
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
                    "linkedin_username": linkedin_username,
                    "linkedin_profile": linkedin_profile,
                    "linkedin_verification": linkedin_verification,
                    "reliability": reliability,
                    "summary": summary,
                    "transcript": transcript,
                },
            )
        except Exception as exc:  # pylint: disable=broad-except
            yield _sse("error", {"error": str(exc)})

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True)
