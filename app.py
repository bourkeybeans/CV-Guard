import os

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from agent_workflow import run_claimcheck

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
            )
        except Exception as exc:  # pylint: disable=broad-except
            return jsonify({"error": str(exc)}), 500

        return jsonify(result)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
