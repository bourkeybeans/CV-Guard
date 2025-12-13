from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

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

        # For now just print - later send to agents
        print("CV PATH:", cv_path)
        print("LinkedIn:", linkedin_url)
        print("GitHub:", github_username)

        return {
            "status": "received",
            "cv_saved": bool(cv_path),
            "linkedin": linkedin_url,
            "github": github_username
        }

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
