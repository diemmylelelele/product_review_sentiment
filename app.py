import os
import uuid
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, request, send_from_directory, url_for, render_template

# import FAISS + LLM pipeline
from classifier import classify

BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def save_upload(file_storage):
    """Save upload and return an absolute URL that main.embed_image_q can fetch via requests.get()."""
    if not file_storage or file_storage.filename == "":
        return None
    if not allowed_file(file_storage.filename):
        return None
    fname = secure_filename(file_storage.filename)
    stem, ext = os.path.splitext(fname)
    fname = f"{stem}_{uuid.uuid4().hex[:8]}{ext}"
    path = UPLOAD_DIR / fname
    file_storage.save(str(path))
    return url_for("uploaded_file", filename=fname, _external=True)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", result=None, text="", k=3, image_url=None, error=None)

    # POST
    try:
        text = (request.form.get("text") or "").strip()
        k = int(request.form.get("k") or 3)
        k = max(1, min(10, k))

        image_file = request.files.get("image")
        image_url = save_upload(image_file) if image_file else None

        result = classify(text=text if text else None, image_url=image_url, k=k)

        return render_template(
            "index.html",
            result=result,
            text=text,
            k=k,
            image_url=image_url,
            error=None,
        )
    except Exception as e:
        return render_template(
            "index.html",
            result=None,
            text=request.form.get("text") or "",
            k=request.form.get("k") or 3,
            image_url=None,
            error=str(e),
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
