"""
IRIS Frame Viewer — localhost web app.
Run:  python viewer/app.py
Open: http://localhost:5000
"""
from flask import Flask, render_template, jsonify, abort, send_file
from pathlib import Path
import json
import base64
import io

app = Flask(__name__, template_folder="templates")

DATA_FILE = Path(__file__).parent.parent / "frames_data.json"


def load_data():
    if not DATA_FILE.exists():
        return None
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


@app.route("/")
def index():
    data = load_data()
    if data is None:
        return (
            "<h2>frames_data.json not found.</h2>"
            "<p>Run <code>python extract_frames.py</code> first.</p>",
            404,
        )
    return render_template("index.html")


@app.route("/api/frames")
def api_frames():
    data = load_data()
    if data is None:
        abort(404)
    # Strip image_b64 from the JSON response — images are served separately
    slim = {
        "video":  data["video"],
        "stats":  data["stats"],
        "frames": [
            {k: v for k, v in f.items() if k != "image_b64"}
            for f in data["frames"]
        ],
    }
    return jsonify(slim)


@app.route("/api/frame/<int:frame_idx>/image")
def frame_image(frame_idx: int):
    """Serve one frame's JPEG directly — no base64, no data URI."""
    data = load_data()
    if data is None:
        abort(404)
    frames = {f["frame_idx"]: f for f in data["frames"]}
    if frame_idx not in frames:
        abort(404)
    img_bytes = base64.b64decode(frames[frame_idx]["image_b64"])
    return send_file(io.BytesIO(img_bytes), mimetype="image/jpeg")


if __name__ == "__main__":
    print("IRIS Frame Viewer running at http://localhost:5000")
    app.run(debug=False, port=5000)
