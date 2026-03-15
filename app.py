import os
import io
import json
import requests
import numpy as np
import face_recognition
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app, origins=["https://calm-banoffee-c97570.netlify.app", "http://localhost"])

def download_image(url, headers=None):
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        # Resize to max 400px for speed
        img.thumbnail((400, 400), Image.LANCZOS)
        return np.array(img)
    except Exception as e:
        return None

def get_face_encoding(img_array):
    try:
        encs = face_recognition.face_encodings(img_array, num_jitters=1, model="small")
        return encs[0] if encs else None
    except:
        return None

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/encode-probe", methods=["POST"])
def encode_probe():
    """Encode the probe face image uploaded by the user"""
    try:
        file = request.files.get("photo")
        if not file:
            return jsonify({"error": "No photo uploaded"}), 400

        img = Image.open(file.stream).convert("RGB")
        img.thumbnail((600, 600), Image.LANCZOS)
        img_array = np.array(img)

        encs = face_recognition.face_encodings(img_array, num_jitters=2, model="small")
        if not encs:
            return jsonify({"error": "No face detected in photo"}), 400

        return jsonify({"encoding": encs[0].tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/scan", methods=["POST"])
def scan():
    """Scan Drive files and stream back matches as NDJSON"""
    data = request.get_json()
    probe_encoding = np.array(data.get("encoding", []))
    files = data.get("files", [])
    access_token = data.get("access_token", "")
    threshold = float(data.get("threshold", 0.5))

    if probe_encoding.size == 0 or not files or not access_token:
        return jsonify({"error": "Missing required fields"}), 400

    auth_headers = {"Authorization": f"Bearer {access_token}"}

    def generate():
        for i, f in enumerate(files):
            file_id = f.get("id")
            file_name = f.get("name")
            thumb_url = f.get("thumbnailLink")

            # Send progress
            yield json.dumps({"type": "progress", "current": i + 1, "total": len(files)}) + "\n"

            # Use thumbnail if available, else full image
            img_url = None
            if thumb_url:
                img_url = thumb_url.replace("=s220", "=s400").replace("=s0", "=s400")
                if "=s" not in img_url:
                    img_url += "=s400"
            
            img_array = None
            if img_url:
                img_array = download_image(img_url)
            
            if img_array is None:
                # Fallback to full image via Drive API
                full_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
                img_array = download_image(full_url, headers=auth_headers)

            if img_array is None:
                continue

            enc = get_face_encoding(img_array)
            if enc is None:
                continue

            distance = face_recognition.face_distance([probe_encoding], enc)[0]
            if distance < threshold:
                yield json.dumps({
                    "type": "match",
                    "id": file_id,
                    "name": file_name,
                    "distance": float(distance),
                    "similarity": round((1 - float(distance)) * 100)
                }) + "\n"

        yield json.dumps({"type": "done"}) + "\n"

    return Response(
        stream_with_context(generate()),
        mimetype="application/x-ndjson"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
