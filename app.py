import os
import io
import json
import requests
import numpy as np
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from PIL import Image
from deepface import DeepFace

app = Flask(__name__)
CORS(app, origins=["https://calm-banoffee-c97570.netlify.app", "http://localhost"])

def download_image_bytes(url, headers=None):
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        return r.content
    except:
        return None

def get_embedding(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.thumbnail((400, 400), Image.LANCZOS)
        img_array = np.array(img)
        result = DeepFace.represent(
            img_path=img_array,
            model_name="Facenet",
            enforce_detection=True,
            detector_backend="opencv"
        )
        return np.array(result[0]["embedding"])
    except:
        return None

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/encode-probe", methods=["POST"])
def encode_probe():
    try:
        file = request.files.get("photo")
        if not file:
            return jsonify({"error": "No photo uploaded"}), 400
        img_bytes = file.read()
        embedding = get_embedding(img_bytes)
        if embedding is None:
            return jsonify({"error": "No face detected in photo"}), 400
        return jsonify({"encoding": embedding.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/scan", methods=["POST"])
def scan():
    data = request.get_json()
    probe_embedding = data.get("encoding", [])
    files = data.get("files", [])
    access_token = data.get("access_token", "")
    threshold = float(data.get("threshold", 0.5))
    # Convert threshold (0-1 distance) to similarity (cosine)
    similarity_threshold = 1 - threshold

    if not probe_embedding or not files or not access_token:
        return jsonify({"error": "Missing required fields"}), 400

    auth_headers = {"Authorization": f"Bearer {access_token}"}

    def generate():
        for i, f in enumerate(files):
            file_id = f.get("id")
            file_name = f.get("name")
            thumb_url = f.get("thumbnailLink")

            yield json.dumps({"type": "progress", "current": i + 1, "total": len(files)}) + "\n"

            # Try thumbnail first, fallback to full image
            img_bytes = None
            if thumb_url:
                url = thumb_url
                if "=s" in url:
                    url = url.rsplit("=s", 1)[0] + "=s400"
                else:
                    url += "=s400"
                img_bytes = download_image_bytes(url)

            if img_bytes is None:
                img_bytes = download_image_bytes(
                    f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media",
                    headers=auth_headers
                )

            if img_bytes is None:
                continue

            embedding = get_embedding(img_bytes)
            if embedding is None:
                continue

            similarity = cosine_similarity(probe_embedding, embedding)
            if similarity >= similarity_threshold:
                yield json.dumps({
                    "type": "match",
                    "id": file_id,
                    "name": file_name,
                    "similarity": round(similarity * 100)
                }) + "\n"

        yield json.dumps({"type": "done"}) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
