# app.py  — Python prediction API (run with: python app.py)
# ──────────────────────────────────────────────────────────────────────────────
# Loads your weights into the rebuilt architecture, applies the SAME M1+M2
# cleaning + raw [0,255] preprocessing as training, and serves /predict.
# ──────────────────────────────────────────────────────────────────────────────
import os, io, json
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # quieter TF logs

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

from model_def import build_model
from cleaning.clean import clean_one_image

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_WEIGHTS  = "models/best_model_m2.weights.h5"   # ← put your file here
THRESHOLD      = 0.16          # your F2-optimal threshold (0.50 for balanced)
APPLY_CLEANING = True          # True for the CLEANED model; False if you switch to RAW

app = Flask(__name__)
CORS(app)                      # allow the PHP/browser page to call this API

print("[app] Building model and loading weights...")
model = build_model(MODEL_WEIGHTS)
_ = model.predict(np.zeros((1, 224, 224, 3), np.float32), verbose=0)   # warm-up
print(f"[app] Ready. threshold={THRESHOLD}  cleaning={APPLY_CLEANING}")


def preprocess(image_bytes):
    """Bytes → (1,224,224,3) float32 in [0,255], matching training exactly."""
    rgb = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"), dtype=np.uint8)
    if APPLY_CLEANING:
        rgb = clean_one_image(rgb)                                  # M1+M2, returns 224×224
    else:
        rgb = np.asarray(Image.fromarray(rgb).resize((224, 224)), dtype=np.uint8)
    return rgb.astype(np.float32)[None]    # raw [0,255] — model normalizes internally


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no image uploaded (field name must be 'image')"}), 400
    try:
        x = preprocess(request.files["image"].read())
        prob = float(model.predict(x, verbose=0).ravel()[0])
        return jsonify({
            "prediction": "Malignant" if prob >= THRESHOLD else "Benign",
            "probability_malignant": round(prob, 4),
            "threshold_used": THRESHOLD,
            "disclaimer": "Research prototype, not a medical diagnosis. Consult a dermatologist."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
