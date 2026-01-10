import os
import cv2
import numpy as np
import logging

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# =====================
# LOG SUSTUR
# =====================
logging.getLogger("insightface").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

# =====================
# FASTAPI
# =====================
api = FastAPI(title="Face Recognition API")

# =====================
# PATHS
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMB_PATH = os.path.join(BASE_DIR, "embeddings.npy")
LBL_PATH = os.path.join(BASE_DIR, "labels.npy")
THR_PATH = os.path.join(BASE_DIR, "threshold.txt")

# =====================
# LOAD FILES (SAFE)
# =====================
if not os.path.exists(EMB_PATH):
    raise RuntimeError("embeddings.npy bulunamadı")

if not os.path.exists(LBL_PATH):
    raise RuntimeError("labels.npy bulunamadı")

if not os.path.exists(THR_PATH):
    raise RuntimeError("threshold.txt bulunamadı")

embeddings = np.load(EMB_PATH)
labels = np.load(LBL_PATH)

with open(THR_PATH, "r") as f:
    AUTO_THRESHOLD = float(f.read().strip())

PERSONS = ["Oh", "Oh1"]

# =====================
# LOAD MODEL
# =====================
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
face_app.prepare(ctx_id=-1, det_size=(640, 640))

# =====================
# HEALTH CHECK
# =====================
@api.get("/")
def root():
    return {
        "status": "OK",
        "persons": PERSONS,
        "threshold": AUTO_THRESHOLD
    }

# =====================
# IDENTIFY
# =====================
@api.post("/identify")
async def identify_face(file: UploadFile = File(...)):
    img_bytes = await file.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Görüntü okunamadı"}
        )

    faces = face_app.get(img)

    if len(faces) != 1:
        return {"result": "Hata: Tek yüz olmalı"}

    emb = faces[0].embedding.reshape(1, -1)
    sims = cosine_similarity(emb, embeddings)[0]

    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    if best_score < AUTO_THRESHOLD:
        return {
            "result": "Bilinmeyen ❌",
            "score": round(best_score, 3)
        }

    return {
        "result": PERSONS[labels[best_idx]],
        "score": round(best_score, 3)
    }
