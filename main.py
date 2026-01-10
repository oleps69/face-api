# =========================================
# PROFESYONEL YÜZ TANIMA API (RAILWAY)
# ROOT DOSYA YAPISI UYUMLU
# =========================================

import os
import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

# =======================
# BASE PATH
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =======================
# DOSYA YOLLARI (ROOT)
# =======================
EMB_PATH = os.path.join(BASE_DIR, "embeddings.npy")
LBL_PATH = os.path.join(BASE_DIR, "labels.npy")
THR_PATH = os.path.join(BASE_DIR, "threshold.txt")

# =======================
# KİŞİ İSİMLERİ
# =======================
PERSONS = ["Oh", "Oh1"]  # labels.npy ile SIRASI UYUMLU OLMALI

# =======================
# INSIGHTFACE MODEL
# =======================
app_face = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app_face.prepare(ctx_id=0, det_size=(640, 640))

# =======================
# VERİLERİ YÜKLE
# =======================
embeddings = np.load(EMB_PATH)
labels = np.load(LBL_PATH)

with open(THR_PATH, "r") as f:
    AUTO_THRESHOLD = float(f.read())

# =======================
# FASTAPI APP
# =======================
api = FastAPI(title="Face Recognition API")

# =======================
# IMAGE DECODE
# =======================
def read_image_from_bytes(image_bytes):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    return img

# =======================
# IDENTIFY ENDPOINT
# =======================
@api.post("/identify")
async def identify_face(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = read_image_from_bytes(contents)

        if img is None:
            return {
                "error": "Görüntü okunamadı"
            }

        faces = app_face.get(img)

        if len(faces) != 1:
            return {
                "result": "Hata",
                "message": "Tek ve net bir yüz olmalı"
            }

        emb = faces[0].embedding.reshape(1, -1)
        sims = cosine_similarity(emb, embeddings)[0]

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score < AUTO_THRESHOLD:
            return {
                "result": "Bilinmeyen",
                "score": round(best_score, 3)
            }

        return {
            "result": PERSONS[labels[best_idx]],
            "score": round(best_score, 3)
        }

    except Exception as e:
        return {
            "error": "Sunucu hatası",
            "detail": str(e)
        }

# =======================
# HEALTH CHECK
# =======================
@api.get("/")
def root():
    return {
        "status": "OK",
        "threshold": AUTO_THRESHOLD,
        "persons": PERSONS
    }
