"""Pemeriksaan singkat apakah bobot YOLO dapat dibaca."""

from utils import MODEL_PATH, load_yolo_model


model = load_yolo_model(MODEL_PATH)
print(f"MODEL TERBACA: {MODEL_PATH}")
print(f"KELAS MODEL: {model.names}")
