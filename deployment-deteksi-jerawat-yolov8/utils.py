"""Fungsi pemuatan model, inferensi YOLOv8, dan visualisasi hasil."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"

@st.cache_resource
def load_yolo_model(
    model_path: Path | str = MODEL_PATH
) -> YOLO:
    """Muat model YOLOv8 dari models/best.pt."""

    path = Path(model_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {path}. "
            "Letakkan model pada models/best.pt."
        )

    return YOLO(str(path))

ACNE_INFO = {
    "Blackhead": {
        "color": (124, 140, 245),
        "cause": "Blackhead terjadi ketika pori-pori terbuka tersumbat oleh minyak dan sel kulit mati.",
        "solution": "Bersihkan wajah secara lembut, gunakan produk non-komedogenik, dan hindari memencet komedo.",
    },

    "Whitehead": {
        "color": (92, 200, 190),
        "cause": "Whitehead terjadi akibat penyumbatan minyak dan sel kulit mati pada pori-pori yang tertutup.",
        "solution": "Gunakan pembersih yang lembut, produk non-komedogenik, dan hindari menggosok atau memencetnya.",
    },

    "Papula": {
        "color": (242, 184, 75),
        "cause": "Papula dipengaruhi oleh penyumbatan pori-pori, pertumbuhan bakteri, dan peradangan.",
        "solution": "Jaga kebersihan wajah, gunakan produk non-komedogenik, dan hindari memencet jerawat.",
    },

    "Pustula": {
        "color": (240, 139, 78),
        "cause": "Pustula terbentuk ketika pori-pori tersumbat kemudian mengalami peradangan.",
        "solution": "Jangan memencet pustula, jaga kebersihan wajah, dan konsultasikan jika kondisinya semakin parah.",
    },

    "Nodules": {
        "color": (239, 90, 90),
        "cause": "Nodul terbentuk akibat penyumbatan dan peradangan yang terjadi pada lapisan kulit lebih dalam.",
        "solution": "Hindari memencet nodul dan lakukan pemeriksaan kepada tenaga kesehatan karena peradangannya lebih dalam.",
    },
}


def normalize_label(label: str) -> str:
    """Samakan nama kelas model dengan nama yang ditampilkan aplikasi."""
    cleaned = label.strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "jerawat blackheads": "Blackhead",
        "jerawat blackhead": "Blackhead",
        "blackheads": "Blackhead",
        "blackhead": "Blackhead",
        "whiteheads": "Whitehead",
        "whitehead": "Whitehead",
        "papula": "Papula",
        "papule": "Papula",
        "pustula": "Pustula",
        "pustule": "Pustula",
        "nodules": "Nodules",
        "nodule": "Nodules",
        "nodul": "Nodules",
    }
    return aliases.get(cleaned, label.strip().title())

def predict_yolo(
    model: YOLO,
    image: Image.Image,
    confidence: float = 0.25,
    iou: float = 0.45,
) -> list[dict]:
    """Jalankan inferensi dan ubah hasil YOLO menjadi format aplikasi."""
    rgb_image = image.convert("RGB")
    image_array = np.asarray(rgb_image)
    width, height = rgb_image.size

    results = model.predict(
        source=image_array,
        conf=float(confidence),
        iou=float(iou),
        imgsz=640,
        verbose=False,
    )

    detections: list[dict] = []
    if not results:
        return detections

    result = results[0]
    names = result.names
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].detach().cpu().tolist()
        class_id = int(box.cls[0].detach().cpu().item())
        score = float(box.conf[0].detach().cpu().item())
        label = normalize_label(str(names[class_id]))

        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))
        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            {
                "label": label,
                "confidence": score,
                "bbox": (
                    x1 / width,
                    y1 / height,
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                ),
            }
        )

    return detections


def draw_boxes(image: Image.Image, detections: list[dict]) -> Image.Image:
    """Gambar bounding box hasil model pada salinan gambar."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    image_width, image_height = annotated.size

    try:
        font = ImageFont.truetype(
            "DejaVuSans-Bold.ttf", size=max(13, image_width // 45)
        )
    except OSError:
        font = ImageFont.load_default()

    for detection in detections:
        x, y, box_width, box_height = detection["bbox"]
        color = ACNE_INFO.get(detection["label"], {"color": (37, 99, 235)})[
            "color"
        ]
        x1, y1 = x * image_width, y * image_height
        x2 = x1 + box_width * image_width
        y2 = y1 + box_height * image_height

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label_text = (
            f"{detection['label']} {detection['confidence'] * 100:.1f}%"
        )
        text_box = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_y = max(0, y1 - text_height - 8)
        draw.rectangle(
            [x1, label_y, min(image_width, x1 + text_width + 10), y1], fill=color
        )
        draw.text((x1 + 5, label_y + 2), label_text, fill=(15, 26, 43), font=font)

    return annotated


def summarize_detections(detections: list[dict]) -> dict:
    """Hitung jumlah, kelas dominan, dan confidence rata-rata."""
    if not detections:
        return {"count": 0, "dominant": "-", "avg_confidence": 0.0}

    tally: dict[str, int] = {}
    for detection in detections:
        label = detection["label"]
        tally[label] = tally.get(label, 0) + 1

    return {
        "count": len(detections),
        "dominant": max(tally, key=tally.get),
        "avg_confidence": sum(item["confidence"] for item in detections)
        / len(detections),
    }

def load_yolo_model(model_path: Path | str = MODEL_PATH) -> YOLO:
    """Muat bobot YOLO dan berikan pesan jelas bila file tidak ditemukan."""
    path = Path(model_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {path}. Letakkan model pada models/best.pt."
        )
    return YOLO(str(path))


def predict_yolo(
    model: YOLO,
    image: Image.Image,
    confidence: float = 0.25,
    iou: float = 0.45,
) -> list[dict]:
    """Jalankan inferensi dan ubah hasil YOLO menjadi format aplikasi."""
    rgb_image = image.convert("RGB")
    image_array = np.asarray(rgb_image)
    width, height = rgb_image.size

    results = model.predict(
        source=image_array,
        conf=float(confidence),
        iou=float(iou),
        imgsz=640,
        verbose=False,
    )

    detections: list[dict] = []
    if not results:
        return detections

    result = results[0]
    names = result.names
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].detach().cpu().tolist()
        class_id = int(box.cls[0].detach().cpu().item())
        score = float(box.conf[0].detach().cpu().item())
        label = normalize_label(str(names[class_id]))

        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))
        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            {
                "label": label,
                "confidence": score,
                "bbox": (
                    x1 / width,
                    y1 / height,
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                ),
            }
        )

    return detections


def draw_boxes(image: Image.Image, detections: list[dict]) -> Image.Image:
    """Gambar bounding box hasil model pada salinan gambar."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    image_width, image_height = annotated.size

    try:
        font = ImageFont.truetype(
            "DejaVuSans-Bold.ttf", size=max(13, image_width // 45)
        )
    except OSError:
        font = ImageFont.load_default()

    for detection in detections:
        x, y, box_width, box_height = detection["bbox"]
        color = ACNE_INFO.get(detection["label"], {"color": (37, 99, 235)})[
            "color"
        ]
        x1, y1 = x * image_width, y * image_height
        x2 = x1 + box_width * image_width
        y2 = y1 + box_height * image_height

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label_text = (
            f"{detection['label']} {detection['confidence'] * 100:.1f}%"
        )
        text_box = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_y = max(0, y1 - text_height - 8)
        draw.rectangle(
            [x1, label_y, min(image_width, x1 + text_width + 10), y1], fill=color
        )
        draw.text((x1 + 5, label_y + 2), label_text, fill=(15, 26, 43), font=font)

    return annotated


def summarize_detections(detections: list[dict]) -> dict:
    """Hitung jumlah, kelas dominan, dan confidence rata-rata."""
    if not detections:
        return {"count": 0, "dominant": "-", "avg_confidence": 0.0}

    tally: dict[str, int] = {}
    for detection in detections:
        label = detection["label"]
        tally[label] = tally.get(label, 0) + 1

    return {
        "count": len(detections),
        "dominant": max(tally, key=lambda label: tally[label]),
        "avg_confidence": sum(item["confidence"] for item in detections)
        / len(detections),
    }
