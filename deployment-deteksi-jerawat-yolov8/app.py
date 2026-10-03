
import tempfile
import threading
import time
from html import escape
from pathlib import Path

import av
import cv2
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

def load_css():
    with open("Assets/CSS/style.css", "r", encoding="utf-8") as file:
        st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)

from utils import (
    ACNE_INFO,
    MODEL_PATH,
    draw_boxes,
    load_yolo_model,
    predict_yolo,
    summarize_detections,
)


st.set_page_config(
    page_title="Detetksi Jerawat",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent

CSS_PATHS = [
    APP_DIR / "style.css",
    APP_DIR.parent / "Assets" / "CSS" / "style.css",
]

style_path = next(
    (path for path in CSS_PATHS if path.is_file()),
    None,
)

if style_path is not None:
    css_content = style_path.read_text(encoding="utf-8")
    st.html(f"<style>{css_content}</style>")
else:
    st.warning("File style.css tidak ditemukan.")

st.markdown(
    """
    <style>
    .main .block-container {padding-top:1.6rem;max-width:1180px}
    section[data-testid="stSidebar"] {background:linear-gradient(180deg,#123B8C,#0B2A66)}
    section[data-testid="stSidebar"] * {color:#fff!important}
    section[data-testid="stSidebar"] .stRadio>label {display:none}
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
      background:rgba(255,255,255,.07);border-radius:10px;padding:10px 12px;margin-bottom:6px
    }
    .card,.stat-box {background:#fff;border:1px solid #E2E8F0;border-radius:14px;
      padding:16px 18px;box-shadow:0 1px 2px rgba(16,25,43,.05);margin-bottom:16px}
    .stat-value {font-size:24px;font-weight:800;color:#10192B}
    .stat-label,.app-header-desc {color:#64748B;font-size:13px}
    .summary-banner {background:#E2F6F3;border:1px solid #BFE8E2;border-radius:12px;
      padding:14px 16px;font-size:13px;color:#0F5952;margin-top:10px}
    h1,h2,h3 {color:#10192B}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Memuat model YOLOv8...")
def get_model():
    return load_yolo_model(MODEL_PATH)


def try_get_model():
    try:
        return get_model(), None
    except Exception as exc:
        return None, str(exc)


with st.sidebar:
    st.markdown("### 🔬 Acne Scan")
    st.caption("Pendeteksi Jerawat")
    st.markdown("---")
    page = st.radio(
        "Menu",
        ["🏠 Home", "🖼️ Deteksi Gambar", "🎬 Deteksi Video", "📷 Kamera", "ℹ️ Tentang"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Pengaturan model**")
    confidence_threshold = st.slider(
        "Confidence", min_value=0.05, max_value=0.90, value=0.25, step=0.05,
        help="Turunkan bila objek sulit terdeteksi; naikkan bila terlalu banyak deteksi keliru.",
    )
    iou_threshold = st.slider(
        "IoU / NMS", min_value=0.10, max_value=0.90, value=0.45, step=0.05,
        help="Mengatur penggabungan kotak yang saling tumpang tindih.",
    )
    model, model_error = try_get_model()
    if model is not None:
        st.success("Model models/best.pt siap")
    else:
        st.error("Model belum dapat dimuat")
        st.caption(model_error)


def run_detection(image: Image.Image):
    if model is None:
        st.error(model_error or "Model belum tersedia.")
        return [], image.convert("RGB")
    detections = predict_yolo(
        model,
        image,
        confidence=confidence_threshold,
        iou=iou_threshold,
    )
    return detections, draw_boxes(image, detections)


def render_detection_info(detections: list[dict]):
    st.markdown("#### Informasi Deteksi")

    if not detections:
        st.info(
            "Tidak ada objek yang melewati ambang confidence. "
            "Gunakan gambar yang jelas atau turunkan Confidence "
            "secara bertahap, misalnya ke 0.20 atau 0.15."
        )
        return

    summary = summarize_detections(detections)

    # Menampilkan label dan confidence
    for detection in detections:
        label = detection["label"]
        percent = detection["confidence"] * 100

        color = ACNE_INFO.get(
            label,
            {"color": (37, 99, 235)},
        )["color"]

        col1, col2, col3 = st.columns([1.3, 4, 1])

        with col1:
            st.markdown(
                f"""
                <span style="
                    display:inline-block;
                    width:10px;
                    height:10px;
                    border-radius:3px;
                    background:rgb{color};
                    margin-right:6px;
                "></span>
                <b>{escape(label)}</b>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.progress(min(int(percent), 100))

        with col3:
            st.markdown(f"**{percent:.1f}%**")

    # Ringkasan
    st.markdown(
        f"""
        <div class="summary-banner">
            <b>Ringkasan:</b>
            ditemukan <b>{summary['count']}</b> objek,
            kelas dominan <b>{escape(summary['dominant'])}</b>,
            dan confidence rata-rata
            <b>{summary['avg_confidence'] * 100:.1f}%</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Mengambil kelas unik
    detected_labels = list(
        dict.fromkeys(
            detection["label"]
            for detection in detections
        )
    )

    st.markdown("### Penyebab dan Solusi")

    for label in detected_labels:
        info = ACNE_INFO.get(label)

        if info is None:
            st.warning(
                f"Informasi untuk kelas '{label}' belum tersedia."
            )
            continue

        cause = info.get("cause") or info.get("penyebab")
        solution = info.get("solution") or info.get("solusi")

        if not cause or not solution:
            st.warning(
                f"Penyebab atau solusi untuk kelas "
                f"'{label}' belum tersedia."
            )
            continue

        label_safe = escape(str(label))
        cause_safe = escape(str(cause))
        solution_safe = escape(str(solution))

        bubble_html = f"""
<div class="acne-guidance-card">
    <div class="acne-label-bubble">
        🔎 {label_safe}
    </div>

    <div class="bubble-grid">
        <div class="info-bubble cause-bubble">
            <div class="bubble-heading">
                🔍 Kemungkinan Penyebab
            </div>

            <div class="bubble-description">
                {cause_safe}
            </div>
        </div>

        <div class="info-bubble solution-bubble">
            <div class="bubble-heading">
                💡 Saran Penanganan
            </div>

            <div class="bubble-description">
                {solution_safe}
            </div>
        </div>
    </div>
</div>
"""

        st.html(bubble_html)

    st.info(
        "Informasi ini bersifat edukasi dan bukan "
        "pengganti diagnosis tenaga medis.",
        icon="ℹ️",
    )

MODEL_LOCK = threading.Lock()
CAMERA_RESULT_LOCK = threading.Lock()
CAMERA_RESULT = {"detections": []}


class AcneRealtimeProcessor(VideoProcessorBase):
    def __init__(self, yolo_model, confidence, iou):
        self.model = yolo_model
        self.confidence = confidence
        self.iou = iou
        self.frame_number = 0
        self.last_result = None

        # Menyimpan informasi deteksi terakhir
        self.latest_detections = []
        self.detection_lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        frame_bgr = frame.to_ndarray(format="bgr24")
        self.frame_number += 1

        # Proses setiap 3 frame agar kamera lebih ringan
        if self.frame_number % 3 != 0 and self.last_result is not None:
            return av.VideoFrame.from_ndarray(
                self.last_result,
                format="bgr24",
            )

        if self.model is None:
            cv2.putText(
                frame_bgr,
                "Model belum tersedia",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            return av.VideoFrame.from_ndarray(
                frame_bgr,
                format="bgr24",
            )

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )
        image = Image.fromarray(frame_rgb)

        try:
            with MODEL_LOCK:
                detections = predict_yolo(
                    self.model,
                    image,
                    confidence=self.confidence,
                    iou=self.iou,
                )

            with CAMERA_RESULT_LOCK:
                CAMERA_RESULT["detections"] = [
                     detection.copy()
                    for detection in detections
                ]
            # Simpan hasil agar dapat ditampilkan di bawah kamera
            with self.detection_lock:
                self.latest_detections = [
                    detection.copy()
                    for detection in detections
                ]

            annotated = draw_boxes(image, detections)

            result_rgb = np.asarray(annotated)
            result_bgr = cv2.cvtColor(
                result_rgb,
                cv2.COLOR_RGB2BGR,
            )

            cv2.putText(
                result_bgr,
                f"Objek terdeteksi: {len(detections)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (50, 220, 120),
                2,
            )

            self.last_result = result_bgr

        except Exception as error:
            self.last_result = frame_bgr.copy()

            cv2.putText(
                self.last_result,
                f"Kesalahan inferensi: {str(error)[:50]}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )

        return av.VideoFrame.from_ndarray(
            self.last_result,
            format="bgr24",
        )

    def get_latest_detections(self) -> list[dict]:
        """Ambil salinan hasil deteksi terbaru secara aman."""
        with self.detection_lock:
            return [
                detection.copy()
                for detection in self.latest_detections
            ]

def page_home():
    st.title("Sistem Deteksi Jerawat ")
    st.markdown(
        "<p class='app-header-desc'>Deployment Streamlit untuk menjalankan model "
        "YOLOv8 lokal pada gambar, cuplikan video, atau foto kamera.</p>",
        unsafe_allow_html=True,
    )
    status = "Siap" if model is not None else "Belum siap"
    values = [
        ("Kelas model", "5"),
        ("Confidence awal", f"{confidence_threshold:.2f}"),
        ("Status model", status),
        ("Mode", "Lokal"),
    ]
    for column, (label, value) in zip(st.columns(4), values):
        with column:
            st.markdown(
                f"<div class='stat-box'><div class='stat-value'>{value}</div>"
                f"<div class='stat-label'>{label}</div></div>",
                unsafe_allow_html=True,
            )
    st.markdown("#### Cara mencoba")
    st.write("1. Buka **Deteksi Gambar** lalu unggah JPG/PNG yang pencahayaannya jelas.")
    st.write("2. Lihat kotak, nama kelas, dan confidence hasil model.")
    st.write("3. Bila tidak terdeteksi, turunkan Confidence sedikit demi sedikit.")
    st.warning(
        "Hasil model merupakan keluaran penelitian/edukasi dan bukan diagnosis medis."
    )


def page_image():
    st.title("Deteksi Gambar")
    uploaded = st.file_uploader("Unggah JPG/PNG", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Pilih gambar untuk memulai deteksi.")
        return

    image = Image.open(uploaded).convert("RGB")
    with st.spinner("Menjalankan inferensi YOLOv8..."):
        detections, annotated = run_detection(image)

    left, right = st.columns(2)
    with left:
        st.markdown("**Gambar asli**")
        st.image(image, width="stretch")
    with right:
        st.markdown(f"**Hasil deteksi — {len(detections)} objek**")
        st.image(annotated, width="stretch")
    render_detection_info(detections)


def page_video():
    st.title("Deteksi Video")
    st.caption("Aplikasi menganalisis satu frame representatif agar deployment tetap ringan.")
    uploaded = st.file_uploader("Unggah MP4/AVI/MOV", type=["mp4", "avi", "mov"])
    if uploaded is None:
        st.info("Pilih video untuk memulai deteksi.")
        return

    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded.getbuffer())
        temp_path = temp_file.name

    left, right = st.columns(2)
    with left:
        st.markdown("**Video asli**")
        st.video(temp_path)

    capture = cv2.VideoCapture(temp_path)
    frame_count = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_count * 0.4))
    ok, frame = capture.read()
    capture.release()

    with right:
        st.markdown("**Hasil pada cuplikan frame**")
        if not ok:
            st.error("Frame video tidak dapat dibaca.")
            return
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections, annotated = run_detection(image)
        st.image(annotated, width="stretch")
    render_detection_info(detections)


def page_camera():
    st.title("Deteksi Kamera Real-time")
    st.caption(
        "Aktifkan kamera untuk mendeteksi jenis jerawat secara langsung."
    )

    if model is None:
        st.error(model_error or "Model YOLO belum tersedia.")
        return

    st.info(
        "Tekan START, lalu izinkan browser menggunakan kamera.",
        icon="📷",
    )

    camera_context = webrtc_streamer(
        key="acne-realtime-camera",
        video_processor_factory=lambda: AcneRealtimeProcessor(
            yolo_model=model,
            confidence=confidence_threshold,
            iou=iou_threshold,
        ),
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "facingMode": "user",
            },
            "audio": False,
        },
        async_processing=False,
    )

    information_placeholder = st.empty()

    while camera_context.state.playing:
        with CAMERA_RESULT_LOCK:
            detections = [
                detection.copy()
                for detection in CAMERA_RESULT["detections"]
            ]

        with information_placeholder.container():
            if detections:
                render_detection_info(detections)
            else:
                st.info(
                    "Belum ada objek yang melewati ambang confidence. "
                    "Arahkan kamera ke area wajah dengan pencahayaan jelas."
                )

        time.sleep(0.8)
        
def page_about():
    st.title("Tentang Sistem")
    st.markdown(
        """
        <div class="card"><b>Metode</b><p class="app-header-desc">
        Sistem menggunakan Ultralytics YOLOv8 dan bobot lokal <code>models/best.pt</code>.
        Lima kelas target adalah Blackhead, Whitehead, Papula, Pustula, dan Nodules.
        </p></div>
        <div class="card"><b>Catatan penelitian</b><p class="app-header-desc">
        Nilai confidence menunjukkan keyakinan model, bukan tingkat keparahan kondisi.
        Ketepatan deployment bergantung pada kualitas anotasi, distribusi dataset,
        hasil pelatihan, pencahayaan, jarak kamera, serta kemiripan data uji dengan data latih.
        </p></div>
        """,
        unsafe_allow_html=True,
    )


if page == "🏠 Home":
    page_home()
elif page == "🖼️ Deteksi Gambar":
    page_image()
elif page == "🎬 Deteksi Video":
    page_video()
elif page == "📷 Kamera":
    page_camera()
else:
    page_about()
