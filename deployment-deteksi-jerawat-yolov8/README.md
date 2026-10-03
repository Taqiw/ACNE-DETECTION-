# Deployment Sistem Deteksi Jenis Jerawat YOLOv8

Paket ini menjalankan inferensi YOLOv8 asli melalui Streamlit. Deteksi acak
yang terdapat pada versi mockup sebelumnya sudah dihapus.

## Struktur penting

```text
deployment-deteksi-jerawat-yolov8/
├── .streamlit/config.toml
├── models/best.pt
├── app.py
├── utils.py
├── check_model.py
├── requirements.txt
├── setup_windows.bat
└── run_app.bat
```

## Cara termudah di Windows

1. Ekstrak ZIP ini.
2. Buka folder `deployment-deteksi-jerawat-yolov8` di VS Code melalui
   **File > Open Folder**. Jangan membuka folder induknya.
3. Pastikan Python 3.11 sudah terpasang.
4. Klik dua kali `setup_windows.bat` dan tunggu instalasi selesai.
5. Klik dua kali `run_app.bat`.
6. Browser akan membuka `http://localhost:8501`.

Skrip BAT menggunakan `.venv\\Scripts\\python.exe` secara langsung, sehingga
tidak memerlukan aktivasi PowerShell dan tidak terkena galat Execution Policy.

## Cara melalui terminal VS Code

Buka **Terminal > New Terminal**, lalu jalankan satu per satu dari folder proyek:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_model.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Jangan menjalankan `python -m streamlit` bila terminal menunjukkan
`Python was not found`. Gunakan interpreter yang ada di `.venv` seperti contoh.

## Memasang model terbaru

1. Unduh `best.pt` dari folder `weights` hasil pelatihan terbaik di Colab.
2. Ganti file `models/best.pt` yang ada di proyek ini.
3. Pastikan namanya tepat `best.pt`, bukan `best (1).pt`.
4. Jalankan `check_model.py`; terminal harus menampilkan `MODEL TERBACA` dan
   daftar lima kelas.
5. Jalankan kembali aplikasi.

Catatan: model dalam ZIP ini berasal dari file model yang telah tersedia pada
percakapan sebelumnya. Jika bobot terbaik eksperimen terbaru Anda berbeda,
gantikan model tersebut sebelum menjadikan hasil deployment sebagai hasil akhir.

## Bila sebagian objek tidak terdeteksi

- Mulai dari Confidence `0.25`.
- Turunkan ke `0.20` atau `0.15` untuk mengecek prediksi lemah.
- Gunakan gambar yang tajam dan pencahayaan merata.
- Jangan memakai filter posisi atau ukuran bounding box tambahan; versi ini
  menampilkan keluaran model asli setelah NMS.
- Bila hasil tetap tidak tepat, perbaikannya harus dilakukan pada dataset,
  anotasi, pembagian data, atau pelatihan model—bukan dengan memindahkan kotak
  secara manual di Streamlit.

## Catatan deployment daring

Untuk Streamlit Community Cloud, unggah seluruh isi proyek ke repositori GitHub.
Karena file bobot dapat berukuran besar, pastikan ukuran `best.pt` masih memenuhi
batas repositori/platform. Isi perintah aplikasi adalah `app.py`.

Hasil sistem digunakan untuk penelitian/edukasi dan bukan diagnosis medis.
