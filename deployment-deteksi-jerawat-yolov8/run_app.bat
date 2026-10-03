@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment belum ada. Jalankan setup_windows.bat terlebih dahulu.
  pause
  exit /b 1
)
if not exist "models\best.pt" (
  echo Model tidak ditemukan. Letakkan file pada models\best.pt.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app.py
pause
