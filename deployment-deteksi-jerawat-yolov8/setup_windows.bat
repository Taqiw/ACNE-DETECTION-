@echo off
cd /d "%~dp0"
echo Membuat virtual environment Python 3.11...
py -3.11 -m venv .venv
if errorlevel 1 goto :error

echo Memasang dependensi...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Instalasi selesai. Jalankan run_app.bat.
pause
exit /b 0

:error
echo.
echo Instalasi gagal. Pastikan Python 3.11 telah dipasang dan centang Add Python to PATH.
pause
exit /b 1
