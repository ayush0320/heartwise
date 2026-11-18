@echo off
REM Double-click to install and start HeartWise on Windows (needs Python 3.12 on PATH).
cd /d "%~dp0"
if not exist venv (
  echo Creating a virtual environment...
  python -m venv venv
)
call venv\Scripts\activate
echo Installing packages (first time only, a few minutes)...
pip install -r requirements.txt
echo.
echo Starting HeartWise. The first run trains the model (about a minute).
python app.py
pause
