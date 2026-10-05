@echo off
REM Humanizer launcher for Windows. Double-click. First run installs everything (free).
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>nul
if errorlevel 1 (
  echo Python 3.10 or newer is required ^(free^): https://www.python.org/downloads/
  echo During install, tick "Add python.exe to PATH".
  pause & exit /b 1
)
if not exist .venv\Scripts\python.exe (
  echo First run: setting up ^(this takes a few minutes^)...
  %PY% -m venv .venv || (pause & exit /b 1)
  .venv\Scripts\python -m pip install --upgrade pip -q
)
if not exist .venv\.installed (
  REM NVIDIA GPU present: install the CUDA build of PyTorch so detectors run on the GPU
  where nvidia-smi >nul 2>nul && (
    echo NVIDIA GPU found, installing GPU version of PyTorch...
    .venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu130 -q
  )
  .venv\Scripts\python -m pip install -r requirements.txt -q || (pause & exit /b 1)
  type nul > .venv\.installed
)
where ollama >nul 2>nul && (
  curl -s http://127.0.0.1:11434/api/tags >nul 2>nul || (start "" /min ollama serve & timeout /t 3 >nul)
)
.venv\Scripts\python setup_models.py
.venv\Scripts\python app.py
pause
