@echo off
REM Creates a Python venv for ComfyUI and installs its requirements.
REM Run this once if ComfyUI wasn't installed through Stability Matrix,
REM or if auto-start fails because the Python executable isn't found.

set COMFYUI_DIR=C:\Users\tyler\Downloads\Data\Packages\ComfyUI
set VENV_DIR=%COMFYUI_DIR%\venv

echo === ComfyUI venv setup ===
echo ComfyUI dir: %COMFYUI_DIR%
echo Venv dir:    %VENV_DIR%
echo.

if not exist "%COMFYUI_DIR%\main.py" (
    echo ERROR: ComfyUI not found at %COMFYUI_DIR%
    echo Edit COMFYUI_DIR in this script to match your install path.
    pause
    exit /b 1
)

if exist "%VENV_DIR%\Scripts\python.exe" (
    echo Venv already exists — reinstalling requirements.
) else (
    echo Creating venv...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create venv. Is Python 3.10/3.11 installed?
        pause
        exit /b 1
    )
)

echo.
echo Installing PyTorch ^(CUDA 12.1^)...
"%VENV_DIR%\Scripts\pip.exe" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo Installing ComfyUI requirements...
"%VENV_DIR%\Scripts\pip.exe" install -r "%COMFYUI_DIR%\requirements.txt"

echo.
echo === Done! ComfyUI venv is ready. ===
echo Maya will now auto-start ComfyUI when generating images.
pause
