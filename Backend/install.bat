@echo off
echo [maya] Setting up Python 3.11 virtual environment...
py -3.11 -m venv .venv
call .venv\Scripts\activate

echo [1/2] Installing PyTorch with CUDA 12.8 (RTX 5070 Ti)...
pip install torch --index-url https://download.pytorch.org/whl/cu128

echo [2/2] Installing MAYA backend dependencies...
pip install -r requirements.txt

echo.
echo Done. Activate with: .venv\Scripts\activate
echo Run with: python server.py
