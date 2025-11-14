# Installation Guide

## Python Version Compatibility

**Recommended:** Python 3.8, 3.9, 3.10, or 3.11

MediaPipe has limited support for Python 3.12+ and may not work with Python 3.13.

## Installation Steps

### Option 1: Standard Installation (Python 3.8-3.11)

```bash
cd sign_language_translator
pip install -r requirements.txt
```

### Option 2: If MediaPipe Installation Fails (Python 3.12+)

If you're using Python 3.12 or 3.13, try installing MediaPipe separately:

```bash
# Install other dependencies first
pip install tensorflow opencv-python numpy flask scikit-learn Pillow python-dateutil

# Try installing MediaPipe (may require pre-release version)
pip install mediapipe

# If that fails, try:
pip install --upgrade pip
pip install mediapipe --pre

# Or use conda (if you have Anaconda/Miniconda):
conda install -c conda-forge mediapipe
```

### Option 3: Use Python 3.11 (Recommended)

If you have multiple Python versions, use Python 3.11:

```bash
# On Windows with py launcher:
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# On Linux/macOS:
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Optional: Text-to-Speech

Install TTS packages separately if needed:

```bash
# Option 1: pyttsx3 (offline)
pip install pyttsx3

# Option 2: gTTS (online, requires internet)
pip install gtts pygame
```

## Verify Installation

Run the setup script to verify:

```bash
python setup.py
```

Or test imports manually:

```python
python -c "import tensorflow; import cv2; import mediapipe; import flask; print('All packages installed!')"
```

## Troubleshooting

### MediaPipe Installation Issues

1. **Python 3.13 not supported**: Downgrade to Python 3.11 or use conda
2. **Windows issues**: Try installing Visual C++ Redistributable
3. **macOS M1/M2**: Use conda-forge version

### TensorFlow Issues

- If TensorFlow fails, try: `pip install tensorflow-cpu` (CPU-only version)
- For GPU support: `pip install tensorflow[and-cuda]`

### OpenCV Issues

- If opencv-python fails, try: `pip install opencv-contrib-python`

## Alternative: Use Conda

If pip installation fails, use conda:

```bash
conda create -n signlang python=3.11
conda activate signlang
conda install -c conda-forge tensorflow opencv mediapipe flask scikit-learn pillow
pip install pyttsx3 gtts pygame
```

