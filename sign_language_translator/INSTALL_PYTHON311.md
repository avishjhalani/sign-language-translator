# Python Version Issue - MediaPipe Compatibility

## Problem
You're using **Python 3.13.2**, but **MediaPipe** (required for hand tracking) only supports **Python 3.8-3.11**.

## Solution Options

### Option 1: Install Python 3.11 (Recommended)

1. **Download Python 3.11:**
   - Visit: https://www.python.org/downloads/release/python-3110/
   - Download Windows installer (64-bit)

2. **Install Python 3.11:**
   - Run the installer
   - ✅ Check "Add Python to PATH"
   - Choose "Install Now" or "Customize installation"

3. **Verify Installation:**
   ```bash
   python3.11 --version
   ```

4. **Create Virtual Environment with Python 3.11:**
   ```bash
   python3.11 -m venv venv
   venv\Scripts\activate
   ```

5. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Use pyenv (Advanced)

If you have `pyenv` installed:
```bash
pyenv install 3.11.0
pyenv local 3.11.0
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Option 3: Continue Without MediaPipe (Limited Functionality)

If you must use Python 3.13, you can install other dependencies but **hand tracking won't work**:

```bash
pip install -r requirements_py313.txt
```

**Note:** Without MediaPipe, you can only use image-based CNN models, not landmark-based MLP models.

## Quick Check

Run this to check your Python version:
```bash
python --version
```

You need: **Python 3.8, 3.9, 3.10, or 3.11**

---

**Recommendation:** Use Python 3.11 for best compatibility with all dependencies.

