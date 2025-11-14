# Quick Start Guide

## 🚀 Fast Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Setup
```bash
python setup.py
```

### 3. Collect Training Data (First Time Only)
```bash
cd src
python collect_data.py
```
- Press `n` to start a new label
- Press `SPACE` to capture images
- Collect at least 20-30 images per sign
- Press `q` to finish current label
- Press `ESC` to exit

### 4. Preprocess Data
```bash
python preprocess_data.py
```
- Choose option 1 (Image-based) or 2 (Landmark-based)

### 5. Train Model
```bash
python train_model.py
```
- Wait for training to complete (5-30 minutes)

### 6. Run Web App
```bash
cd ..
python app.py
```

### 7. Open Browser
Navigate to: `http://127.0.0.1:5000`

## 📝 Quick Tips

- **Minimum Data**: 20-30 images per sign (more = better accuracy)
- **Good Lighting**: Ensure consistent lighting when collecting data
- **Clear Background**: Plain backgrounds work best
- **Hand Visibility**: Keep hands clearly visible in frame

## 🎯 Example Workflow

1. Collect data for signs: "Hello", "Thanks", "Yes", "No"
2. Preprocess → Train → Run app
3. Show signs to webcam and see predictions!

## ⚠️ Troubleshooting

**Camera not working?**
- Check camera permissions
- Try different camera index in `app.py`

**Model not found?**
- Make sure you completed steps 3-5 (collect, preprocess, train)

**Low accuracy?**
- Collect more training data
- Ensure consistent lighting and background

---

For detailed instructions, see [README.md](README.md)

