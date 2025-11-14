# AI Sign Language Translator

A complete AI-powered Sign Language Translator that detects and translates sign language gestures into text using TensorFlow, OpenCV, and MediaPipe. The application runs as a web application via Flask on localhost.

## 🎯 Features

- **Real-time Sign Detection**: Uses webcam to detect sign language gestures in real-time
- **Two Model Approaches**: 
  - Image-based CNN for direct image classification
  - Landmark-based MLP using MediaPipe hand landmarks
- **Web Interface**: Beautiful, responsive web UI built with Flask and Bootstrap
- **Text-to-Speech**: Optional speech output for detected signs
- **Sentence Builder**: Build sentences by adding detected words
- **Confidence Scores**: Display prediction confidence in real-time
- **FPS Monitoring**: Real-time frame rate display

## 📋 Requirements

- Python 3.8 or higher
- Webcam
- Windows/Linux/macOS

## 🚀 Installation

1. **Clone or download this repository**

2. **Navigate to the project directory**
   ```bash
   cd sign_language_translator
   ```

3. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/macOS:
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📚 Usage Guide

### Step 1: Collect Training Data

Before training a model, you need to collect sign language gesture data:

```bash
cd src
python collect_data.py
```

**Instructions:**
- Press `n` to start collecting data for a new label (e.g., "A", "B", "Hello", "Thanks")
- Press `SPACE` to capture an image
- Press `q` to finish current label and move to next
- Press `ESC` to exit

The script will:
- Save images in `dataset/{label}/` folders
- Extract and save hand landmarks in `dataset/{label}_landmarks/` folders

**Recommended:** Collect at least 50-100 images per sign for better accuracy.

### (Optional) Step 1.5: Augment the Dataset

Boost the amount of training data by synthesizing new images with random rotations, shifts, brightness adjustments, and flips:

```bash
cd src
python augment_dataset.py --augmentations_per_image 4 --copy_original
```

- `--augmentations_per_image`: how many synthetic variants to create per source image (default: 3)
- `--copy_original`: include the original images alongside the augmented ones in the output directory
- `--output_dir`: change where augmented images are written (default: `../dataset_augmented`)

Once complete, you can point `preprocess_data.py` at the augmented folder (e.g., pass `../dataset_augmented` when prompted) or merge the augmented images back into `dataset/`.

### (Optional) Step 1.6: Download Images from the Web

Automatically fetch additional images per label to enlarge the dataset:

```bash
cd src
# Option A: default queries per letter
python download_images.py --labels A B C --per_label 150

# Option B: custom queries for specific labels
python download_images.py --labels A --query_map A:"ASL letter A hand sign" --per_label 200
```

If a backend package is missing, install one of:
```bash
pip install simple_image_download
# or
pip install duckduckgo-search requests
```
Images are saved under `dataset/<LABEL>`. Review and delete incorrect images before preprocessing.

### Step 2: Preprocess Data

Preprocess the collected data for training:

```bash
python preprocess_data.py
```

Choose:
- **Option 1**: Image-based preprocessing (for CNN)
- **Option 2**: Landmark-based preprocessing (for MLP)

This will create preprocessed data files in the `model/` directory.

### Step 3: Train the Model

Train your sign language recognition model:

```bash
python train_model.py
```

The script will:
- Automatically detect whether to use CNN or MLP based on data type
- Train the model with data augmentation (for images)
- Save the best model to `model/sign_model.h5`
- Save label encoder to `model/label_encoder.pkl`

**Training Tips:**
- More data = better accuracy
- Training may take 10-30 minutes depending on dataset size
- GPU acceleration is automatically used if available

### Step 4: Run the Web Application

Start the Flask web server:

```bash
cd ..
python app.py
```

The server will start on `http://127.0.0.1:5000`

### Step 5: Use the Web Interface

1. Open your browser and navigate to `http://127.0.0.1:5000`
2. Click **"Start Camera"** to begin video feed
3. Show sign language gestures to your webcam
4. The detected sign will appear above the video feed
5. Use **"Add Word"** to build sentences
6. Click **"Speak Output"** to hear the current prediction (requires TTS setup)

## 📁 Project Structure

```
sign_language_translator/
├── app.py                      # Flask main application
├── model/                      # Trained models and preprocessed data
│   ├── sign_model.h5          # Trained TensorFlow model
│   └── label_encoder.pkl      # Label encoder for predictions
├── src/                        # Source code
│   ├── collect_data.py        # Data collection script
│   ├── preprocess_data.py     # Data preprocessing
│   ├── train_model.py         # Model training
│   └── predict.py             # Prediction module
├── templates/                  # HTML templates
│   └── index.html             # Main web interface
├── static/                     # Static files
│   ├── css/
│   │   └── style.css          # Custom styling
│   └── js/
│       └── script.js          # Frontend JavaScript
├── dataset/                    # Training data (created during collection)
│   ├── A/                     # Images for sign "A"
│   ├── B/                     # Images for sign "B"
│   └── ...                    # More labels
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Configuration

### Model Settings

You can modify model architecture in `src/train_model.py`:
- CNN layers and filters
- MLP layer sizes
- Learning rate and optimizer
- Training epochs

### Camera Settings

Camera index can be changed in `app.py` (default: 0):
```python
camera = cv2.VideoCapture(0)  # Change 0 to 1, 2, etc. for different cameras
```

## 🎤 Text-to-Speech Setup

The application supports two TTS options:

### Option 1: pyttsx3 (Offline)
```bash
pip install pyttsx3
```
Works offline, uses system TTS engine.

### Option 2: gTTS (Online)
```bash
pip install gtts pygame
```
Requires internet connection, better quality voices.

## 🚀 Deployment

### Local Deployment (Current)
The app runs on `http://127.0.0.1:5000` by default.

### Deploy to Render

1. **Create a `render.yaml` file:**
   ```yaml
   services:
     - type: web
       name: sign-language-translator
       env: python
       buildCommand: pip install -r requirements.txt
       startCommand: python app.py
   ```

2. **Update `app.py` for production:**
   ```python
   if __name__ == '__main__':
       app.run(host='0.0.0.0', port=5000, debug=False)
   ```

3. **Push to GitHub and connect to Render**

**Note:** Webcam access requires HTTPS and user permission. Consider using WebRTC for browser-based camera access in production.

### Deploy to Vercel

Vercel is primarily for static sites. For Flask apps, consider:
- **Heroku** (free tier available)
- **Railway** (easy deployment)
- **PythonAnywhere** (Python-focused hosting)

### Alternative: Streamlit Version

For easier deployment, you can convert to Streamlit:

```python
import streamlit as st
import cv2
from predict import SignLanguagePredictor

st.title("Sign Language Translator")
predictor = SignLanguagePredictor()

# Streamlit camera input
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    prediction, confidence, _ = predictor.predict(cv2_img)
    st.write(f"Prediction: {prediction} ({confidence:.2%})")
```

## 🐛 Troubleshooting

### Camera Not Working
- Check camera permissions
- Try different camera index (0, 1, 2, etc.)
- Ensure no other application is using the camera

### Model Not Found
- Ensure you've completed Steps 1-3 (collect, preprocess, train)
- Check that `model/sign_model.h5` exists

### Low Accuracy
- Collect more training data (aim for 100+ images per sign)
- Ensure consistent lighting and background
- Try data augmentation (already enabled for images)
- Adjust MediaPipe confidence thresholds in `predict.py`

### Import Errors
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Check Python version (3.8+ required)

## 📝 Notes

- **Training Data**: The more diverse your training data, the better the model will perform
- **Lighting**: Consistent lighting improves detection accuracy
- **Background**: Plain backgrounds work best
- **Hand Position**: Keep hands clearly visible in frame
- **Model Size**: CNN models are larger but may be more accurate; MLP models are faster

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is open source and available for educational and personal use.

## 🙏 Acknowledgments

- **MediaPipe** by Google for hand tracking
- **TensorFlow** for deep learning framework
- **OpenCV** for computer vision
- **Flask** for web framework

## 📧 Support

For issues or questions, please open an issue on the repository.

---

**Happy Sign Language Translating!** 👐

