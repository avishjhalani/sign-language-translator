"""
Flask Web Application for Sign Language Translator
Main application file that handles webcam streaming and real-time predictions.
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import sys
import os
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from predict import SignLanguagePredictor

app = Flask(__name__)

# Initialize predictor
predictor = None
camera = None
camera_active = False
last_prediction = None
last_confidence = 0.0
fps = 0
frame_count = 0
start_time = time.time()

def init_camera():
    """Initialize camera."""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            camera = cv2.VideoCapture(1)  # Try second camera
    return camera

def init_predictor():
    """Initialize sign language predictor."""
    global predictor
    if predictor is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'model', 'sign_model.h5')
        label_encoder_path = os.path.join(base_dir, 'model', 'label_encoder.pkl')
        predictor = SignLanguagePredictor(model_path, label_encoder_path)
    return predictor

def generate_frames():
    """Generate video frames with predictions."""
    global camera, predictor, camera_active, last_prediction, last_confidence, fps, frame_count, start_time
    MIN_CONFIDENCE = 0.4
    
    camera = init_camera()
    if camera is None or not camera.isOpened():
        return
    
    predictor = init_predictor()
    
    while camera_active:
        success, frame = camera.read()
        if not success:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
        
        # Get hand landmarks
        if predictor.input_type == 'landmark':
            landmarks, hand_landmarks = predictor.extract_landmarks(frame)
            if hand_landmarks:
                frame = predictor.draw_landmarks(frame, hand_landmarks)
        
        # Predict sign (every 5 frames to reduce computation)
        if frame_count % 5 == 0:
            predicted_label, confidence, _ = predictor.predict(frame)
            if predicted_label and confidence >= MIN_CONFIDENCE:
                last_prediction = predicted_label
                last_confidence = confidence
            else:
                # Reset when low confidence or no hand detected to avoid sticky stale label
                last_prediction = None
                last_confidence = 0.0
        
        # Draw prediction on frame
        if last_prediction:
            # Background rectangle for text
            cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (400, 100), (0, 255, 0), 2)
            
            # Prediction text
            text = f"Sign: {last_prediction}"
            confidence_text = f"Confidence: {last_confidence:.2%}"
            fps_text = f"FPS: {fps:.1f}"
            
            cv2.putText(frame, text, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, confidence_text, (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, fps_text, (20, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # No prediction message
            cv2.putText(frame, "No hand detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        
        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    # Cleanup
    if camera:
        camera.release()

@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start camera streaming."""
    global camera_active, frame_count, start_time
    camera_active = True
    frame_count = 0
    start_time = time.time()
    return jsonify({'status': 'success', 'message': 'Camera started'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera streaming."""
    global camera_active, camera
    camera_active = False
    if camera:
        camera.release()
        camera = None
    return jsonify({'status': 'success', 'message': 'Camera stopped'})

@app.route('/get_prediction', methods=['GET'])
def get_prediction():
    """Get current prediction."""
    global last_prediction, last_confidence
    return jsonify({
        'prediction': last_prediction or 'None',
        'confidence': last_confidence
    })

@app.route('/debug_prediction', methods=['GET'])
def debug_prediction():
    """Return a one-off prediction with raw probabilities and classes for debugging."""
    global predictor, camera
    if predictor is None:
        predictor = init_predictor()
    cam = init_camera()
    if cam is None or not cam.isOpened():
        return jsonify({'error': 'Camera not available'}), 500

    success, frame = cam.read()
    if not success:
        return jsonify({'error': 'Failed to read frame'}), 500

    frame = cv2.flip(frame, 1)
    label, confidence, probs = predictor.predict(frame)

    classes = list(predictor.label_encoder.classes_) if predictor.label_encoder else []
    top = []
    if probs is not None and classes:
        prob_list = probs.tolist()
        top_indices = sorted(range(len(prob_list)), key=lambda i: prob_list[i], reverse=True)[:5]
        top = [{'label': classes[i], 'prob': float(prob_list[i])} for i in top_indices]

    return jsonify({
        'label': label,
        'confidence': confidence,
        'top': top,
        'classes': classes
    })


if __name__ == '__main__':
    print("=" * 60)
    print("Sign Language Translator - Flask Web Application")
    print("=" * 60)
    print("\nStarting server on http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)

