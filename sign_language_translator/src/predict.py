"""
Prediction Module
Contains functions for real-time sign language prediction using trained model.
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import os

class SignLanguagePredictor:
    """Class for sign language prediction using MediaPipe and TensorFlow."""
    
    def __init__(self, model_path='../model/sign_model.h5', label_encoder_path='../model/label_encoder.pkl'):
        """Initialize the predictor with model and label encoder."""
        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
			min_detection_confidence=0.5,
			min_tracking_confidence=0.4
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Load model and label encoder
        self.model = None
        self.label_encoder = None
        self.input_type = None  # 'image' or 'landmark'
        self.load_model()
    
    def load_model(self):
        """Load the trained model and label encoder."""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"Model loaded from {self.model_path}")
            else:
                print(f"Warning: Model not found at {self.model_path}")
                return False
            
            if os.path.exists(self.label_encoder_path):
                with open(self.label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                print(f"Label encoder loaded. Classes: {list(self.label_encoder.classes_)}")
            else:
                print(f"Warning: Label encoder not found at {self.label_encoder_path}")
                return False
            
            # Determine input type from model input shape
            input_shape = self.model.input_shape
            if len(input_shape) == 4:  # (batch, height, width, channels)
                self.input_type = 'image'
                print("Model type: Image-based CNN")
            else:  # (batch, features)
                self.input_type = 'landmark'
                print("Model type: Landmark-based MLP")
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def extract_landmarks(self, frame):
        """Extract hand landmarks from frame using MediaPipe."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            # Extract 21 landmarks (x, y, z) = 63 features
            landmark_array = []
            for landmark in landmarks.landmark:
                landmark_array.extend([landmark.x, landmark.y, landmark.z])
            return np.array(landmark_array), landmarks
        return None, None
    
    def preprocess_image(self, frame):
        """Preprocess image for CNN model."""
        # Resize to model input size
        img = cv2.resize(frame, (224, 224))
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict(self, frame):
        """Predict sign language gesture from frame."""
        if self.model is None or self.label_encoder is None:
            return None, 0.0, None
        
        try:
            if self.input_type == 'image':
                # Preprocess image
                processed_img = self.preprocess_image(frame)
                # Predict
                predictions = self.model.predict(processed_img, verbose=0)
            else:
                # Extract landmarks
                landmarks, _ = self.extract_landmarks(frame)
                if landmarks is None:
                    return None, 0.0, None
                # Add batch dimension
                landmarks = np.expand_dims(landmarks, axis=0)
                # Predict
                predictions = self.model.predict(landmarks, verbose=0)
            
            # Get predicted class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_label = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            return predicted_label, confidence, predictions[0]
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0, None
    
    def draw_landmarks(self, frame, landmarks):
        """Draw hand landmarks on frame."""
        if landmarks:
            self.mp_drawing.draw_landmarks(
                frame, landmarks, self.mp_hands.HAND_CONNECTIONS
            )
        return frame
    
    def close(self):
        """Close MediaPipe hands processor."""
        self.hands.close()

