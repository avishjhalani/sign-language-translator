"""
Data Collection Script for Sign Language Gestures
Captures hand gesture images from webcam and saves them in labeled folders.
Uses MediaPipe Hands for hand detection and landmark extraction.
"""

import cv2
import os
import numpy as np
import mediapipe as mp
from datetime import datetime

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Create dataset directory structure
DATASET_DIR = '../dataset'
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

def create_label_folder(label):
    """Create a folder for a specific label if it doesn't exist."""
    label_path = os.path.join(DATASET_DIR, label)
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    return label_path

def save_landmarks(landmarks, label, index):
    """Save hand landmarks as numpy array."""
    landmarks_dir = os.path.join(DATASET_DIR, f'{label}_landmarks')
    if not os.path.exists(landmarks_dir):
        os.makedirs(landmarks_dir)
    
    # Extract 21 landmarks (x, y, z) = 63 features
    landmark_array = []
    for landmark in landmarks.landmark:
        landmark_array.extend([landmark.x, landmark.y, landmark.z])
    
    np.save(os.path.join(landmarks_dir, f'{label}_{index:04d}.npy'), np.array(landmark_array))
    return os.path.join(landmarks_dir, f'{label}_{index:04d}.npy')

def collect_data():
    """Main function to collect sign language gesture data."""
    cap = cv2.VideoCapture(0)
    
    print("=" * 60)
    print("Sign Language Data Collection Tool")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Enter the label/class name (e.g., 'A', 'B', 'Hello', 'Thanks')")
    print("2. Press 'SPACE' to capture an image")
    print("3. Press 'q' to quit and move to next label")
    print("4. Press 'ESC' to exit completely")
    print("=" * 60)
    
    current_label = None
    image_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
        
        # Display instructions
        if current_label:
            cv2.putText(frame, f"Label: {current_label}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {image_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Press 'n' to start new label", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.putText(frame, "SPACE: Capture | q: Next Label | ESC: Exit", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Data Collection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Start new label collection
        if key == ord('n'):
            label_input = input("\nEnter label name (or press Enter to skip): ").strip()
            if label_input:
                current_label = label_input
                label_path = create_label_folder(current_label)
                image_count = len([f for f in os.listdir(label_path) if f.endswith('.jpg')])
                print(f"Started collecting data for label: {current_label}")
                print(f"Existing images: {image_count}")
        
        # Capture image
        if key == ord(' ') and current_label:
            label_path = create_label_folder(current_label)
            image_filename = os.path.join(label_path, f'{current_label}_{image_count:04d}.jpg')
            cv2.imwrite(image_filename, frame)
            
            # Save landmarks if hand detected
            if results.multi_hand_landmarks:
                save_landmarks(results.multi_hand_landmarks[0], current_label, image_count)
            
            image_count += 1
            print(f"Saved: {image_filename} (Total: {image_count})")
        
        # Quit current label
        if key == ord('q'):
            if current_label:
                print(f"\nFinished collecting data for '{current_label}'. Total images: {image_count}")
                current_label = None
                image_count = 0
        
        # Exit completely
        if key == 27:  # ESC key
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\nData collection completed!")

if __name__ == "__main__":
    collect_data()

