"""
Data Preprocessing Script
Preprocesses collected images and hand landmarks for model training.
Supports both image-based and landmark-based approaches.
"""

import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mediapipe as mp

# Check if augmented dataset exists, otherwise use original
if os.path.exists('../dataset_augmented'):
    DATASET_DIR = '../dataset_augmented'
    print("Using augmented dataset for preprocessing...")
else:
    DATASET_DIR = '../dataset'
OUTPUT_DIR = '../model'
IMG_SIZE = (224, 224)  # Standard input size for CNN

def load_images_from_folder(folder_path):
    """Load and preprocess images from a folder."""
    images = []
    labels = []
    
    if not os.path.exists(folder_path):
        return images, labels
    
    label = os.path.basename(folder_path)
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                # Resize image
                img = cv2.resize(img, IMG_SIZE)
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Normalize to [0, 1]
                img = img.astype(np.float32) / 255.0
                images.append(img)
                labels.append(label)
    
    return images, labels

def load_landmarks_from_folder(folder_path):
    """Load hand landmark arrays from a folder."""
    landmarks = []
    labels = []
    
    if not os.path.exists(folder_path):
        return landmarks, labels
    
    # Extract label from folder name (e.g., 'A_landmarks' -> 'A')
    label = os.path.basename(folder_path).replace('_landmarks', '')
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            landmark_path = os.path.join(folder_path, filename)
            landmark_array = np.load(landmark_path)
            landmarks.append(landmark_array)
            labels.append(label)
    
    return landmarks, labels

def preprocess_images():
    """Preprocess image-based dataset."""
    print("Loading images from dataset...")
    all_images = []
    all_labels = []
    
    # Load images from all label folders
    for label_folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, label_folder)
        if os.path.isdir(folder_path) and not label_folder.endswith('_landmarks'):
            images, labels = load_images_from_folder(folder_path)
            all_images.extend(images)
            all_labels.extend(labels)
            print(f"Loaded {len(images)} images for label: {label_folder}")
    
    if len(all_images) == 0:
        print("No images found! Please collect data first using collect_data.py")
        return None, None, None, None
    
    print(f"\nTotal images loaded: {len(all_images)}")
    print("Converting to numpy arrays (this may take a moment for large datasets)...")
    
    # Convert to numpy arrays in chunks to avoid memory issues
    try:
        # Try to convert all at once first
        X = np.array(all_images, dtype=np.float32)
        y = np.array(all_labels)
    except MemoryError:
        print("Memory error: Processing in chunks...")
        # Process in chunks
        chunk_size = 5000
        X_chunks = []
        for i in range(0, len(all_images), chunk_size):
            chunk = all_images[i:i+chunk_size]
            X_chunks.append(np.array(chunk, dtype=np.float32))
            print(f"Processed chunk {i//chunk_size + 1}/{(len(all_images)-1)//chunk_size + 1}")
        X = np.concatenate(X_chunks, axis=0)
        y = np.array(all_labels)
        del all_images, X_chunks  # Free memory
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Save label encoder
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"\nPreprocessing complete!")
    print(f"Total images: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {list(label_encoder.classes_)}")
    
    return X_train, X_test, y_train, y_test

def preprocess_landmarks():
    """Preprocess landmark-based dataset."""
    print("Loading landmarks from dataset...")
    all_landmarks = []
    all_labels = []
    
    # Load landmarks from all landmark folders
    for folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder)
        if os.path.isdir(folder_path) and folder.endswith('_landmarks'):
            landmarks, labels = load_landmarks_from_folder(folder_path)
            all_landmarks.extend(landmarks)
            all_labels.extend(labels)
            print(f"Loaded {len(landmarks)} landmarks for label: {labels[0] if labels else 'N/A'}")
    
    if len(all_landmarks) == 0:
        print("No landmarks found! Please collect data first using collect_data.py")
        return None, None, None, None
    
    # Convert to numpy arrays
    X = np.array(all_landmarks)
    y = np.array(all_labels)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Save label encoder
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"\nPreprocessing complete!")
    print(f"Total landmarks: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {list(label_encoder.classes_)}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    print("=" * 60)
    print("Data Preprocessing Script")
    print("=" * 60)
    print("\nChoose preprocessing method:")
    print("1. Image-based (CNN)")
    print("2. Landmark-based (MLP)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        X_train, X_test, y_train, y_test = preprocess_images()
        if X_train is not None:
            # Save preprocessed data
            np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
            np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test)
            np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
            np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)
            print("\nPreprocessed data saved to model/ directory")
    elif choice == "2":
        X_train, X_test, y_train, y_test = preprocess_landmarks()
        if X_train is not None:
            # Save preprocessed data
            np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
            np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test)
            np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
            np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)
            print("\nPreprocessed data saved to model/ directory")
    else:
        print("Invalid choice!")

