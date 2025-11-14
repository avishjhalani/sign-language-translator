"""
Model Training Script
Trains a CNN for image-based classification or MLP for landmark-based classification.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pickle

OUTPUT_DIR = '../model'
MODEL_PATH = os.path.join(OUTPUT_DIR, 'sign_model.h5')

def build_cnn_model(num_classes, input_shape=(224, 224, 3)):
    """Build a CNN model for image classification."""
    model = keras.Sequential([
        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_mlp_model(num_classes, input_shape=63):
    """Build an MLP model for landmark classification."""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Main training function."""
    print("=" * 60)
    print("Sign Language Model Training")
    print("=" * 60)
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    X_train_path = os.path.join(OUTPUT_DIR, 'X_train.npy')
    X_test_path = os.path.join(OUTPUT_DIR, 'X_test.npy')
    y_train_path = os.path.join(OUTPUT_DIR, 'y_train.npy')
    y_test_path = os.path.join(OUTPUT_DIR, 'y_test.npy')
    label_encoder_path = os.path.join(OUTPUT_DIR, 'label_encoder.pkl')
    
    if not all(os.path.exists(p) for p in [X_train_path, X_test_path, y_train_path, y_test_path, label_encoder_path]):
        print("Error: Preprocessed data not found!")
        print("Please run preprocess_data.py first.")
        return
    
    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)
    
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    num_classes = len(label_encoder.classes_)
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {list(label_encoder.classes_)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Determine model type based on input shape
    if len(X_train.shape) == 4:  # Image data (batch, height, width, channels)
        print("\nDetected image data - Building CNN model...")
        model = build_cnn_model(num_classes, X_train.shape[1:])
        use_augmentation = True
    else:  # Landmark data (batch, features)
        print("\nDetected landmark data - Building MLP model...")
        model = build_mlp_model(num_classes, X_train.shape[1])
        use_augmentation = False
    
    model.summary()
    
    # Data augmentation for images
    if use_augmentation:
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        train_generator = datagen.flow(X_train, y_train, batch_size=32)
    else:
        train_generator = None
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Train model
    print("\nStarting training...")
    epochs = 50
    batch_size = 32
    
    if use_augmentation:
        history = model.fit(
            train_generator,
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")
    
    # Load best model if it exists
    if os.path.exists(MODEL_PATH):
        model.load_weights(MODEL_PATH)
        final_loss, final_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nBest Model - Test Accuracy: {final_accuracy:.4f}")
        print(f"Best Model - Test Loss: {final_loss:.4f}")
    
    print(f"\nModel saved to: {MODEL_PATH}")
    print("Training completed!")

if __name__ == "__main__":
    # Set memory growth for GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU detected and configured")
    
    train_model()

