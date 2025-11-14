"""
Automated preprocessing and training script
Preprocesses the augmented dataset and trains the model automatically.
"""

import os
import sys
import subprocess

def run_preprocess():
    """Run preprocessing with image-based option."""
    print("=" * 60)
    print("Step 1: Preprocessing Data")
    print("=" * 60)
    
    # Modify DATASET_DIR to use augmented dataset if it exists
    preprocess_script = 'preprocess_data.py'
    
    # Check if augmented dataset exists
    augmented_dir = '../dataset_augmented'
    if os.path.exists(augmented_dir):
        print(f"Using augmented dataset: {augmented_dir}")
        # Temporarily modify the script to use augmented dataset
        with open(preprocess_script, 'r') as f:
            content = f.read()
        
        # Replace DATASET_DIR
        modified_content = content.replace(
            "DATASET_DIR = '../dataset'",
            "DATASET_DIR = '../dataset_augmented'"
        )
        
        with open(preprocess_script + '.tmp', 'w') as f:
            f.write(modified_content)
        
        # Run with input "1" for image-based
        process = subprocess.Popen(
            [sys.executable, preprocess_script + '.tmp'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input='1\n')
        
        # Cleanup
        if os.path.exists(preprocess_script + '.tmp'):
            os.remove(preprocess_script + '.tmp')
        
        print(stdout)
        if stderr:
            print("Errors:", stderr)
        
        return process.returncode == 0
    else:
        print("Using original dataset: ../dataset")
        process = subprocess.Popen(
            [sys.executable, preprocess_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input='1\n')
        
        print(stdout)
        if stderr:
            print("Errors:", stderr)
        
        return process.returncode == 0

def run_train():
    """Run model training."""
    print("\n" + "=" * 60)
    print("Step 2: Training Model")
    print("=" * 60)
    
    process = subprocess.Popen(
        [sys.executable, 'train_model.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Stream output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    if process.stderr:
        for line in process.stderr:
            print("Error:", line, end='')
    
    return process.returncode == 0

if __name__ == "__main__":
    print("=" * 60)
    print("Automated Preprocessing and Training")
    print("=" * 60)
    
    # Step 1: Preprocess
    if not run_preprocess():
        print("\nERROR: Preprocessing failed!")
        sys.exit(1)
    
    # Step 2: Train
    if not run_train():
        print("\nERROR: Training failed!")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("SUCCESS: Model training completed!")
    print("=" * 60)
    print("\nYou can now run the app with: python ../app.py")

