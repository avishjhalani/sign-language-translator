"""
Quick Setup Script
Helps verify installation and create necessary directories.
"""

import os
import sys

def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")
    required_packages = [
        'tensorflow',
        'cv2',
        'mediapipe',
        'numpy',
        'flask',
        'sklearn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'cv2':
                __import__('cv2')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Please run: pip install -r requirements.txt")
        return False
    else:
        print("\nAll dependencies are installed!")
        return True

def create_directories():
    """Create necessary directories if they don't exist."""
    print("\nCreating directories...")
    directories = [
        'model',
        'dataset',
        'templates',
        'static/css',
        'static/js',
        'src'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ {directory}/")

def main():
    print("=" * 60)
    print("Sign Language Translator - Setup Script")
    print("=" * 60)
    print()
    
    # Create directories
    create_directories()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    print("\n" + "=" * 60)
    if deps_ok:
        print("Setup complete! You can now:")
        print("1. Run 'python src/collect_data.py' to collect training data")
        print("2. Run 'python src/preprocess_data.py' to preprocess data")
        print("3. Run 'python src/train_model.py' to train the model")
        print("4. Run 'python app.py' to start the web application")
    else:
        print("Setup incomplete. Please install missing dependencies first.")
    print("=" * 60)

if __name__ == "__main__":
    main()

