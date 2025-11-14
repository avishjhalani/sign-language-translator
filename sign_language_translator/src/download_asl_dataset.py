"""
Download Public ASL Alphabet Dataset
Downloads a publicly available ASL alphabet dataset and integrates it into the project.
"""

import os
import zipfile
import urllib.request
import shutil
from pathlib import Path

DATASET_DIR = '../dataset'
DATASET_URL = "https://github.com/loicmarie/sign-language-alphabet-recognizer/raw/master/dataset/asl_alphabet_train.zip"
BACKUP_URL = "https://www.kaggle.com/datasets/grassknoted/asl-alphabet/download?datasetVersionNumber=1"

def download_file(url, dest_path, chunk_size=8192):
    """Download a file from URL with progress."""
    print(f"Downloading from {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=lambda blocknum, blocksize, totalsize: 
            print(f"\rProgress: {min(100, (blocknum * blocksize * 100) // totalsize)}%", end='') if totalsize > 0 else None)
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"\nDownload failed: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract zip file to directory."""
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete!")
        return True
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False

def organize_dataset(source_dir, target_dir):
    """Organize downloaded dataset into our folder structure."""
    print("Organizing dataset...")
    
    # Common dataset structures we might encounter
    possible_paths = [
        os.path.join(source_dir, 'asl_alphabet_train'),
        os.path.join(source_dir, 'asl_alphabet_train', 'asl_alphabet_train'),
        os.path.join(source_dir, 'train'),
        source_dir
    ]
    
    train_dir = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            # Check if it contains letter folders
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            if any(d.isalpha() and len(d) == 1 for d in subdirs):
                train_dir = path
                break
    
    if not train_dir:
        print("Could not find organized dataset structure. Checking all subdirectories...")
        # Try to find any directory with letter folders
        for root, dirs, files in os.walk(source_dir):
            subdirs = [d for d in dirs if d.isalpha() and len(d) == 1 and d.isupper()]
            if len(subdirs) >= 10:  # At least 10 letters found
                train_dir = root
                break
    
    if not train_dir:
        print("ERROR: Could not find dataset with letter-labeled folders!")
        return False
    
    print(f"Found dataset at: {train_dir}")
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy/move letter folders
    letters = [d for d in os.listdir(train_dir) if d.isalpha() and len(d) == 1 and d.isupper()]
    letters.sort()
    
    print(f"Found {len(letters)} letter folders: {', '.join(letters)}")
    
    total_images = 0
    for letter in letters:
        src_folder = os.path.join(train_dir, letter)
        dst_folder = os.path.join(target_dir, letter)
        
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder, exist_ok=True)
        
        # Copy images
        image_files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            src_path = os.path.join(src_folder, img_file)
            # Rename to avoid conflicts
            base_name = os.path.splitext(img_file)[0]
            ext = os.path.splitext(img_file)[1]
            dst_path = os.path.join(dst_folder, f"{base_name}_ds{ext}")
            
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
                total_images += 1
        
        print(f"  {letter}: {len(image_files)} images")
    
    print(f"\nTotal images copied: {total_images}")
    return True

def download_kaggle_dataset():
    """Alternative: Try to download from Kaggle (requires kaggle API)."""
    try:
        import kaggle
        print("Kaggle API found. Attempting to download dataset...")
        # This would require kaggle.json credentials
        # kaggle.api.dataset_download_files('grassknoted/asl-alphabet', path='../temp', unzip=True)
        print("Note: Kaggle download requires API credentials. Skipping...")
        return False
    except ImportError:
        print("Kaggle API not installed. Install with: pip install kaggle")
        return False

def main():
    """Main function to download and organize ASL dataset."""
    print("=" * 60)
    print("ASL Alphabet Dataset Downloader")
    print("=" * 60)
    
    # Create temp directory for download
    temp_dir = '../temp_dataset'
    os.makedirs(temp_dir, exist_ok=True)
    zip_path = os.path.join(temp_dir, 'asl_alphabet_train.zip')
    
    # Try to download
    print("\nAttempting to download ASL alphabet dataset...")
    print("Note: This dataset is large (~1GB). Download may take several minutes.")
    
    success = download_file(DATASET_URL, zip_path)
    
    if not success:
        print("\nPrimary download failed. Trying alternative method...")
        print("Please manually download the ASL alphabet dataset from:")
        print("https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        print("Or visit: https://github.com/loicmarie/sign-language-alphabet-recognizer")
        print("\nAfter downloading, extract it and run this script with --manual flag")
        return
    
    # Extract
    extract_to = os.path.join(temp_dir, 'extracted')
    os.makedirs(extract_to, exist_ok=True)
    
    if not extract_zip(zip_path, extract_to):
        print("Failed to extract dataset!")
        return
    
    # Organize into our dataset structure
    if organize_dataset(extract_to, DATASET_DIR):
        print("\n" + "=" * 60)
        print("Dataset successfully integrated!")
        print(f"Dataset location: {os.path.abspath(DATASET_DIR)}")
        print("=" * 60)
        
        # Cleanup temp files
        print("\nCleaning up temporary files...")
        try:
            shutil.rmtree(temp_dir)
            print("Cleanup complete!")
        except Exception as e:
            print(f"Warning: Could not delete temp directory: {e}")
    else:
        print("\nFailed to organize dataset. Please check the structure manually.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--manual':
        print("Manual mode: Please specify the path to extracted dataset:")
        manual_path = input("Path: ").strip()
        if os.path.exists(manual_path):
            organize_dataset(manual_path, DATASET_DIR)
        else:
            print("Path not found!")
    else:
        main()

