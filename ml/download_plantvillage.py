"""
Download and prepare the PlantVillage dataset using Kaggle Hub.
"""
import os
import kagglehub
import os
import sys
import shutil
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

# Set console output encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Output directories
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "plantvillage_raw" / "PlantVillage"  # Updated path to point to the PlantVillage directory
PROCESSED_DIR = DATA_DIR / "plantvillage"

# Create directories
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def download_dataset():
    """Download the PlantVillage dataset using Kaggle Hub."""
    print("Downloading PlantVillage dataset...")
    try:
        # Download the dataset
        dataset_path = kagglehub.dataset_download("emmarex/plantdisease")
        print(f"Dataset downloaded to: {dataset_path}")
        
        # The dataset is downloaded as a zip file, so we need to extract it
        zip_path = Path(dataset_path) / "plantdisease.zip"
        if zip_path.exists():
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(RAW_DATA_DIR)
            print(f"Dataset extracted to: {RAW_DATA_DIR}")
        
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def prepare_dataset():
    """Prepare the dataset by splitting into train/validation/test sets."""
    print("\nPreparing dataset...")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (PROCESSED_DIR / split).mkdir(exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(RAW_DATA_DIR) 
                 if os.path.isdir(RAW_DATA_DIR / d) and not d.startswith('.')]
    
    print(f"Found {len(class_dirs)} classes in the dataset")
    
    # Process each class
    for class_dir in class_dirs:
        class_path = RAW_DATA_DIR / class_dir
        
        # Create class directories in train/val/test
        for split in ['train', 'val', 'test']:
            (PROCESSED_DIR / split / class_dir).mkdir(exist_ok=True)
        
        # Get all images for this class
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            print(f"No images found in {class_dir}, skipping...")
            continue
            
        print(f"Processing {len(images)} images for class: {class_dir}")
        
        # Split into train (70%), val (15%), test (15%)
        train_val, test = train_test_split(images, test_size=0.15, random_state=42)
        train, val = train_test_split(train_val, test_size=0.176, random_state=42)  # 0.15/0.85 ≈ 0.176
        
        # Copy images to respective directories
        for img_list, split_name in [(train, 'train'), (val, 'val'), (test, 'test')]:
            dest_dir = PROCESSED_DIR / split_name / class_dir
            
            for img in img_list:
                src = class_path / img
                dst = dest_dir / img
                
                if not dst.exists():
                    shutil.copy2(src, dst)
            
            print(f"  - {split_name}: {len(img_list)} images")
    
    print(f"\n✅ Dataset prepared successfully at {PROCESSED_DIR}")

def prepare_existing_dataset():
    """Prepare the dataset from the existing location."""
    print("Preparing dataset from existing location...")
    
    # Create output directories if they don't exist
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(PROCESSED_DIR, split), exist_ok=True)
    
    # Get all class directories (first level)
    class_dirs = [d for d in os.listdir(RAW_DATA_DIR) 
                 if os.path.isdir(os.path.join(RAW_DATA_DIR, d)) 
                 and not d.startswith('.')
                 and d != 'segmented'  # Skip any segmented images
                 and d != 'PlantVillage']  # Skip the nested PlantVillage directory
    
    print(f"Found {len(class_dirs)} classes in the dataset")
    
    # Process each class
    for class_dir in class_dirs:
        class_path = os.path.join(RAW_DATA_DIR, class_dir)
        
        # Create class directories in train/val/test
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(PROCESSED_DIR, split, class_dir), exist_ok=True)
        
        # Get all images for this class, handling nested structure
        images = []
        
        # Check the main directory
        try:
            images.extend([f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        except Exception as e:
            print(f"  Warning: Could not read directory {class_path}: {e}")
        
        # Check for nested directories
        for root, _, files in os.walk(class_path):
            if root != str(class_path):  # Skip the root directory we already checked
                rel_path = os.path.relpath(root, RAW_DATA_DIR)
                images.extend([os.path.join(rel_path, f) for f in files 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not images:
            print(f"No images found in {class_dir}, skipping...")
            continue
            
        print(f"Found {len(images)} images for class: {class_dir}")
        
        if not images:
            print(f"No images found in {class_dir}, skipping...")
            continue
            
        print(f"Processing {len(images)} images for class: {class_dir}")
        
        # Split into train (70%), val (15%), test (15%)
        train_val, test = train_test_split(images, test_size=0.15, random_state=42)
        train, val = train_test_split(train_val, test_size=0.176, random_state=42)  # 0.15/0.85 ≈ 0.176
        
        # Copy images to respective directories
        for img_list, split_name in [(train, 'train'), (val, 'val'), (test, 'test')]:
            dest_dir = os.path.join(PROCESSED_DIR, split_name, class_dir)
            
            for img in img_list:
                src = os.path.join(class_path, img)
                # Check if we need to look in the nested directory
                if not os.path.exists(src) and 'PlantVillage' in img:
                    src = os.path.join(class_path, 'PlantVillage', os.path.basename(img))
                
                if os.path.exists(src):
                    dst = os.path.join(dest_dir, os.path.basename(img))
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                else:
                    print(f"  Warning: Source file not found: {src}")
            
            print(f"  - {split_name}: {len(img_list)} images")
    
    print(f"\n✅ Dataset prepared successfully at {PROCESSED_DIR}")

def main():
    # Check if we already have the dataset
    if os.path.exists(RAW_DATA_DIR) and any(os.scandir(RAW_DATA_DIR)):
        print(f"Found existing dataset at {RAW_DATA_DIR}")
        prepare_existing_dataset()
    else:
        # Try to download the dataset
        print(f"No dataset found at {RAW_DATA_DIR}")
        if not download_dataset():
            print("Failed to download the dataset. Please check your internet connection and try again.")
            print(f"Or manually download the dataset to: {RAW_DATA_DIR}")
            return
        prepare_dataset()
    
    print("\n🎉 Dataset preparation complete!")
    print(f"You can now train the model using:")
    print(f"python ml/train_plantvillage.py --data-dir {PROCESSED_DIR} --output-dir output")

if __name__ == "__main__":
    main()
