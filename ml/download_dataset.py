"""
Script to download and prepare the PlantVillage dataset for training.
This script assumes you have the dataset downloaded from Kaggle and extracted.
"""
import os
import shutil
import zipfile
import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

# Class mapping for PlantVillage dataset
PLANT_DISEASE_MAPPING = {
    'Apple___Apple_scab': 'Apple_Scab',
    'Apple___Black_rot': 'Apple_Black_Rot',
    'Apple___Cedar_apple_rust': 'Apple_Cedar_Rust',
    'Apple___healthy': 'Apple_Healthy',
    'Blueberry___healthy': 'Blueberry_Healthy',
    'Cherry_(including_sour)___Powdery_mildew': 'Cherry_Powdery_Mildew',
    'Cherry_(including_sour)___healthy': 'Cherry_Healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_': 'Corn_Common_Rust',
    'Corn_(maize)___Northern_Leaf_Blight': 'Corn_Northern_Leaf_Blight',
    'Corn_(maize)___healthy': 'Corn_Healthy',
    'Grape___Black_rot': 'Grape_Black_Rot',
    'Grape___Esca_(Black_Measles)': 'Grape_Esca',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Grape_Leaf_Blight',
    'Grape___healthy': 'Grape_Healthy',
    'Orange___Haunglongbing_(Citrus_greening)': 'Citrus_Greening',
    'Peach___Bacterial_spot': 'Peach_Bacterial_Spot',
    'Peach___healthy': 'Peach_Healthy',
    'Pepper,_bell___Bacterial_spot': 'Pepper_Bacterial_Spot',
    'Pepper,_bell___healthy': 'Pepper_Healthy',
    'Potato___Early_blight': 'Potato_Early_Blight',
    'Potato___Late_blight': 'Potato_Late_Blight',
    'Potato___healthy': 'Potato_Healthy',
    'Raspberry___healthy': 'Raspberry_Healthy',
    'Soybean___healthy': 'Soybean_Healthy',
    'Squash___Powdery_mildew': 'Squash_Powdery_Mildew',
    'Strawberry___Leaf_scorch': 'Strawberry_Leaf_Scorch',
    'Strawberry___healthy': 'Strawberry_Healthy',
    'Tomato___Bacterial_spot': 'Tomato_Bacterial_Spot',
    'Tomato___Early_blight': 'Tomato_Early_Blight',
    'Tomato___Late_blight': 'Tomato_Late_Blight',
    'Tomato___Leaf_Mold': 'Tomato_Leaf_Mold',
    'Tomato___Septoria_leaf_spot': 'Tomato_Septoria_Leaf_Spot',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Tomato_Spider_Mites',
    'Tomato___Target_Spot': 'Tomato_Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Tomato_Yellow_Leaf_Curl',
    'Tomato___Tomato_mosaic_virus': 'Tomato_Mosaic_Virus',
    'Tomato___healthy': 'Tomato_Healthy'
}

def create_dataset_structure(base_dir, output_dir, test_size=0.2, val_size=0.1, random_state=42):
    """
    Create train/validation/test split from the PlantVillage dataset
    """
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(base_dir) 
                 if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith('.')]
    
    print(f"Found {len(class_dirs)} classes in the dataset")
    
    # Process each class
    for class_dir in class_dirs:
        class_name = PLANT_DISEASE_MAPPING.get(class_dir, class_dir)
        
        # Create class directories in train/val/test
        for split_dir in [train_dir, val_dir, test_dir]:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
        
        # Get all images for this class
        class_path = os.path.join(base_dir, class_dir)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            print(f"No images found in {class_dir}, skipping...")
            continue
            
        print(f"Processing {len(images)} images for class: {class_name}")
        
        # Split into train, val, test
        train_val, test = train_test_split(images, test_size=test_size, random_state=random_state)
        train, val = train_test_split(train_val, test_size=val_size/(1-test_size), 
                                     random_state=random_state)
        
        # Copy images to respective directories
        for img_list, split_name in [(train, 'train'), (val, 'val'), (test, 'test')]:
            dest_dir = os.path.join(output_dir, split_name, class_name)
            
            for img in img_list:
                src = os.path.join(class_path, img)
                dst = os.path.join(dest_dir, img)
                
                # Skip if already exists
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
            
            print(f"  - {split_name}: {len(img_list)} images")
    
    print(f"\nDataset prepared successfully at {output_dir}")
    print(f"Train: {os.path.join(output_dir, 'train')}")
    print(f"Validation: {os.path.join(output_dir, 'val')}")
    print(f"Test: {os.path.join(output_dir, 'test')}")

def main():
    parser = argparse.ArgumentParser(description='Prepare PlantVillage dataset for training')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Path to the extracted PlantVillage dataset directory')
    parser.add_argument('--output-dir', type=str, default='data/plantvillage',
                       help='Output directory for processed dataset (default: data/plantvillage)')
    parser.add_argument('--test-size', type=float, default=0.15,
                       help='Proportion of data for testing (default: 0.15)')
    parser.add_argument('--val-size', type=float, default=0.15,
                       help='Proportion of training data for validation (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Preparing PlantVillage dataset from {args.input_dir}")
    print(f"Output will be saved to {args.output_dir}")
    print(f"Test size: {args.test_size}, Validation size: {args.val_size}, Random seed: {args.seed}")
    
    # Create dataset structure
    create_dataset_structure(
        base_dir=args.input_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed
    )

if __name__ == "__main__":
    main()
