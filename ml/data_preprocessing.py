import os
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
import json

def create_directory_structure(base_dir):
    """Create the directory structure for processed data"""
    dirs = ['train', 'val', 'test']
    for dir_name in dirs:
        os.makedirs(os.path.join(base_dir, dir_name), exist_ok=True)

def get_class_dirs(data_dir):
    """Get a list of class directories"""
    return [d for d in os.listdir(data_dir) 
            if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]

def process_image(image_path, target_size=(224, 224)):
    """Process a single image"""
    try:
        img = Image.open(image_path)
        # Convert to RGB if image is RGBA or grayscale
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        # Resize image
        img = img.resize(target_size, Image.LANCZOS)
        return img
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_dataset(source_dir, target_dir, class_names, split_ratios=(0.7, 0.15, 0.15), 
                   img_extensions=('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
    """
    Process the dataset and split into train/val/test sets
    
    Args:
        source_dir: Directory containing class folders with images
        target_dir: Base directory to save processed data
        class_names: List of class names (subdirectory names)
        split_ratios: Tuple of (train, val, test) ratios
        img_extensions: Tuple of valid image file extensions
    """
    # Create target directories
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    test_dir = os.path.join(target_dir, 'test')
    
    # Create class directories in each split
    for split_dir in [train_dir, val_dir, test_dir]:
        for class_name in class_names:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
    
    # Process each class
    class_stats = {}
    
    for class_name in class_names:
        print(f"\nProcessing class: {class_name}")
        class_dir = os.path.join(source_dir, class_name)
        
        # Get all image files
        image_files = []
        for ext in img_extensions:
            image_files.extend(Path(class_dir).glob(f'*{ext}'))
        
        if not image_files:
            print(f"No images found for class {class_name}")
            continue
            
        print(f"Found {len(image_files)} images")
        class_stats[class_name] = len(image_files)
        
        # Split into train/val/test
        train_files, test_val_files = train_test_split(
            image_files, 
            test_size=(1 - split_ratios[0]), 
            random_state=42
        )
        val_files, test_files = train_test_split(
            test_val_files,
            test_size=split_ratios[2]/(split_ratios[1] + split_ratios[2]),
            random_state=42
        )
        
        # Process and save images
        for files, split_name in zip(
            [train_files, val_files, test_files],
            ['train', 'val', 'test']
        ):
            print(f"  Saving {len(files)} images to {split_name}")
            for img_path in files:
                img = process_image(img_path)
                if img is not None:
                    target_path = os.path.join(
                        target_dir, 
                        split_name, 
                        class_name, 
                        os.path.basename(img_path)
                    )
                    img.save(target_path)
    
    # Save dataset statistics
    stats = {
        'total_images': sum(class_stats.values()),
        'num_classes': len(class_stats),
        'class_distribution': class_stats,
        'split_ratios': {
            'train': split_ratios[0],
            'val': split_ratios[1],
            'test': split_ratios[2]
        },
        'splits': {
            'train': int(sum(class_stats.values()) * split_ratios[0]),
            'val': int(sum(class_stats.values()) * split_ratios[1]),
            'test': int(sum(class_stats.values()) * split_ratios[2])
        }
    }
    
    with open(os.path.join(target_dir, 'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\nDataset processing complete!")
    print(f"Total images: {stats['total_images']}")
    print(f"Number of classes: {stats['num_classes']}")
    print(f"Train/Val/Test split: {len(train_files)}/{len(val_files)}/{len(test_files)}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess dataset for crop health prediction')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Path to the directory containing class folders with images')
    parser.add_argument('--target_dir', type=str, default='data/processed',
                        help='Path to save the processed dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio of training data (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Ratio of validation data (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Ratio of test data (default: 0.15)')
    
    args = parser.parse_args()
    
    # Validate ratios
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    
    # Create target directory
    os.makedirs(args.target_dir, exist_ok=True)
    
    # Get class names
    class_names = get_class_dirs(args.source_dir)
    if not class_names:
        raise ValueError(f"No class folders found in {args.source_dir}")
    
    print(f"Found {len(class_names)} classes: {', '.join(class_names)}")
    
    # Process dataset
    process_dataset(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        class_names=class_names,
        split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio)
    )

if __name__ == "__main__":
    main()
