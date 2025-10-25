from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os

def generate_class_mapping(data_dir, output_path='class_mapping.json'):
    """
    Generate and save class mapping from a directory structure.
    
    Args:
        data_dir (str): Directory containing class subdirectories
        output_path (str): Path to save the class mapping JSON file
    """
    print(f"🔍 Generating class mapping from {data_dir}...")
    
    # Use ImageDataGenerator to get class indices
    datagen = ImageDataGenerator()
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    
    # Invert the mapping from {class_name: index} to {index: class_name}
    class_mapping = {str(v): k for k, v in generator.class_indices.items()}
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    print(f"✅ Class mapping saved to {output_path}")
    print("\nClass Mappings:")
    for idx, class_name in class_mapping.items():
        print(f"  {idx}: {class_name}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate class mapping from dataset directory')
    parser.add_argument('--data-dir', type=str, default='data/processed/train',
                       help='Directory containing class subdirectories')
    parser.add_argument('--output', type=str, default='class_mapping.json',
                       help='Path to save the class mapping JSON file')
    
    args = parser.parse_args()
    generate_class_mapping(args.data_dir, args.output)
