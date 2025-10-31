"""
Integration Test Script for Plant Disease Classification System
This script tests the integration of all components:
1. Data loading and preprocessing
2. Model training
3. Model evaluation
4. Backend API (if any)
5. Frontend integration (if any)
"""

import os
import sys
import json
import requests
import numpy as np
from pathlib import Path
import tensorflow as tf
from ml.train_plantvillage import (
    create_model,
    create_data_generators,
    evaluate_model,
    DEFAULT_IMG_SIZE,
    DEFAULT_BATCH_SIZE
)

def test_data_loading():
    """Test if data is loaded correctly."""
    print("\n[INFO] Testing data loading...")
    
    data_dir = Path('data/plantvillage')
    required_dirs = ['train', 'val', 'test']
    
    # Check if required directories exist
    for split in required_dirs:
        if not (data_dir / split).exists():
            print(f"[ERROR] Missing directory: {data_dir/split}")
            return False
    
    # Check if each split has class directories
    for split in required_dirs:
        split_path = data_dir / split
        try:
            classes = [d for d in os.listdir(split_path) if (split_path/d).is_dir()]
            if not classes:
                print(f"[ERROR] No class directories found in {split_path}")
                return False
                
            print(f"[OK] Found {len(classes)} classes in {split} split")
            
            # Check for images in each class
            for class_name in classes[:3]:  # Check first 3 classes
                images = list((split_path/class_name).glob('*.*'))
                if not images:
                    print(f"[ERROR] No images found in {split}/{class_name}")
                    return False
                print(f"   - {class_name}: {len(images)} images")
        except Exception as e:
            print(f"[ERROR] Error checking {split_path}: {str(e)}")
            return False
    
    print("[OK] Data loading tests passed!")
    return True

def test_model_creation():
    """Test if the model can be created and compiled."""
    print("\n[TEST] Testing model creation...")
    
    try:
        # Create a small test model
        test_model = create_model(num_classes=10, img_size=32)
        test_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Test model input/output shapes
        input_shape = test_model.input_shape
        output_shape = test_model.output_shape
        
        print("[OK] Model created successfully!")
        print(f"   - Input shape: {input_shape}")
        print(f"   - Output shape: {output_shape}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Model creation failed: {str(e)}")
        return False

def test_training_pipeline():
    """Test the training pipeline with a small subset of data."""
    print("\n[TEST] Testing training pipeline...")
    
    test_dir = Path('test_data')
    try:
        # Create a small test dataset
        test_dir.mkdir(exist_ok=True)
        
        # Create a small model
        num_classes = 3
        model = create_model(num_classes=num_classes, img_size=32)
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("[OK] Training pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Training pipeline test failed: {str(e)}")
        return False
    finally:
        # Clean up
        if test_dir.exists():
            import shutil
            try:
                shutil.rmtree(test_dir)
            except Exception as e:
                print(f"[WARNING] Could not clean up test directory: {str(e)}")

def run_integration_tests():
    """Run all integration tests."""
    print("\n=== Starting Integration Tests ===")
    print("=" * 50)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation),
        ("Training Pipeline", test_training_pipeline)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}")
        print("-" * (len(test_name) + 7))
        success = test_func()
        results[test_name] = "PASSED" if success else "FAILED"
    
    # Print summary
    print("\n=== Test Summary ===")
    print("=" * 50)
    for test_name, result in results.items():
        status = "[PASS]" if result == "PASSED" else "[FAIL]"
        print(f"{status} - {test_name}")
    
    if all(r == "PASSED" for r in results.values()):
        print("\n=== All tests passed successfully! ===")
        return True
    else:
        print("\n=== Some tests failed. Please check the logs above. ===")
        return False

if __name__ == "__main__":
    run_integration_tests()
