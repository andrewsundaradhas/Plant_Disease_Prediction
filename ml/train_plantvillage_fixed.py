"""
Train a plant disease classification model using the PlantVillage dataset.
"""
import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
    Callback
)
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Generator, Optional
import gc

# Configure TensorFlow for better memory management
try:
    # Enable memory growth for GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU available: {len(gpus)} devices")
        # Enable mixed precision training
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print(f'Mixed precision enabled. Compute dtype: {policy.compute_dtype}, Variable dtype: {policy.variable_dtype}')
except Exception as e:
    print(f"Error configuring GPU: {e}")

# Configuration
CONFIG = {
    'IMG_SIZE': 224,
    'BATCH_SIZE': 16,  # Reduced from 32 to 16 to save memory
    'EPOCHS': 10,
    'LEARNING_RATE': 1e-3,
    'SEED': 42
}

def create_model(num_classes):
    """Create and compile a simple CNN model."""
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'], 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    optimizer = Adam(learning_rate=CONFIG['LEARNING_RATE'])
    
    # Simple model without gradient accumulation for now
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    return model

def create_data_generators(data_dir: str, batch_size: int) -> tuple:
    """Create data generators with memory optimizations."""
    # Data augmentation for training with reduced augmentation to save memory
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,  # Reduced from 20
        width_shift_range=0.15,  # Reduced from 0.2
        height_shift_range=0.15,  # Reduced from 0.2
        shear_range=0.15,  # Reduced from 0.2
        zoom_range=0.15,  # Reduced from 0.2
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],  # Added brightness adjustment
        validation_split=0.1  # Use 10% of training data for validation
    )
    
    # Data generator for validation and testing (only rescaling)
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=None,
        dtype=tf.float32  # Explicitly set dtype to float32
    )
    
    # Create generators
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    print(f"Loading data from:")
    print(f"- Training data: {train_dir}")
    print(f"- Validation data: {val_dir}")
    print(f"- Test data: {test_dir}")
    
    # Training generator with reduced memory footprint
    print("\nCreating training generator...")
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=CONFIG['SEED'],
        interpolation='bilinear',  # Slightly faster than default 'bicubic'
        color_mode='rgb',
        follow_links=False
    )
    
    # Validation generator
    print("\nCreating validation generator...")
    val_generator = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'val') if os.path.exists(os.path.join(data_dir, 'val')) else os.path.join(data_dir, 'train'),
        target_size=(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        interpolation='bilinear',
        color_mode='rgb',
        subset='validation' if not os.path.exists(os.path.join(data_dir, 'val')) else None
    )
    
    # Test generator
    print("\nCreating test generator...")
    test_generator = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        interpolation='bilinear',
        color_mode='rgb'
    )
    
    # Save class indices
    class_indices = train_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    
    print(f"\nFound {len(class_indices)} classes:")
    for i, (class_name, idx) in enumerate(class_indices.items()):
        print(f"  {i+1}. {class_name} (idx: {idx})")
    
    # Save class mapping
    os.makedirs('models', exist_ok=True)
    with open('models/class_mapping.json', 'w') as f:
        json.dump(class_indices, f, indent=2)
    
    return train_generator, val_generator, test_generator, class_indices

class MemoryCleanupCallback(Callback):
    """Callback to clean up memory after each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()
        if tf.config.list_physical_devices('GPU'):
            tf.config.experimental.reset_memory_allocated()

def train_model(data_dir, output_dir, batch_size, epochs, learning_rate):
    """Train the model with memory optimizations."""
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Log file for training progress
    log_file = os.path.join(output_dir, 'training_progress.log')
    with open(log_file, 'w') as f:
        f.write("=== Training Progress Log ===\n")
    
    # Log memory info
    def log_memory_info():
        if tf.config.list_physical_devices('GPU'):
            gpu_info = tf.config.experimental.get_memory_info('GPU:0')
            return f"GPU Memory - Current: {gpu_info['current']/1024**2:.1f}MB, Peak: {gpu_info['peak']/1024**2:.1f}MB"
        return "Running on CPU"
    
    print("\n" + "="*50)
    print(f"Starting training with batch size: {batch_size}")
    print(f"Memory info: {log_memory_info()}")
    print("="*50 + "\n")
    
    # Create data generators
    train_generator, val_generator, test_generator, class_indices = create_data_generators(
        data_dir, batch_size
    )
    
    # Save class indices
    with open(os.path.join(output_dir, 'class_indices.json'), 'w') as f:
        json.dump(class_indices, f, indent=2)
    
    # Calculate steps per epoch
    steps_per_epoch = max(1, train_generator.samples // batch_size)
    validation_steps = max(1, val_generator.samples // batch_size)
    
    print(f"\nTraining samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Test samples: {test_generator.samples}")
    print(f"Steps per epoch: {steps_per_epoch}")
    
    # Create the model
    with tf.device('/device:GPU:0' if tf.config.list_physical_devices('GPU') else '/device:CPU:0'):
        model = create_model(len(class_indices))
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(output_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
            save_weights_only=False
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
            min_delta=0.001
        ),
        TensorBoard(
            log_dir=os.path.join(output_dir, 'logs'),
            histogram_freq=1,
            update_freq='epoch',
            profile_batch=0  # Disable profiling to save memory
        ),
        MemoryCleanupCallback()
    ]
    
    # Train the model with smaller chunks to save memory
    print("\nStarting model training...")
    try:
        # Use minimal configuration for maximum compatibility
        try:
            history = model.fit(
                train_generator,
                steps_per_epoch=len(train_generator),
                epochs=epochs,
                validation_data=val_generator,
                validation_steps=len(val_generator),
                callbacks=callbacks
            )
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            print("Trying with even simpler configuration...")
            # Try with even simpler configuration
            history = model.fit(
                train_generator,
                epochs=epochs,
                validation_data=val_generator,
                callbacks=callbacks
            )
        
        # Save the final model
        final_model_path = os.path.join(output_dir, 'final_model.h5')
        model.save(final_model_path)
        print(f"\nModel saved to {final_model_path}")
        
        # Evaluate on test set
        print("\nEvaluating on test data...")
        test_steps = max(1, test_generator.samples // batch_size)
        test_results = model.evaluate(
            test_generator,
            steps=test_steps,
            verbose=1,
            workers=4,
            use_multiprocessing=True,
            max_queue_size=10
        )
        
        test_metrics = dict(zip(model.metrics_names, test_results))
        test_accuracy = test_metrics.get('accuracy', 0)
        test_loss = test_metrics.get('loss', 0)
        
        print(f"\nTest Results:")
        print(f"- Loss: {test_loss:.4f}")
        print(f"- Accuracy: {test_accuracy:.4f}")
        
        # Save test results
        with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        # Plot training history
        plot_training_history(history, output_dir)
        
        return model, history, test_generator
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        with open(log_file, 'a') as f:
            f.write(f"\nTraining error: {str(e)}\n")
        raise

def plot_training_history(history, output_dir: str) -> None:
    """Plot training and validation metrics with enhanced visualizations."""
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create a figure with 1 row and 2 columns
    plt.figure(figsize=(18, 6))
    
    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy', fontsize=12, pad=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.title('Model Loss', fontsize=12, pad=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot learning rate if it's in history
    if 'lr' in history.history:
        plt.subplot(1, 3, 3)
        plt.plot(history.history['lr'], 'g-', label='Learning Rate', linewidth=2)
        plt.title('Learning Rate Schedule', fontsize=12, pad=10)
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Learning Rate', fontsize=10)
        plt.legend(loc='upper right', fontsize=9)
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout(pad=2.0)
    
    # Save high-quality plot
    plot_path = os.path.join(plots_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save a smaller version for quick viewing
    plot_path_small = os.path.join(plots_dir, 'training_history_small.png')
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], 'b-', label='Train', linewidth=1.5)
    plt.plot(history.history['val_accuracy'], 'r-', label='Val', linewidth=1.5)
    plt.title('Model Accuracy', fontsize=10)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], 'b-', label='Train', linewidth=1.5)
    plt.plot(history.history['val_loss'], 'r-', label='Val', linewidth=1.5)
    plt.title('Model Loss', fontsize=10)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(plot_path_small, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"\nTraining plots saved to {plot_path}")
    print(f"Quick-view plot saved to {plot_path_small}")
    
    # Save training history to JSON for later analysis
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=2)
    print(f"Training history saved to {history_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a plant disease classification model with memory optimizations')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Path to the dataset directory (should contain train/val/test subdirectories)')
    parser.add_argument('--output-dir', type=str, default='output',
                      help='Output directory for models and logs (default: output)')
    parser.add_argument('--batch-size', type=int, default=CONFIG['BATCH_SIZE'],
                      help=f'Batch size for training (default: {CONFIG["BATCH_SIZE"]})')
    parser.add_argument('--epochs', type=int, default=CONFIG['EPOCHS'],
                      help=f'Number of training epochs (default: {CONFIG["EPOCHS"]})')
    parser.add_argument('--learning-rate', type=float, default=CONFIG['LEARNING_RATE'],
                      help=f'Initial learning rate (default: {CONFIG["LEARNING_RATE"]})')
    parser.add_argument('--img-size', type=int, default=CONFIG['IMG_SIZE'],
                      help=f'Image size (width=height) for model input (default: {CONFIG["IMG_SIZE"]})')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    CONFIG.update({
        'BATCH_SIZE': args.batch_size,
        'EPOCHS': args.epochs,
        'LEARNING_RATE': args.learning_rate,
        'IMG_SIZE': args.img_size
    })
    
    # Print training configuration
    print("\n" + "="*60)
    print("PLANT DISEASE CLASSIFICATION - TRAINING CONFIGURATION")
    print("="*60)
    print(f"{'Data directory:':<30} {os.path.abspath(args.data_dir)}")
    print(f"{'Output directory:':<30} {os.path.abspath(args.output_dir)}")
    print(f"{'Batch size:':<30} {CONFIG['BATCH_SIZE']}")
    print(f"{'Epochs:':<30} {CONFIG['EPOCHS']}")
    print(f"{'Learning rate:':<30} {CONFIG['LEARNING_RATE']}")
    print(f"{'Image size:':<30} {CONFIG['IMG_SIZE']}x{CONFIG['IMG_SIZE']}")
    print(f"{'Device:':<30} {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
    print("="*60 + "\n")
    
    # Verify data directory structure
    required_dirs = ['train']  # Only require train, others can be auto-split
    for dir_name in required_dirs:
        dir_path = os.path.join(args.data_dir, dir_name)
        if not os.path.isdir(dir_path):
            print(f"Error: Required directory not found: {dir_path}")
            return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    start_time = tf.timestamp()
    
    try:
        # Train the model
        train_model(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=CONFIG['BATCH_SIZE'],
            epochs=CONFIG['EPOCHS'],
            learning_rate=CONFIG['LEARNING_RATE']
        )
        
        training_time = tf.timestamp() - start_time
        print(f"\nTraining completed in {training_time:.1f} seconds")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Save error log
        with open(os.path.join(args.output_dir, 'error.log'), 'w') as f:
            f.write(f"Training error: {str(e)}\n\n")
            traceback.print_exc(file=f)
    
    # Clean up
    tf.keras.backend.clear_session()
    gc.collect()
    
    print("\nTraining process finished.")

if __name__ == "__main__":
    main()
