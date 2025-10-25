"""
Train a plant disease classification model using the PlantVillage dataset.
This version uses a simpler model architecture to avoid compatibility issues.
"""
import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)
import matplotlib.pyplot as plt

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
SEED = 42

def create_model(num_classes):
    """Create a simple CNN model."""
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# [Rest of the code remains the same as train_plantvillage_rgb.py]

def create_data_generators(data_dir, batch_size):
    """Create data generators for training, validation, and testing."""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Data generator for validation and testing (only rescaling)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    print(f"Loading data from:")
    print(f"- Training data: {train_dir}")
    print(f"- Validation data: {val_dir}")
    print(f"- Test data: {test_dir}")
    
    # Training generator
    print("\nCreating training generator...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=SEED
    )
    
    # Validation generator
    print("\nCreating validation generator...")
    val_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Test generator
    print("\nCreating test generator...")
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
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

def train_model(data_dir, output_dir, batch_size, epochs, learning_rate):
    """Train the model."""
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data generators
    train_generator, val_generator, test_generator, class_indices = create_data_generators(
        data_dir, batch_size
    )
    
    # Create the model
    model = create_model(len(class_indices))
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(output_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(output_dir, 'logs'),
            histogram_freq=1
        )
    ]
    
    # Print model summary
    model.summary()
    
    # Train the model
    print("\nStarting model training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate the model on test data
    print("\nEvaluating on test data...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save the final model
    model.save(os.path.join(output_dir, 'final_model.h5'))
    print(f"\nModel saved to {output_dir}")
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    return model, history, test_generator

def plot_training_history(history, output_dir):
    """Plot training and validation metrics."""
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, 'training_history.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Training plots saved to {plot_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a plant disease classification model')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to the dataset directory (should contain train/val/test subdirectories)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for models and logs (default: output)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help=f'Batch size for training (default: {BATCH_SIZE})')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help=f'Number of training epochs (default: {EPOCHS})')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE,
                       help=f'Initial learning rate (default: {LEARNING_RATE})')
    
    args = parser.parse_args()
    
    print("=== Plant Disease Classification Model Training ===")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 50 + "\n")
    
    # Train the model
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )

if __name__ == "__main__":
    # Set memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            # Enable memory growth for all GPUs
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print("Memory growth enabled for all GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(f"Error setting memory growth: {e}")
    
    main()
