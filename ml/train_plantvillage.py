"""
Train a deep learning model on the PlantVillage dataset for plant disease classification.
This script handles the entire training pipeline including data preparation, model training,
and evaluation.
"""

import logging
import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from datetime import datetime

# Set random seed for reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Default parameters
DEFAULT_BATCH_SIZE = 32
DEFAULT_IMG_SIZE = 224
DEFAULT_EPOCHS = 50
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_DROPOUT_RATE = 0.5
DEFAULT_VAL_SPLIT = 0.15
DEFAULT_TEST_SPLIT = 0.15

def create_model(num_classes, img_size=DEFAULT_IMG_SIZE, dropout_rate=DEFAULT_DROPOUT_RATE):
    """
    Create a CNN model using EfficientNetB0 as base with custom top layers.
    
    Args:
        num_classes: Number of output classes
        img_size: Size of input images (width=height)
    """
    # Input shape for RGB images
    input_shape = (img_size, img_size, 3)
    
    # Create base model with pre-trained weights
    base_model = EfficientNetB0(
        weights=None,  # Don't load imagenet weights to avoid shape mismatch
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate * 0.6)(x)  # Slightly less dropout
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def create_data_generators(data_dir, batch_size, img_size):
    """
    Create data generators for training, validation, and testing.
    
    Args:
        data_dir: Base directory containing 'train', 'val', 'test' subdirectories
        batch_size: Batch size for data generators
        img_size: Size to resize images to
        
    Returns:
        Tuple of (train_generator, val_generator, test_generator, class_indices)
    """
    try:
        # Print directory structure for debugging
        print("\nDirectory structure:")
        for root, dirs, files in os.walk(data_dir):
            level = root.replace(data_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for d in dirs[:3]:  # Show first 3 subdirectories
                print(f"{subindent}{d}/")
            if len(dirs) > 3:
                print(f"{subindent}... and {len(dirs) - 3} more directories")
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2  # Use 20% of training data for validation
        )
        
        # Validation and test data generator (no augmentation)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create data generators
        print("\nCreating training generator...")
        train_generator = train_datagen.flow_from_directory(
            os.path.join(data_dir, 'train'),
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            seed=SEED
        )
        
        print("\nCreating validation generator...")
        val_generator = train_datagen.flow_from_directory(
            os.path.join(data_dir, 'train'),
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            seed=SEED
        )
        
        print("\nCreating test generator...")
        test_generator = val_test_datagen.flow_from_directory(
            os.path.join(data_dir, 'test'),
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Get class indices
        class_indices = train_generator.class_indices
        
        # Print class information
        class_names = {v: k for k, v in class_indices.items()}
        
        print(f"\n✅ Found {len(class_indices)} classes:")
        for i, (class_name, idx) in enumerate(class_indices.items()):
            print(f"  {i+1}. {class_name} (idx: {idx})")
        
        # Save class mapping
        os.makedirs('models', exist_ok=True)
        with open('models/class_mapping.json', 'w') as f:
            json.dump(class_indices, f, indent=2)
        
        return train_generator, val_generator, test_generator, class_indices
        
    except Exception as e:
        print(f"\n❌ Error creating data generators: {str(e)}")
        print("\nPossible solutions:")
        print("1. Make sure the dataset is properly downloaded and extracted")
        print("2. Check that the directory structure is correct")
        print("3. Verify that there are image files in the directories")
        print("\nDirectory structure should be:")
        print("data/")
        print("  plantvillage/")
        print("    train/")
        print("      class1/")
        print("        image1.jpg")
        print("        ...")
        print("      class2/")
        print("    test/")
        print("      class1/")
        print("      class2/")
        print("    val/")
        print("      class1/")
        print("      class2/")
        raise
        print("└── test/")
        print("    └── ...")
        exit(1)

def train_model(data_dir, output_dir, batch_size, img_size, epochs, learning_rate):
    """
    Train the model with the given parameters.
    
    Args:
        data_dir: Directory containing train/val/test subdirectories
        output_dir: Directory to save model and logs
        batch_size: Batch size for training
        img_size: Size of input images
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        
    Returns:
        Trained model and training history
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(output_dir, 'training_log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    try:
        # Create data generators
        logging.info("Creating data generators...")
        train_generator, val_generator, test_generator, class_indices = create_data_generators(
            data_dir, batch_size, img_size
        )
        
        # Save class indices
        with open(os.path.join(output_dir, 'class_indices.json'), 'w') as f:
            json.dump(class_indices, f, indent=2)
        
        # Create model
        num_classes = len(class_indices)
        logging.info(f"Creating model with {num_classes} classes...")
        model = create_model(
            num_classes=num_classes,
            img_size=img_size,
            dropout_rate=0.5
        )
        
        # Compile the model
        logging.info("Compiling model...")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')]
        )
        
        # Prepare model checkpoint
        checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}.h5')
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=checkpoint_path,
                save_best_only=True,
                save_weights_only=False,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            TensorBoard(
                log_dir=os.path.join(output_dir, 'logs'),
                histogram_freq=1
            )
        ]
        
        # Log model summary
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        logging.info("\n" + "\n".join(model_summary))
        
        # Train the model
        logging.info("Starting training...")
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = os.path.join(output_dir, 'final_model.h5')
        model.save(final_model_path)
        logging.info(f"Model saved to {final_model_path}")
        
        # Save training history
        history_path = os.path.join(output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history.history, f)
        
        return model, history, test_generator
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}", exc_info=True)
        raise

def evaluate_model(model, test_generator):
    """
    Evaluate the model on the test set and print metrics.
    
    Args:
        model: Trained Keras model
        test_generator: Test data generator
    """
    print("\n🧪 Evaluating on test set...")
    
    try:
        # Evaluate the model
        results = model.evaluate(test_generator, verbose=1, return_dict=True)
        
        # Print metrics
        print("\n📊 Test Results:")
        for metric_name, value in results.items():
            print(f"- {metric_name.capitalize()}: {value:.4f}")
            
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        print("\nTrying alternative evaluation method...")
        
        # Fallback to basic evaluation if the above fails
        try:
            loss, accuracy = model.evaluate(test_generator, verbose=1)
            print("\n📊 Basic Test Results:")
            print(f"- Loss: {loss:.4f}")
            print(f"- Accuracy: {accuracy:.4f}")
            results = {'loss': loss, 'accuracy': accuracy}
        except Exception as e2:
            print(f"\n❌ Could not evaluate model: {str(e2)}")
            results = {}
    
    return results

def plot_training_history(history, output_dir):
    """
    Plot training history and save the plots.
    
    Args:
        history: Training history object from model.fit()
        output_dir: Directory to save the plots
    """
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'plots', 'training_history.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Training plots saved to {plot_path}")

def main():
    # Set console output encoding to UTF-8
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    parser = argparse.ArgumentParser(description='Train a plant disease classification model')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to the dataset directory (should contain train/val/test subdirectories)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for models and logs (default: output)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Image size (width=height) for model input (default: 224)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Initial learning rate (default: 1e-3)')
    
    args = parser.parse_args()
    
    print("=== Plant Disease Classification Model Training ===")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Epochs: {args.epochs}")
    print(f"📊 Learning rate: {args.learning_rate}")
    print("=" * 50 + "\n")
    
    # Train the model
    model, history, test_generator = train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    # Evaluate the model
    evaluate_model(model, test_generator)
    
    # Plot training history
    plot_training_history(history, args.output_dir)
    
    print("\n✨ Training completed successfully!")

if __name__ == "__main__":
    main()
