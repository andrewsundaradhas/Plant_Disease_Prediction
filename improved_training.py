import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import TopKCategoricalAccuracy
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_data_generators(data_dir, img_size=(224, 224), batch_size=32, val_split=0.2):
    """Create data generators with augmentation for training and validation."""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=val_split
    )

    # Validation data generator (only rescaling)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split
    )

    # Training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # Validation generator
    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Save class indices
    class_indices = train_generator.class_indices
    with open('output/class_indices.json', 'w') as f:
        json.dump(class_indices, f)

    return train_generator, validation_generator, class_indices

def create_model(num_classes, img_size=(224, 224, 3), base_model_name='EfficientNetB0'):
    """Create a model with transfer learning."""
    # Load pre-trained model
    base_model_class = getattr(applications, base_model_name)
    base_model = base_model_class(
        include_top=False,
        weights='imagenet',
        input_shape=img_size,
        pooling='avg'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create new model on top
    inputs = tf.keras.Input(shape=img_size)
    x = base_model(inputs, training=False)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model, base_model

def compile_model(model, learning_rate=1e-4):
    """Compile the model with optimizer and metrics."""
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[
            'accuracy',
            TopKCategoricalAccuracy(k=3, name='top3_accuracy'),
            TopKCategoricalAccuracy(k=5, name='top5_accuracy')
        ]
    )
    return model

def train_model():
    """Main training function."""
    try:
        # Configuration
        config = {
            'data_dir': 'data/plantvillage',
            'output_dir': 'output',
            'img_size': (224, 224),
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 1e-4,
            'patience': 10,
            'min_lr': 1e-6,
            'base_model': 'EfficientNetB0',
            'val_split': 0.2
        }
        
        # Create output directory
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # Save config
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Create data generators
        logger.info("Creating data generators...")
        train_gen, val_gen, class_indices = create_data_generators(
            data_dir=config['data_dir'],
            img_size=config['img_size'],
            batch_size=config['batch_size'],
            val_split=config['val_split']
        )
        
        # Create model
        logger.info(f"Creating model with {len(class_indices)} classes...")
        model, base_model = create_model(
            num_classes=len(class_indices),
            img_size=config['img_size'] + (3,),
            base_model_name=config['base_model']
        )
        
        # Compile model
        logger.info("Compiling model...")
        model = compile_model(model, learning_rate=config['learning_rate'])
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=str(output_dir / 'best_model.keras'),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=config['patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=config['patience'] // 2,
                min_lr=config['min_lr'],
                verbose=1
            ),
            TensorBoard(
                log_dir=str(output_dir / 'logs'),
                histogram_freq=1
            ),
            CSVLogger(
                filename=str(output_dir / 'training_log.csv'),
                append=True
            )
        ]
        
        # Train the model
        logger.info("Starting training...")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=config['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the final model
        final_model_path = output_dir / 'final_model.keras'
        model.save(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        # Plot training history
        plot_training_history(history, output_dir)
        
        return history
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

def plot_training_history(history, output_dir):
    """Plot training history."""
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
    
    # Save the figure
    plt.tight_layout()
    plot_path = output_dir / 'training_history.png'
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Training history plot saved to {plot_path}")

if __name__ == "__main__":
    # Enable memory growth for GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            logger.info("GPU memory growth enabled")
        except RuntimeError as e:
            logger.warning(f"Could not set GPU memory growth: {e}")
    
    # Run training
    train_model()
