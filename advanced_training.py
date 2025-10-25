import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    CSVLogger, TerminateOnNaN, LearningRateScheduler
)
from tensorflow.keras.optimizers import Adam, schedules
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from datetime import datetime

# Configure TensorFlow for optimal CPU performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce logging
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count() // 2)  # Use half CPU cores
tf.config.set_visible_devices([], 'GPU')  # Ensure we're using CPU

# Configuration
config = {
    'data_dir': 'data/PlantVillage',
    'img_size': (160, 160),  # Good balance between speed and accuracy
    'batch_size': 16,        # Smaller batches for CPU
    'epochs': 100,           # Max epochs (early stopping will likely stop earlier)
    'initial_lr': 1e-3,      # Initial learning rate
    'min_lr': 1e-6,          # Minimum learning rate
    'warmup_epochs': 5,      # Number of warmup epochs
    'patience': 10,          # Patience for early stopping
    'output_dir': 'advanced_output',
    'model_name': 'EfficientNetB0',
    'use_class_weights': True,
    'use_augmentation': True
}

# Create output directory
os.makedirs(config['output_dir'], exist_ok=True)

# Save config
with open(f"{config['output_dir']}/config.json", 'w') as f:
    json.dump(config, f, indent=2)

print(f"Using TensorFlow {tf.__version__}")
print(f"CPU Cores: {os.cpu_count()}")
print(f"Batch size: {config['batch_size']}")
print(f"Image size: {config['img_size']}")

# Learning rate schedule with warmup
def lr_schedule(epoch):
    """Learning rate schedule with warmup."""
    if epoch < config['warmup_epochs']:
        # Linear warmup
        return config['initial_lr'] * (epoch + 1) / config['warmup_epochs']
    else:
        # Cosine decay
        progress = (epoch - config['warmup_epochs']) / (config['epochs'] - config['warmup_epochs'])
        return config['min_lr'] + 0.5 * (config['initial_lr'] - config['min_lr']) * (1 + np.cos(np.pi * progress))

# Data augmentation
def get_data_generators():
    """Create data generators with augmentation."""
    if config['use_augmentation']:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=0.2
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )

    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # Training generator
    train_generator = train_datagen.flow_from_directory(
        config['data_dir'],
        target_size=config['img_size'],
        batch_size=config['batch_size'],
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # Validation generator
    validation_generator = val_datagen.flow_from_directory(
        config['data_dir'],
        target_size=config['img_size'],
        batch_size=config['batch_size'],
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Calculate class weights
    class_weights = None
    if config['use_class_weights']:
        # Get class indices and counts
        class_indices = train_generator.class_indices
        num_samples = len(train_generator.filenames)
        num_classes = len(class_indices)
        
        # Calculate class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.arange(num_classes),
            y=train_generator.classes
        )
        class_weights = dict(enumerate(class_weights))
        print("Class weights:", class_weights)
        
        # Save class indices
        with open(f"{config['output_dir']}/class_indices.json", 'w') as f:
            json.dump(class_indices, f)
    
    return train_generator, validation_generator, class_weights

def create_model(num_classes):
    """Create model with transfer learning."""
    # Load pre-trained model
    base_model = applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=config['img_size'] + (3,),
        pooling='avg'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Create new model on top
    model = models.Sequential([
        base_model,
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

def train():
    """Main training function."""
    print("\n===== Setting up data generators =====")
    train_generator, validation_generator, class_weights = get_data_generators()
    
    print("\n===== Creating model =====")
    num_classes = len(train_generator.class_indices)
    model, base_model = create_model(num_classes)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=config['initial_lr']),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=f"{config['output_dir']}/best_model.h5",
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=config['patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config['patience']//2,
            min_lr=config['min_lr'],
            verbose=1
        ),
        LearningRateScheduler(lr_schedule, verbose=1),
        CSVLogger(f"{config['output_dir']}/training_log.csv"),
        TerminateOnNaN()
    ]
    
    print("\n===== Starting initial training =====")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=config['epochs'],
        callbacks=callbacks,
        class_weight=class_weights,
        workers=4,
        use_multiprocessing=True,
        max_queue_size=10
    )
    
    # Fine-tuning
    print("\n===== Starting fine-tuning =====")
    base_model.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=config['initial_lr']/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune for fewer epochs
    history_fine = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=config['epochs'],
        initial_epoch=history.epoch[-1] + 1,
        callbacks=callbacks,
        class_weight=class_weights,
        workers=4,
        use_multiprocessing=True,
        max_queue_size=10
    )
    
    # Save final model
    model.save(f"{config['output_dir']}/final_model.h5")
    
    # Plot training history
    plot_history(history.history, 'initial_training')
    if 'history_fine' in locals():
        plot_history(history_fine.history, 'fine_tuning')
    
    print("\n===== Training completed successfully! =====")
    print(f"Model saved to {config['output_dir']}")

def plot_history(history, prefix=''):
    """Plot training history."""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.get('accuracy', []), label='Training Accuracy')
    if 'val_accuracy' in history:
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model Accuracy ({prefix})')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.get('loss', []), label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss ({prefix})')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Save the figure
    plt.tight_layout()
    plot_path = f"{config['output_dir']}/{prefix}_history.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Training history plot saved to {plot_path}")

if __name__ == "__main__":
    print("===== Starting Plant Disease Classification Training =====")
    print("Configuration:")
    for key, value in config.items():
        print(f"- {key}: {value}")
    
    # Check data directory
    if not os.path.exists(config['data_dir']):
        print(f"\nError: Data directory not found at {config['data_dir']}")
        print("Please update the 'data_dir' in the config.")
        sys.exit(1)
    
    try:
        train()
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
