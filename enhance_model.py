import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    CSVLogger, TerminateOnNaN, LearningRateScheduler
)
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from datetime import datetime

# Configure TensorFlow for optimal CPU performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce logging
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count() // 2)

# Configuration
config = {
    'data_dir': 'data/PlantVillage',  # Update this if your data is elsewhere
    'img_size': (224, 224),          # Standard size for most models
    'batch_size': 16,                # Smaller batches for CPU
    'epochs': 50,                    # Max epochs
    'initial_lr': 1e-4,              # Start with lower learning rate for fine-tuning
    'min_lr': 1e-7,                  # Minimum learning rate
    'patience': 10,                  # Early stopping patience
    'output_dir': 'enhanced_output',  # New output directory
    'pretrained_model': 'output/final_model.h5',  # Your existing model
    'use_augmentation': True
}

# Create output directory
os.makedirs(config['output_dir'], exist_ok=True)

print(f"Using TensorFlow {tf.__version__}")
print(f"CPU Cores: {os.cpu_count()}")
print(f"Model: {config['pretrained_model']}")

def create_data_generators():
    """Create data generators with augmentation."""
    if config['use_augmentation']:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
    else:
        train_datagen = ImageDataGenerator(
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
    validation_generator = train_datagen.flow_from_directory(
        config['data_dir'],
        target_size=config['img_size'],
        batch_size=config['batch_size'],
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights = dict(enumerate(class_weights))
    print("Class weights:", class_weights)
    
    # Save class indices
    with open(f"{config['output_dir']}/class_indices.json", 'w') as f:
        json.dump(train_generator.class_indices, f)
    
    return train_generator, validation_generator, class_weights

def load_and_enhance_model(num_classes):
    """Create a new model with the correct number of output classes."""
    print(f"\nCreating a new model with {num_classes} output classes")
    
    # Use MobileNetV2 as base model
    base_model = applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=config['img_size'] + (3,),
        pooling='avg'
    )
    
    # Create new model on top
    model = models.Sequential([
        base_model,
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=config['initial_lr']),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
    )
    
    return model, base_model

def train():
    """Main training function."""
    print("\n===== Setting up data generators =====")
    train_generator, validation_generator, class_weights = create_data_generators()
    
    print("\n===== Creating and training model =====")
    model, base_model = load_and_enhance_model(num_classes=len(train_generator.class_indices))
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=f"{config['output_dir']}/enhanced_model.h5",
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
        CSVLogger(f"{config['output_dir']}/training_log.csv"),
        TerminateOnNaN()
    ]
    
    print("\n===== Starting model enhancement =====")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=config['epochs'],
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Save final model
    model.save(f"{config['output_dir']}/final_enhanced_model.keras")
    
    # Plot training history
    plot_history(history.history)
    
    print("\n===== Model enhancement completed! =====")
    print(f"Enhanced model saved to {config['output_dir']}")

def plot_history(history):
    """Plot training history."""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.get('accuracy', []), label='Training Accuracy')
    if 'val_accuracy' in history:
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.get('loss', []), label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Save the figure
    plt.tight_layout()
    plot_path = f"{config['output_dir']}/training_history.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Training history plot saved to {plot_path}")

if __name__ == "__main__":
    print("===== Starting Model Enhancement =====")
    print("Configuration:")
    for key, value in config.items():
        print(f"- {key}: {value}")
    
    # Check if pretrained model exists
    if not os.path.exists(config['pretrained_model']):
        print(f"\nError: Pretrained model not found at {config['pretrained_model']}")
        sys.exit(1)
    
    try:
        train()
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
