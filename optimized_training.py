import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
    TerminateOnNaN
)
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import sys

# Configure TensorFlow to use CPU efficiently
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce logging
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count() // 2)  # Use half the CPU cores

# Configure TensorFlow to use CPU
tf.config.set_visible_devices([], 'GPU')  # Disable GPU

print(f"Using TensorFlow {tf.__version__}")
print(f"Number of CPU cores available: {os.cpu_count()}")
print(f"Number of CPU threads being used: {os.environ['OMP_NUM_THREADS']}")

# Configuration
config = {
    'data_dir': 'data/PlantVillage',  # Update this path
    'img_size': (128, 128),  # Reduced size for CPU
    'batch_size': 16,  # Smaller batch size for CPU
    'epochs': 50,
    'learning_rate': 1e-3,
    'min_lr': 1e-6,
    'patience': 8,
    'output_dir': 'optimized_output',
    'model_name': 'MobileNetV2',  # Lighter model for CPU
    'augmentation': True,
    'use_class_weights': True
}

# Create output directory
os.makedirs(config['output_dir'], exist_ok=True)

# Save config
with open(f"{config['output_dir']}/config.json", 'w') as f:
    json.dump(config, f, indent=2)

# Data augmentation
def create_generators():
    if config['augmentation']:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
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

    # Save class indices
    class_indices = train_generator.class_indices
    with open(f"{config['output_dir']}/class_indices.json", 'w') as f:
        json.dump(class_indices, f)
    
    return train_generator, validation_generator, class_indices

def calculate_class_weights(generator):
    """Calculate class weights to handle imbalanced classes."""
    from sklearn.utils.class_weight import compute_class_weight
    
    # Get all labels
    labels = []
    for i in range(len(generator)):
        _, batch_labels = generator[i]
        labels.extend(np.argmax(batch_labels, axis=1))
        if i >= 10:  # Only check first few batches for efficiency
            break
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    return dict(enumerate(class_weights))

def create_model():
    """Create a model with transfer learning."""
    # Use a smaller base model for CPU
    base_model = applications.MobileNetV2(
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
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(len(class_indices), activation='softmax')
    ])
    
    return model, base_model

def train():
    # Create data generators
    print("Creating data generators...")
    train_generator, validation_generator, _ = create_generators()
    
    # Calculate class weights if needed
    class_weights = None
    if config['use_class_weights']:
        print("Calculating class weights...")
        class_weights = calculate_class_weights(train_generator)
        print("Class weights:", class_weights)
    
    # Create model
    print(f"Creating {config['model_name']} model...")
    model, base_model = create_model()
    
    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config['learning_rate'],
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )
    
    # Compile model
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
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
        CSVLogger(
            f"{config['output_dir']}/training_log.csv",
            append=True
        ),
        TerminateOnNaN()
    ]
    
    # Train the model
    print("Starting training...")
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
    
    # Fine-tuning: Unfreeze some layers
    print("Starting fine-tuning...")
    base_model.trainable = True
    for layer in base_model.layers[:100]:  # Keep first 100 layers frozen
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=config['learning_rate']/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training
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
    
    # Save the final model
    model.save(f"{config['output_dir']}/final_model.h5")
    
    # Plot training history
    plot_history(history.history, 'initial_training')
    if 'history_fine' in locals():
        plot_history(history_fine.history, 'fine_tuning')
    
    print("Training completed successfully!")

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
    print("Starting training with configuration:")
    for key, value in config.items():
        print(f"- {key}: {value}")
    
    # Check data directory
    if not os.path.exists(config['data_dir']):
        print(f"Error: Data directory not found at {config['data_dir']}")
        print("Please update the 'data_dir' in the config.")
        sys.exit(1)
    
    try:
        train()
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
