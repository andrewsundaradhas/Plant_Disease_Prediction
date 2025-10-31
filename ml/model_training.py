import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)
from datetime import datetime
import matplotlib.pyplot as plt

def create_model(num_classes=38, img_size=224):
    """
    Create a CNN model using EfficientNetB0 as base
    """
    # Load pre-trained model without top layers
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def create_data_generators(train_dir, val_dir, test_dir, batch_size=32, img_size=224):
    """
    Create data generators with augmentation for training and validation
    """
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation and test data generator (only rescaling)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Save class indices
    class_indices = train_generator.class_indices
    with open('class_mapping.json', 'w') as f:
        json.dump(class_indices, f)
    
    return train_generator, validation_generator, test_generator, class_indices

def train_model(train_dir, val_dir, test_dir, epochs=50, batch_size=32, img_size=224):
    """
    Train the model with callbacks and save the best model
    """
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Create data generators
    train_generator, validation_generator, test_generator, class_indices = create_data_generators(
        train_dir, val_dir, test_dir, batch_size, img_size
    )
    
    # Create model
    model = create_model(num_classes=len(class_indices), img_size=img_size)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    # Callbacks
    log_dir = os.path.join('logs', 'fit', datetime.now().strftime('%Y%m%d-%H%M%S'))
    callbacks = [
        ModelCheckpoint(
            filepath='models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq='batch'
        )
    ]
    
    # Train the model
    print("🚀 Starting model training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    model.save('models/final_model.h5')
    print("✅ Final model saved to models/final_model.h5")
    
    # Evaluate on test set
    print("\n🧪 Evaluating on test set...")
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_generator)
    print(f"\n📊 Test Accuracy: {test_accuracy:.4f}")
    print(f"📊 Test Precision: {test_precision:.4f}")
    print(f"📊 Test Recall: {test_recall:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    return history

def plot_training_history(history):
    """
    Plot training and validation metrics
    """
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
    os.makedirs('results', exist_ok=True)
    plt.tight_layout()
    plt.savefig('results/training_history.png')
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a crop health prediction model')
    parser.add_argument('--train_dir', type=str, default='data/processed/train',
                        help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, default='data/processed/val',
                        help='Path to validation data directory')
    parser.add_argument('--test_dir', type=str, default='data/processed/test',
                        help='Path to test data directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size (width=height)')
    
    args = parser.parse_args()
    
    train_model(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size
    )
