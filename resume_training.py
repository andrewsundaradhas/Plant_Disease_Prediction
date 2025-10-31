import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('resume_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingTimeCallback(tf.keras.callbacks.Callback):
    """Callback to track and log training time."""
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self.start_time = None
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds")
        
        # Update training progress log
        with open('output/training_progress.log', 'a') as f:
            f.write(f"Epoch {epoch + 1} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}\n")

def load_training_history():
    """Load existing training history."""
    history_path = 'output/training_history.json'
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            return json.load(f)
    return None

def save_training_history(history, epochs_trained):
    """Save updated training history."""
    history_path = 'output/training_history.json'
    history_dict = {
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'learning_rate': history.history.get('lr', [0.001] * len(history.history['loss']))
    }
    
    # Save the updated history
    with open(history_path, 'w') as f:
        json.dump(history_dict, f)

def resume_training():
    try:
        from ml.data_loader import PlantVillageDataLoader
        from ml.model import create_model, compile_model
        
        # Configuration
        config = {
            'data_dir': 'data/plantvillage',
            'output_dir': 'output',
            'batch_size': 32,
            'img_size': (224, 224),
            'epochs': 30,  # Total epochs (including already trained)
            'initial_epoch': 1,  # Start from epoch 1 (0-based)
            'learning_rate': 1e-4,
            'patience': 10,  # For early stopping
            'min_delta': 1e-4,  # Minimum change to qualify as improvement
            'factor': 0.5,  # Factor for reducing learning rate
            'min_lr': 1e-6  # Minimum learning rate
        }
        
        # Load existing model or create a new one
        model_path = 'output/final_model.h5'
        if os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load data
        logger.info("Loading data...")
        data_loader = PlantVillageDataLoader(
            data_dir=config['data_dir'],
            img_size=config['img_size'],
            batch_size=config['batch_size']
        )
        
        train_gen, val_gen, _ = data_loader.get_data_generators()
        class_weights = data_loader.get_class_weights()
        
        # Set up callbacks
        callbacks = [
            ModelCheckpoint(
                filepath='output/checkpoints/model_epoch_{epoch:02d}.h5',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                save_weights_only=False,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=config['patience'],
                min_delta=config['min_delta'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=config['factor'],
                patience=config['patience'] // 2,
                min_lr=config['min_lr'],
                verbose=1
            ),
            TensorBoard(
                log_dir='output/logs',
                histogram_freq=1,
                update_freq='epoch'
            ),
            TrainingTimeCallback()
        ]
        
        # Compile model
        model = compile_model(model, learning_rate=config['learning_rate'])
        
        # Train the model
        logger.info("Resuming model training...")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=config['epochs'],
            initial_epoch=config['initial_epoch'],
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Save the final model
        final_model_path = 'output/final_model.h5'
        model.save(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        # Save training history
        save_training_history(history, config['epochs'])
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    import time
    logger.info("🚀 Resuming training...")
    start_time = time.time()
    
    try:
        resume_training()
        training_time = (time.time() - start_time) / 3600  # Convert to hours
        logger.info(f"✨ Training completed in {training_time:.2f} hours")
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        sys.exit(1)
