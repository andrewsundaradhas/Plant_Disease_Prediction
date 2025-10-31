import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import numpy as np
import logging

# Import local modules
from .data_loader import PlantVillageDataLoader
from .model import create_model, compile_model, get_callbacks

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingTimeCallback(Callback):
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
    
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        logger.info("Training started")
    
    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        avg_epoch_time = np.mean(self.epoch_times) if self.epoch_times else 0
        logger.info(f"Training completed in {total_time:.2f} seconds")
        logger.info(f"Average time per epoch: {avg_epoch_time:.2f} seconds")
        self.training_metrics = {
            'total_training_time': total_time,
            'avg_epoch_time': avg_epoch_time,
            'epoch_times': self.epoch_times
        }

def train_model(
    data_dir: str,
    output_dir: str = 'models',
    model_name: str = 'plant_disease_model',
    base_model: str = 'EfficientNetB0',
    img_size: tuple = (224, 224),
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    freeze_base: bool = True,
    dropout_rate: float = 0.5,
    dense_units: int = 1024,
    test_split: float = 0.2,
    val_split: float = 0.2,
    seed: int = 42,
    use_class_weights: bool = True
) -> dict:
    """
    Train a plant disease classification model.
    
    Args:
        data_dir: Directory containing the training data
        output_dir: Directory to save the trained model and logs
        model_name: Name of the model
        base_model: Base model architecture to use
        img_size: Image size (height, width)
        batch_size: Batch size for training
        epochs: Maximum number of epochs to train for
        learning_rate: Initial learning rate
        freeze_base: Whether to freeze the base model weights
        dropout_rate: Dropout rate for regularization
        dense_units: Number of units in the dense layer
        test_split: Fraction of data to use for testing
        val_split: Fraction of training data to use for validation
        seed: Random seed for reproducibility
        use_class_weights: Whether to use class weights for imbalanced classes
        
    Returns:
        Dictionary containing training history and metrics
    """
    # Create output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(output_dir) / f"{model_name}_{timestamp}"
    model_dir = run_dir / 'saved_model'
    logs_dir = run_dir / 'logs'
    
    model_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {run_dir}")
    
    # Initialize data loader
    logger.info("Loading data...")
    data_loader = PlantVillageDataLoader(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        test_split=test_split,
        val_split=val_split,
        seed=seed
    )
    
    # Get data generators
    train_gen, val_gen, _ = data_loader.get_data_generators()
    
    # Get class weights
    class_weights = data_loader.get_class_weights() if use_class_weights else None
    
    # Create and compile model
    logger.info("Creating model...")
    model = create_model(
        input_shape=(*img_size, 3),
        num_classes=data_loader.get_num_classes(),
        base_model_name=base_model,
        freeze_base=freeze_base,
        dropout_rate=dropout_rate,
        dense_units=dense_units
    )
    
    model = compile_model(model, learning_rate=learning_rate)
    
    # Get callbacks
    checkpoint_path = str(model_dir / 'best_model.h5')
    tensorboard_logs = str(logs_dir / 'tensorboard')
    
    callbacks = get_callbacks(
        checkpoint_path=checkpoint_path,
        log_dir=tensorboard_logs,
        patience=10,
        monitor='val_accuracy',
        mode='max'
    )
    
    # Add custom training time callback
    time_callback = TrainingTimeCallback()
    callbacks.append(time_callback)
    
    # Train the model
    logger.info("Starting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Save the final model
    final_model_path = model_dir / 'final_model.h5'
    model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save training history
    history_path = run_dir / 'training_history.json'
    history_dict = {
        'history': history.history,
        'params': history.params,
        'epoch': history.epoch,
        'training_metrics': getattr(time_callback, 'training_metrics', {})
    }
    
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    logger.info(f"Training history saved to {history_path}")
    
    # Save model architecture
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    with open(run_dir / 'model_summary.txt', 'w') as f:
        f.write('\n'.join(model_summary))
    
    # Save training configuration
    config = {
        'model_name': model_name,
        'base_model': base_model,
        'img_size': img_size,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'freeze_base': freeze_base,
        'dropout_rate': dropout_rate,
        'dense_units': dense_units,
        'test_split': test_split,
        'val_split': val_split,
        'seed': seed,
        'use_class_weights': use_class_weights,
        'num_classes': data_loader.get_num_classes(),
        'class_names': data_loader.get_class_names(),
        'timestamp': timestamp,
        'output_dir': str(run_dir)
    }
    
    config_path = run_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Training configuration saved to {config_path}")
    
    return {
        'model': model,
        'history': history.history,
        'config': config,
        'output_dir': str(run_dir),
        'model_path': str(final_model_path)
    }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a plant disease classification model.')
    
    # Required arguments
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing the training data')
    
    # Optional arguments
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Directory to save the trained model and logs')
    parser.add_argument('--model-name', type=str, default='plant_disease_model',
                        help='Name of the model')
    parser.add_argument('--base-model', type=str, default='EfficientNetB0',
                        choices=['EfficientNetB0', 'ResNet50', 'MobileNetV2', 'DenseNet121'],
                        help='Base model architecture to use')
    parser.add_argument('--img-size', type=int, nargs=2, default=[224, 224],
                        metavar=('HEIGHT', 'WIDTH'),
                        help='Image size (height, width)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of epochs to train for')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--no-freeze-base', action='store_false', dest='freeze_base',
                        help='Do not freeze the base model weights')
    parser.add_argument('--dropout-rate', type=float, default=0.5,
                        help='Dropout rate for regularization')
    parser.add_argument('--dense-units', type=int, default=1024,
                        help='Number of units in the dense layer')
    parser.add_argument('--test-split', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Fraction of training data to use for validation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no-class-weights', action='store_false', dest='use_class_weights',
                        help='Do not use class weights for imbalanced classes')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )
    
    # Set random seeds for reproducibility
    tf.keras.utils.set_random_seed(args.seed)
    
    try:
        # Train the model
        results = train_model(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_name=args.model_name,
            base_model=args.base_model,
            img_size=tuple(args.img_size),
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            freeze_base=args.freeze_base,
            dropout_rate=args.dropout_rate,
            dense_units=args.dense_units,
            test_split=args.test_split,
            val_split=args.val_split,
            seed=args.seed,
            use_class_weights=args.use_class_weights
        )
        
        logger.info(f"Training completed successfully. Model saved to {results['output_dir']}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise
