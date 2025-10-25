import tensorflow as tf
from tensorflow.keras import layers, models, applications
from typing import Tuple, Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 10,
    base_model_name: str = 'EfficientNetB0',
    freeze_base: bool = True,
    dropout_rate: float = 0.5,
    dense_units: int = 1024
) -> tf.keras.Model:
    """
    Create a transfer learning model for plant disease classification.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
        base_model_name: Name of the base model to use (EfficientNetB0, ResNet50, etc.)
        freeze_base: Whether to freeze the base model weights
        dropout_rate: Dropout rate for regularization
        dense_units: Number of units in the dense layer
        
    Returns:
        A compiled Keras model
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Data augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    
    # Base model (pre-trained on ImageNet)
    base_models = {
        'EfficientNetB0': applications.EfficientNetB0,
        'ResNet50': applications.ResNet50,
        'MobileNetV2': applications.MobileNetV2,
        'DenseNet121': applications.DenseNet121
    }
    
    if base_model_name not in base_models:
        raise ValueError(f"Unsupported base model: {base_model_name}. "
                         f"Choose from: {', '.join(base_models.keys())}")
    
    base_model = base_models[base_model_name](
        include_top=False,
        weights='imagenet',
        input_tensor=x,
        pooling='avg'
    )
    
    # Freeze the base model
    base_model.trainable = not freeze_base
    
    # Add custom head
    x = base_model.output
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 1e-4,
    optimizer: str = 'adam',
    metrics: Optional[list] = None
) -> tf.keras.Model:
    """
    Compile the model with the specified optimizer and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for the optimizer
        optimizer: Name of the optimizer to use
        metrics: List of metrics to track during training
        
    Returns:
        The compiled Keras model
    """
    if metrics is None:
        metrics = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    
    # Define optimizer
    if optimizer.lower() == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer.lower() == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer.lower() == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    
    # Compile model
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=metrics
    )
    
    return model

def get_callbacks(
    checkpoint_path: str = 'models/best_model.h5',
    log_dir: str = 'logs/fit',
    patience: int = 10,
    monitor: str = 'val_loss',
    mode: str = 'min'
) -> list:
    """
    Get a list of callbacks for model training.
    
    Args:
        checkpoint_path: Path to save the best model
        log_dir: Directory to save TensorBoard logs
        patience: Number of epochs with no improvement before early stopping
        monitor: Metric to monitor for early stopping and model checkpointing
        mode: One of {'min', 'max'}. Whether to minimize or maximize the monitored metric
        
    Returns:
        A list of Keras callbacks
    """
    from pathlib import Path
    
    # Create directories if they don't exist
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        # Save the best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            mode=mode,
            verbose=1
        ),
        
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode=mode
        ),
        
        # Learning rate reducer
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        
        # TensorBoard
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq='batch',
            profile_batch=0  # Disable profiling to avoid overhead
        )
    ]
    
    return callbacks

def load_model(
    model_path: str,
    custom_objects: Optional[Dict[str, Any]] = None
) -> tf.keras.Model:
    """
    Load a saved Keras model.
    
    Args:
        model_path: Path to the saved model
        custom_objects: Dictionary mapping names (strings) to custom classes or functions
                       to be considered during deserialization
        
    Returns:
        A Keras model instance
    """
    if custom_objects is None:
        custom_objects = {}
    
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    model = create_model(
        input_shape=(224, 224, 3),
        num_classes=10,
        base_model_name='EfficientNetB0',
        freeze_base=True
    )
    
    model = compile_model(model)
    model.summary()
    
    # Get callbacks
    callbacks = get_callbacks()
    print(f"Using {len(callbacks)} callbacks:")
    for callback in callbacks:
        print(f"- {callback.__class__.__name__}")
