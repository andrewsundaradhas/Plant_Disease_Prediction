"""
Model service for handling ML model loading and inference.
"""
import os
import json
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoadingError(Exception):
    """Exception raised when there is an error loading the model."""
    pass

class ModelInferenceError(Exception):
    """Exception raised when there is an error during model inference."""
    pass

class CropHealthModel:
    """Wrapper class for the crop health prediction model."""
    
    def __init__(self, model_path: str, class_mapping_path: str):
        """Initialize the model wrapper.
        
        Args:
            model_path: Path to the saved model directory
            class_mapping_path: Path to the JSON file containing class mapping
        """
        self.model_path = model_path
        self.class_mapping_path = class_mapping_path
        self.model = None
        self.class_names = []
        self.input_shape = (224, 224)  # Default input size for most CNN models
        
        # Load model and class mapping
        self._load_model()
        self._load_class_mapping()
    
    def _load_model(self):
        """Load the trained model from disk."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Check if the model exists
            if not os.path.exists(self.model_path):
                raise ModelLoadingError(f"Model not found at {self.model_path}")
            
            # Load the model
            self.model = tf.keras.models.load_model(self.model_path)
            
            # Get input shape from the model
            if hasattr(self.model, 'input_shape') and self.model.input_shape[1:3]:
                self.input_shape = self.model.input_shape[1:3]
                
            logger.info(f"Model loaded successfully. Input shape: {self.input_shape}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise ModelLoadingError(f"Failed to load model: {str(e)}")
    
    def _load_class_mapping(self):
        """Load class mapping from JSON file."""
        try:
            logger.info(f"Loading class mapping from {self.class_mapping_path}")
            
            # Check if the file exists
            if not os.path.exists(self.class_mapping_path):
                raise ModelLoadingError(f"Class mapping file not found at {self.class_mapping_path}")
            
            # Load class mapping
            with open(self.class_mapping_path, 'r') as f:
                class_mapping = json.load(f)
            
            # Convert to list of class names
            self.class_names = [""] * len(class_mapping)
            for class_name, idx in class_mapping.items():
                self.class_names[idx] = class_name
                
            logger.info(f"Loaded {len(self.class_names)} classes")
            
        except Exception as e:
            logger.error(f"Error loading class mapping: {str(e)}")
            raise ModelLoadingError(f"Failed to load class mapping: {str(e)}")
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Preprocess image data for model inference.
        
        Args:
            image_data: Binary image data
            
        Returns:
            Preprocessed image as a numpy array
        """
        try:
            # Load image from bytes
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image
            image = image.resize(self.input_shape)
            
            # Convert to numpy array and normalize
            image_array = np.array(image) / 255.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ModelInferenceError(f"Failed to preprocess image: {str(e)}")
    
    def predict(self, image_data: bytes) -> Dict[str, any]:
        """Make a prediction on a single image.
        
        Args:
            image_data: Binary image data
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_data)
            
            # Make prediction
            predictions = self.model.predict(processed_image)
            
            # Get top prediction
            predicted_class_idx = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = self.class_names[predicted_class_idx]
            
            # Get all class probabilities
            class_probabilities = {
                self.class_names[i]: float(prob) 
                for i, prob in enumerate(predictions[0])
            }
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'class_probabilities': class_probabilities,
                'all_predictions': predictions.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise ModelInferenceError(f"Prediction failed: {str(e)}")
    
    def batch_predict(self, image_data_list: List[bytes]) -> List[Dict[str, any]]:
        """Make predictions on a batch of images.
        
        Args:
            image_data_list: List of binary image data
            
        Returns:
            List of prediction results
        """
        try:
            # Preprocess all images
            processed_images = [self.preprocess_image(img_data) for img_data in image_data_list]
            
            # Stack images into a batch
            batch = np.vstack(processed_images)
            
            # Make predictions
            predictions = self.model.predict(batch)
            
            # Process predictions
            results = []
            for pred in predictions:
                predicted_class_idx = int(np.argmax(pred))
                confidence = float(pred[predicted_class_idx])
                predicted_class = self.class_names[predicted_class_idx]
                
                class_probabilities = {
                    self.class_names[i]: float(prob) 
                    for i, prob in enumerate(pred)
                }
                
                results.append({
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'class_probabilities': class_probabilities,
                    'all_predictions': pred.tolist()
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error during batch prediction: {str(e)}")
            raise ModelInferenceError(f"Batch prediction failed: {str(e)}")

# Global model instance
_model_instance = None

def get_model() -> CropHealthModel:
    """Get the global model instance (singleton pattern)."""
    global _model_instance
    if _model_instance is None:
        from ...core.config import settings
        _model_instance = CropHealthModel(
            model_path=settings.MODEL_PATH,
            class_mapping_path=settings.CLASS_MAPPING_PATH
        )
    return _model_instance

def predict_image(image_data: bytes) -> Dict[str, any]:
    """Make a prediction on a single image.
    
    Args:
        image_data: Binary image data
        
    Returns:
        Dictionary containing prediction results
    """
    model = get_model()
    return model.predict(image_data)

def predict_batch(image_data_list: List[bytes]) -> List[Dict[str, any]]:
    """Make predictions on a batch of images.
    
    Args:
        image_data_list: List of binary image data
        
    Returns:
        List of prediction results
    """
    model = get_model()
    return model.batch_predict(image_data_list)
