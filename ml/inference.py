import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
import time
import logging
from PIL import Image, UnidentifiedImageError
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CropHealthPredictor:
    """
    A class for making predictions on crop health using a trained deep learning model.
    """
    
    def __init__(self, model_path: str, class_mapping_path: str = 'class_mapping.json',
                 img_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the predictor with a trained model and class mapping.
        
        Args:
            model_path: Path to the saved model file (h5 or saved_model format)
            class_mapping_path: Path to the JSON file containing class mapping
            img_size: Target image size (height, width) for model input
        ""
        self.model_path = model_path
        self.class_mapping_path = class_mapping_path
        self.img_size = img_size
        self.model = None
        self.class_mapping = {}
        self.inverse_class_mapping = {}
        self.loaded = False
        
        # Load model and class mapping
        self.load()
    
    def load(self) -> None:
        """Load the model and class mapping."""
        start_time = time.time()
        
        try:
            # Load model
            logger.info(f"Loading model from {self.model_path}...")
            self.model = tf.keras.models.load_model(self.model_path)
            
            # Load class mapping
            self._load_class_mapping()
            
            # Create inverse mapping for faster lookup
            self.inverse_class_mapping = {v: k for k, v in self.class_mapping.items()}
            
            # Verify model output shape matches number of classes
            if hasattr(self.model, 'output_shape') and self.model.output_shape[1] != len(self.class_mapping):
                logger.warning(
                    f"Model output shape {self.model.output_shape[1]} "
                    f"doesn't match number of classes {len(self.class_mapping)}"
                )
            
            self.loaded = True
            logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds. "
                       f"Number of classes: {len(self.class_mapping)}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.loaded = False
            raise
    
    def _load_class_mapping(self) -> None:
        """Load class mapping from JSON file."""
        try:
            with open(self.class_mapping_path, 'r') as f:
                self.class_mapping = json.load(f)
            logger.info(f"Loaded class mapping with {len(self.class_mapping)} classes")
        except FileNotFoundError:
            logger.warning(f"Class mapping file not found at {self.class_mapping_path}")
            self.class_mapping = {}
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in class mapping file: {self.class_mapping_path}")
            self.class_mapping = {}
    
    def preprocess_image(self, img_input: Union[str, np.ndarray, bytes]) -> Optional[np.ndarray]:
        """
        Preprocess an image for model inference.
        
        Args:
            img_input: Can be a file path, numpy array, or bytes
            
        Returns:
            Preprocessed image as a numpy array or None if processing fails
        """
        try:
            # Handle different input types
            if isinstance(img_input, str):  # File path
                img = image.load_img(img_input, target_size=self.img_size)
            elif isinstance(img_input, bytes):  # Bytes/uploaded file
                img = Image.open(io.BytesIO(img_input)).convert('RGB')
                img = img.resize(self.img_size, Image.LANCZOS)
            elif isinstance(img_input, np.ndarray):  # Numpy array
                img = Image.fromarray(img_input).convert('RGB')
                img = img.resize(self.img_size, Image.LANCZOS)
            else:
                raise ValueError(f"Unsupported image input type: {type(img_input)}")
            
            # Convert to array and normalize
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0  # Normalize to [0,1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
            return img_array
            
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            logger.error(f"Error loading image: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return None
    
    def predict(self, img_input: Union[str, np.ndarray, bytes], 
               top_k: int = 5) -> Dict[str, any]:
        """
        Make a prediction on an image.
        
        Args:
            img_input: Image file path, numpy array, or bytes
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary containing prediction results
        """
        if not self.loaded:
            return {"error": "Model not loaded", "success": False}
        
        start_time = time.time()
        
        try:
            # Preprocess the image
            processed_img = self.preprocess_image(img_input)
            if processed_img is None:
                return {"error": "Failed to process image", "success": False}
            
            # Make prediction
            predictions = self.model.predict(processed_img, verbose=0)
            prediction_time = time.time() - start_time
            
            # Get top K predictions
            top_indices = np.argsort(predictions[0])[-top_k:][::-1]
            top_predictions = []
            
            for idx in top_indices:
                class_name = self.inverse_class_mapping.get(str(idx), f"class_{idx}")
                confidence = float(predictions[0][idx])
                top_predictions.append({
                    "class_name": class_name,
                    "class_id": int(idx),
                    "confidence": confidence,
                    "confidence_percent": round(confidence * 100, 2)
                })
            
            # Get the top prediction
            top_pred = top_predictions[0] if top_predictions else None
            
            # Prepare response
            response = {
                "success": True,
                "predictions": top_predictions,
                "top_prediction": top_pred,
                "prediction_time_seconds": round(prediction_time, 4),
                "model": os.path.basename(self.model_path),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add additional metadata if input is a file path
            if isinstance(img_input, str):
                response["image_path"] = img_input
                response["file_name"] = os.path.basename(img_input)
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            return {
                "error": f"Prediction failed: {str(e)}",
                "success": False
            }
    
    def get_class_names(self) -> List[str]:
        """Get a list of all class names."""
        return list(self.class_mapping.keys())
    
    def get_class_id(self, class_name: str) -> Optional[int]:
        """Get the class ID for a given class name."""
        return self.class_mapping.get(class_name)
    
    def get_class_name(self, class_id: int) -> str:
        """Get the class name for a given class ID."""
        return self.inverse_class_mapping.get(str(class_id), f"class_{class_id}")


def load_predictor(model_dir: str = 'models', model_name: str = 'best_model.h5') -> CropHealthPredictor:
    """
    Helper function to load a predictor with default paths.
    
    Args:
        model_dir: Directory containing the model file
        model_name: Name of the model file
        
    Returns:
        An instance of CropHealthPredictor
    """
    model_path = os.path.join(model_dir, model_name)
    class_mapping_path = os.path.join('class_mapping.json')
    
    if not os.path.exists(model_path):
        # Try to find any .h5 file if the specified one doesn't exist
        model_files = list(Path(model_dir).glob('*.h5'))
        if model_files:
            model_path = str(model_files[0])
            logger.warning(f"Using model: {model_path}")
    
    return CropHealthPredictor(
        model_path=model_path,
        class_mapping_path=class_mapping_path
    )


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Crop Health Prediction')
    parser.add_argument('image_path', type=str, help='Path to the image file for prediction')
    parser.add_argument('--model', type=str, default='models/best_model.h5',
                       help='Path to the trained model')
    parser.add_argument('--class-mapping', type=str, default='class_mapping.json',
                       help='Path to the class mapping JSON file')
    parser.add_argument('--top-k', type=int, default=3,
                       help='Number of top predictions to show')
    
    args = parser.parse_args()
    
    # Load predictor
    try:
        predictor = CropHealthPredictor(
            model_path=args.model,
            class_mapping_path=args.class_mapping
        )
        
        # Make prediction
        result = predictor.predict(args.image_path, top_k=args.top_k)
        
        # Print results
        if result.get('success', False):
            print("\nPrediction Results:")
            print(f"Image: {result.get('file_name', args.image_path)}")
            print(f"Top Prediction: {result['top_prediction']['class_name']} "
                  f"({result['top_prediction']['confidence_percent']}%)")
            
            if args.top_k > 1:
                print("\nTop Predictions:")
                for i, pred in enumerate(result['predictions'], 1):
                    print(f"{i}. {pred['class_name']}: {pred['confidence_percent']}%")
            
            print(f"\nPrediction time: {result['prediction_time_seconds']:.2f} seconds")
            
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")

def save_class_mapping(generator, save_path='class_mapping.json'):
    """
    Save class mapping from a Keras ImageDataGenerator.
    
    Args:
        generator: Keras ImageDataGenerator with class_indices attribute
        save_path (str): Path to save the class mapping JSON file
    """
    # Invert the mapping from {class_name: index} to {index: class_name}
    class_mapping = {str(v): k for k, v in generator.class_indices.items()}
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    # Save to JSON file
    with open(save_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    print(f"✅ Class mapping saved to {save_path}")
    return class_mapping
    class_mapping = {str(v): k for k, v in generator.class_indices.items()}
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save to JSON file
    with open(save_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    print(f"✅ Class mapping saved to {save_path}")

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Crop Health Prediction')
    parser.add_argument('image_path', type=str, help='Path to the image file for prediction')
    parser.add_argument('--model', type=str, default='models/crop_health_model',
                       help='Path to the trained model')
    parser.add_argument('--class-mapping', type=str, default='class_mapping.json',
                       help='Path to the class mapping JSON file')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = CropHealthPredictor(args.model, args.class_mapping)
    
    # Make prediction
    result = predictor.predict(args.image_path)
    
    # Print results
    print("\n🔮 Prediction Results:")
    print(f"Image: {result['image_path']}")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    
    # Print top 5 classes
    print("\nTop 5 Predictions:")
    sorted_probs = sorted(
        result['class_probabilities'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:5]
    
    for class_name, prob in sorted_probs:
        print(f"  {class_name}: {prob:.2%}")
