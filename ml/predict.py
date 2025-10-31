import os
import json
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
from PIL import Image
import io
import base64

# Import local modules
from .model import load_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlantDiseasePredictor:
    """Class for making predictions with a trained plant disease classification model."""
    
    def __init__(
        self, 
        model_path: str,
        class_names: Optional[List[str]] = None,
        img_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model
            class_names: List of class names in the same order as model outputs
            img_size: Input image size (height, width) expected by the model
        """
        self.model_path = model_path
        self.img_size = img_size
        self.class_names = class_names or []
        self.model = None
        
        # Load the model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained model."""
        try:
            logger.info(f"Loading model from {self.model_path}...")
            self.model = load_model(self.model_path)
            
            # If class names not provided, try to load from a config file
            if not self.class_names and hasattr(self.model, 'config'):
                try:
                    config = json.loads(self.model.config)
                    if 'class_names' in config:
                        self.class_names = config['class_names']
                        logger.info(f"Loaded {len(self.class_names)} class names from model config")
                except (AttributeError, json.JSONDecodeError):
                    pass
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_image(
        self, 
        image: Union[str, bytes, Image.Image, np.ndarray],
        format: str = None
    ) -> np.ndarray:
        """
        Preprocess an image for prediction.
        
        Args:
            image: Input image as file path, bytes, PIL Image, or numpy array
            format: Format of the input image ('file', 'bytes', 'pil', 'numpy')
            
        Returns:
            Preprocessed image as a numpy array
        """
        try:
            # Determine input type if not specified
            if format is None:
                if isinstance(image, str):
                    format = 'file'
                elif isinstance(image, bytes):
                    format = 'bytes'
                elif isinstance(image, Image.Image):
                    format = 'pil'
                elif isinstance(image, np.ndarray):
                    format = 'numpy'
                else:
                    raise ValueError("Could not determine input format. Please specify the 'format' parameter.")
            
            # Load image based on format
            if format == 'file':
                img = Image.open(image).convert('RGB')
            elif format == 'bytes':
                img = Image.open(io.BytesIO(image)).convert('RGB')
            elif format == 'pil':
                img = image.convert('RGB') if image.mode != 'RGB' else image
            elif format == 'numpy':
                if image.ndim == 3 and image.shape[2] == 3:
                    img = Image.fromarray(image)
                elif image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
                    img = Image.fromarray(image.squeeze()).convert('L').convert('RGB')
                else:
                    raise ValueError(f"Unsupported numpy array shape: {image.shape}")
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Resize and preprocess
            img = img.resize(self.img_size, Image.Resampling.BICUBIC)
            img_array = np.array(img) / 255.0
            
            # Add batch dimension if needed
            if len(img_array.shape) == 3:
                img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def predict(
        self, 
        image: Union[str, bytes, Image.Image, np.ndarray],
        format: str = None,
        top_k: int = 3,
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """
        Make a prediction on a single image.
        
        Args:
            image: Input image as file path, bytes, PIL Image, or numpy array
            format: Format of the input image ('file', 'bytes', 'pil', 'numpy')
            top_k: Number of top predictions to return
            return_probabilities: Whether to return probabilities for all classes
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Preprocess the image
            img_array = self.preprocess_image(image, format=format)
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            
            # Get top-k predictions
            top_k = min(top_k, predictions.shape[1])
            top_indices = np.argsort(predictions[0])[-top_k:][::-1]
            top_probs = predictions[0][top_indices]
            
            # Prepare results
            results = {
                'success': True,
                'predictions': [],
                'top_class': None,
                'top_probability': None,
                'all_probabilities': predictions[0].tolist() if return_probabilities else None,
                'class_names': self.class_names if self.class_names else []
            }
            
            # Add top-k predictions
            for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
                class_name = self.class_names[idx] if (self.class_names and idx < len(self.class_names)) else str(idx)
                results['predictions'].append({
                    'class_id': int(idx),
                    'class_name': class_name,
                    'probability': float(prob),
                    'rank': i + 1
                })
            
            if results['predictions']:
                results['top_class'] = results['predictions'][0]['class_name']
                results['top_probability'] = results['predictions'][0]['probability']
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'success': False,
                'error': str(e),
                'predictions': []
            }
    
    def predict_batch(
        self, 
        images: List[Union[str, bytes, Image.Image, np.ndarray]],
        format: str = None,
        top_k: int = 3,
        batch_size: int = 32,
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """
        Make predictions on a batch of images.
        
        Args:
            images: List of input images
            format: Format of the input images ('file', 'bytes', 'pil', 'numpy')
            top_k: Number of top predictions to return per image
            batch_size: Batch size for prediction
            return_probabilities: Whether to return probabilities for all classes
            
        Returns:
            Dictionary containing prediction results for all images
        """
        try:
            # Preprocess all images
            img_arrays = []
            valid_indices = []
            
            for i, img in enumerate(images):
                try:
                    img_array = self.preprocess_image(img, format=format)
                    img_arrays.append(img_array[0])  # Remove batch dimension
                    valid_indices.append(i)
                except Exception as e:
                    logger.warning(f"Error preprocessing image {i}: {e}")
            
            if not img_arrays:
                raise ValueError("No valid images provided for prediction")
            
            img_arrays = np.array(img_arrays)
            
            # Make predictions in batches
            predictions = []
            num_batches = (len(img_arrays) + batch_size - 1) // batch_size
            
            for i in range(0, len(img_arrays), batch_size):
                batch = img_arrays[i:i + batch_size]
                batch_preds = self.model.predict(batch, verbose=0)
                predictions.append(batch_preds)
            
            predictions = np.vstack(predictions)
            
            # Process predictions
            results = {
                'success': True,
                'predictions': [],
                'valid_indices': valid_indices
            }
            
            for i in range(len(valid_indices)):
                img_preds = predictions[i]
                top_k_indices = np.argsort(img_preds)[-top_k:][::-1]
                top_k_probs = img_preds[top_k_indices]
                
                img_results = []
                for j, (idx, prob) in enumerate(zip(top_k_indices, top_k_probs)):
                    class_name = self.class_names[idx] if (self.class_names and idx < len(self.class_names)) else str(idx)
                    img_results.append({
                        'class_id': int(idx),
                        'class_name': class_name,
                        'probability': float(prob),
                        'rank': j + 1
                    })
                
                results['predictions'].append({
                    'image_index': valid_indices[i],
                    'top_class': img_results[0]['class_name'] if img_results else None,
                    'top_probability': img_results[0]['probability'] if img_results else None,
                    'predictions': img_results,
                    'all_probabilities': img_preds.tolist() if return_probabilities else None
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            return {
                'success': False,
                'error': str(e),
                'predictions': []
            }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make predictions with a trained plant disease classification model.')
    
    # Required arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the input image or directory of images')
    
    # Optional arguments
    parser.add_argument('--class-names', type=str, nargs='+', default=None,
                        help='List of class names in the same order as model outputs')
    parser.add_argument('--img-size', type=int, nargs=2, default=[224, 224],
                        metavar=('HEIGHT', 'WIDTH'),
                        help='Input image size (height, width)')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Number of top predictions to return')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for prediction (when processing multiple images)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file to save predictions (JSON format)')
    parser.add_argument('--show-probabilities', action='store_true',
                        help='Show probabilities for all classes')
    
    return parser.parse_args()

def main():
    """Main function for command-line usage."""
    # Parse command line arguments
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize predictor
        predictor = PlantDiseasePredictor(
            model_path=args.model_path,
            class_names=args.class_names,
            img_size=tuple(args.img_size)
        )
        
        # Check if input is a directory or a single file
        if os.path.isdir(args.image):
            # Process directory of images
            image_paths = [os.path.join(args.image, f) for f in os.listdir(args.image) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            if not image_paths:
                raise ValueError(f"No image files found in directory: {args.image}")
            
            logger.info(f"Found {len(image_paths)} images to process")
            
            # Make predictions
            results = predictor.predict_batch(
                images=image_paths,
                format='file',
                top_k=args.top_k,
                batch_size=args.batch_size,
                return_probabilities=args.show_probabilities
            )
            
            # Add image paths to results
            for i, result in enumerate(results['predictions']):
                result['image_path'] = image_paths[result['image_index']]
            
        else:
            # Process single image
            results = predictor.predict(
                image=args.image,
                format='file',
                top_k=args.top_k,
                return_probabilities=args.show_probabilities
            )
            results['image_path'] = args.image
        
        # Print results
        if 'predictions' in results and results['predictions']:
            if isinstance(results['predictions'], list) and 'image_path' in results['predictions'][0]:
                # Batch results
                for i, result in enumerate(results['predictions']):
                    print(f"\nImage: {result['image_path']}")
                    print("-" * 50)
                    for pred in result['predictions']:
                        print(f"{pred['rank']}. {pred['class_name']}: {pred['probability']:.4f}")
                    print()
            else:
                # Single image results
                print(f"\nImage: {results.get('image_path', 'N/A')}")
                print("-" * 50)
                for pred in results['predictions']:
                    print(f"{pred['rank']}. {pred['class_name']}: {pred['probability']:.4f}")
                print()
        
        # Save results to file if specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Predictions saved to {args.output}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    main()
