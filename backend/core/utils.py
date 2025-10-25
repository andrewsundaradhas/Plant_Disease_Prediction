""
Utility functions for the Crop Health Prediction System.
"""
import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from fastapi import UploadFile
import aiofiles

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def save_upload_file(upload_file: UploadFile, destination: Path) -> str:
    """
    Save an uploaded file to the specified destination.
    
    Args:
        upload_file: The uploaded file object
        destination: Path where to save the file
        
    Returns:
        str: Path to the saved file
    """
    try:
        # Ensure the directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the file asynchronously
        async with aiofiles.open(destination, 'wb') as buffer:
            content = await upload_file.read()
            await buffer.write(content)
            
        logger.info(f"File saved successfully to {destination}")
        return str(destination)
        
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise

def get_file_extension(filename: str) -> str:
    """Get the file extension from a filename."""
    return os.path.splitext(filename)[1].lower()

def is_valid_image_extension(extension: str) -> bool:
    """Check if the file extension is a valid image format."""
    return extension.lower() in {'.jpg', '.jpeg', '.png'}

def format_prediction_results(
    predictions: np.ndarray, 
    class_names: List[str],
    confidence_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Format model predictions into a more readable format.
    
    Args:
        predictions: Numpy array of predictions from the model
        class_names: List of class names corresponding to prediction indices
        confidence_threshold: Minimum confidence score to include in results
        
    Returns:
        List of formatted prediction results
    """
    results = []
    
    for pred in predictions:
        # Get top prediction
        top_idx = np.argmax(pred)
        confidence = float(pred[top_idx])
        
        if confidence >= confidence_threshold:
            results.append({
                'class_name': class_names[top_idx],
                'confidence': confidence,
                'all_predictions': {
                    class_name: float(conf) 
                    for class_name, conf in zip(class_names, pred)
                }
            })
    
    return results

def load_json_file(file_path: Union[str, Path]) -> Any:
    """Load JSON data from a file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {str(e)}")
        raise

def save_json_file(data: Any, file_path: Union[str, Path]) -> None:
    """Save data to a JSON file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Data saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {str(e)}")
        raise

def validate_image_file(file: UploadFile) -> bool:
    """
    Validate that the uploaded file is a valid image.
    
    Args:
        file: The uploaded file to validate
        
    Returns:
        bool: True if the file is valid, False otherwise
    """
    # Check file extension
    file_extension = get_file_extension(file.filename)
    if not is_valid_image_extension(file_extension):
        return False
    
    # Check content type
    content_type = file.content_type
    if not content_type or not content_type.startswith('image/'):
        return False
    
    return True

def create_directory(directory: Union[str, Path]) -> None:
    """Create a directory if it doesn't exist."""
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {str(e)}")
        raise
