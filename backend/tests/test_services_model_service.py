"""
Tests for the ModelService class.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from PIL import Image
import io

# Import the service to test
from services.model_service import ModelService

# Test configuration
TEST_MODEL_PATH = "test_model.h5"
TEST_CLASS_NAMES = ["healthy", "diseased", "pest_infested"]
TEST_IMAGE_SIZE = (224, 224)

@pytest.fixture
def mock_model():
    """Create a mock TensorFlow model."""
    mock_model = MagicMock()
    # Mock the predict method to return dummy predictions
    mock_model.predict.return_value = np.array([[0.1, 0.7, 0.2]])  # Mock prediction probabilities
    return mock_model

@pytest.fixture
def model_service(mock_model):
    """Create a ModelService instance with a mock model."""
    with patch('tensorflow.keras.models.load_model', return_value=mock_model):
        service = ModelService(
            model_path=TEST_MODEL_PATH,
            class_names=TEST_CLASS_NAMES,
            img_size=TEST_IMAGE_SIZE
        )
        return service

@pytest.fixture
def test_image():
    """Create a test image in memory."""
    # Create a simple RGB image
    img = Image.new('RGB', TEST_IMAGE_SIZE, color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr.getvalue()

def test_singleton_pattern():
    """Test that ModelService follows the singleton pattern."""
    with patch('tensorflow.keras.models.load_model'):
        instance1 = ModelService.get_instance()
        instance2 = ModelService.get_instance()
        assert instance1 is instance2

@pytest.mark.asyncio
async def test_load_model(model_service, mock_model):
    """Test model loading functionality."""
    # The model should be loaded in the fixture
    assert model_service.model is not None
    assert model_service.class_names == TEST_CLASS_NAMES
    assert model_service.img_size == TEST_IMAGE_SIZE

@pytest.mark.asyncio
async def test_preprocess_image(model_service, test_image):
    """Test image preprocessing."""
    # Preprocess the test image
    processed_img = await model_service._preprocess_image(test_image)
    
    # Check the output shape and type
    assert isinstance(processed_img, np.ndarray)
    assert processed_img.shape == (1, *TEST_IMAGE_SIZE, 3)  # Batch of 1 image
    assert processed_img.dtype == np.float32
    # Check if values are normalized to [0,1]
    assert processed_img.max() <= 1.0
    assert processed_img.min() >= 0.0

@pytest.mark.asyncio
async def test_predict(model_service, test_image, mock_model):
    """Test prediction functionality."""
    # Make a prediction
    result = await model_service.predict(test_image)
    
    # Check the result structure
    assert isinstance(result, dict)
    assert 'class_name' in result
    assert 'confidence' in result
    assert 'all_predictions' in result
    assert 'timestamp' in result
    
    # Check the prediction values
    assert result['class_name'] in TEST_CLASS_NAMES
    assert 0 <= result['confidence'] <= 1.0
    assert len(result['all_predictions']) == len(TEST_CLASS_NAMES)
    
    # Check that the mock model's predict method was called
    mock_model.predict.assert_called_once()

@pytest.mark.asyncio
async def test_predict_with_invalid_image(model_service):
    """Test prediction with invalid image data."""
    # Test with None
    with pytest.raises(ValueError, match="Image data cannot be None"):
        await model_service.predict(None)
    
    # Test with empty bytes
    with pytest.raises(ValueError, match="Image data is empty"):
        await model_servi
