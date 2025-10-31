"""
Tests for the Prediction model and related functionality.
"""
import pytest
from bson import ObjectId
from datetime import datetime, timedelta

# Import test utilities
from tests.test_utils import assert_error_response

@pytest.mark.asyncio
async def test_create_prediction(db, test_user):
    """Test creating a new prediction."""
    from models.prediction_model import PredictionDB, PredictionCreate, PredictionStatus
    
    # Create test prediction data
    prediction_data = {
        "user_id": ObjectId(test_user["id"]),
        "image_path": "test/path/to/image.jpg",
        "status": PredictionStatus.PENDING,
        "model_version": "1.0.0",
        "metadata": {"test": True}
    }
    
    # Create prediction
    prediction = await PredictionDB.create(db, PredictionCreate(**prediction_data))
    
    # Verify prediction was created
    assert prediction is not None
    assert str(prediction.user_id) == test_user["id"]
    assert prediction.image_path == prediction_data["image_path"]
    assert prediction.status == prediction_data["status"]
    assert prediction.model_version == prediction_data["model_version"]
    assert prediction.metadata == prediction_data["metadata"]
    assert hasattr(prediction, "created_at")
    assert hasattr(prediction, "updated_at")
    assert prediction.created_at <= datetime.utcnow()
    assert prediction.updated_at <= datetime.utcnow()

@pytest.mark.asyncio
async def test_get_prediction_by_id(db, test_prediction):
    """Test retrieving a prediction by ID."""
    from models.prediction_model import PredictionDB
    
    # Get the test prediction
    prediction = await PredictionDB.get_by_id(db, str(test_prediction["_id"]))
    
    # Verify prediction data
    assert prediction is not None
    assert str(prediction.id) == str(test_prediction["_id"])
    assert str(prediction.user_id) == str(test_prediction["user_id"])
    assert prediction.image_path == test_prediction["image_path"]

@pytest.mark.asyncio
async def test_get_nonexistent_prediction(db):
    """Test retrieving a non-existent prediction."""
    from models.prediction_model import PredictionDB
    
    # Try to get a prediction with a non-existent ID
    non_existent_id = str(ObjectId())
    prediction = await PredictionDB.get_by_id(db, non_existent_id)
    assert prediction is None

@pytest.mark.asyncio
async def test_update_prediction(db, test_prediction):
    """Test updating a prediction's information."""
    from models.prediction_model import PredictionDB, PredictionUpdate, PredictionStatus
    
    # Update prediction data
    update_data = {
        "status": PredictionStatus.COMPLETED,
        "result": {"class": "healthy", "confidence": 0.95},
        "error": None
    }
    
    # Perform update
    updated_prediction = await PredictionDB.update(
        db, 
        prediction_id=str(test_prediction["_id"]),
        prediction_update=PredictionUpdate(**update_data)
    )
    
    # Verify updates
    assert updated_prediction is not None
    assert updated_prediction.status == update_data["status"]
    assert updated_prediction.result == update_data["result"]
    assert updated_prediction.error is None
    assert updated_prediction.updated_at > datetime.utcnow() - timedelta(seconds=5)
    
    # Get the prediction again to verify updates were saved
    prediction = await PredictionDB.get_by_id(db, str(test_prediction["_id"]))
    assert prediction.status == update_data["status"]
    assert prediction.result == update_data["result"]

@pytest.mark.asyncio
async def test_delete_prediction(db, test_prediction):
    """Test deleting a prediction."""
    from models.prediction_model import PredictionDB
    
    # Delete the prediction
    deleted = await PredictionDB.delete(db, str(test_prediction["_id"]))
    assert deleted is True
    
    # Verify prediction no longer exists
    prediction = await PredictionDB.get_by_id(db, str(test_prediction["_id"]))
    assert prediction is None

@pytest.mark.asyncio
async def test_get_predictions_by_user(db, test_user):
    """Test retrieving predictions for a specific user."""
    from models.prediction_model import PredictionDB, PredictionCreate, PredictionStatus
    
    # Create some test predictions
    predictions_data = [
        {
            "user_id": ObjectId(test_user["id"]),
            "image_path": f"test/path/to/image_{i}.jpg",
            "status": PredictionStatus.PENDING,
            "model_version": "1.0.0"
        }
        for i in range(3)
    ]
    
    # Add predictions to database
    for pred_data in predictions_data:
        await PredictionDB.create(db, PredictionCreate(**pred_data))
    
    # Get predictions for the test user
    user_predictions = await PredictionDB.get_by_user(
        db, 
        user_id=test_user["id"]
    )
    
    # Verify we got the correct number of predictions
    assert len(user_predictions) >= 3  # Could be more if other tests added predictions
    
    # Verify all predictions belong to the test user
    for pred in user_predictions:
        assert str(pred.user_id) == test_user["id"]

@pytest.mark.asyncio
async def test_list_predictions(db, test_prediction):
    """Test listing predictions with pagination."""
    from models.prediction_model import PredictionDB
    
    # Get all predictions
    predictions = await PredictionDB.list_predictions(db)
    
    # Should find at least the test prediction
    assert len(predictions) >= 1
    
    # Test pagination
    predictions_page1 = await PredictionDB.list_predictions(db, skip=0, limit=1)
    assert len(predictions_page1) == 1
    
    predictions_page2 = await PredictionDB.list_predictions(db, skip=1, limit=1)
    if len(predictions) > 1:
        assert len(predictions_page2) == 1
        # Verify different pages return different predictions
        assert predictions_page1[0].id != predictions_page2[0].id

@pytest.mark.asyncio
async def test_count_predictions(db, test_prediction):
    """Test counting predictions."""
    from models.prediction_model import PredictionDB, PredictionStatus
    
    # Count all predictions
    count = await PredictionDB.count(db)
    assert count >= 1  # At least the test prediction
    
    # Count by status
    pending_count = await PredictionDB.count(
        db, 
        status=PredictionStatus.PENDING
    )
    assert pending_count >= 1  # Test prediction should be pending
    
    # Count by user
    user_count = await PredictionDB.count(
        db, 
        user_id=str(test_prediction["user_id"])
    )
    assert user_count >= 1  # At least the test prediction

@pytest.mark.asyncio
async def test_get_user_prediction_count(db, test_user):
    """Test getting prediction count for a user by time period."""
    from models.prediction_model import PredictionDB, PredictionCreate
    
    # Create a new prediction for the test user
    prediction_data = {
        "user_id": ObjectId(test_user["id"]),
        "image_path": "test/path/to/recent_image.jpg",
        "model_version": "1.0.0"
    }
    await PredictionDB.create(db, PredictionCreate(**prediction_data))
    
    # Get count for today
    daily_count = await PredictionDB.get_user_prediction_count(
        db,
        user_id=test_user["id"],
        time_period="day"
    )
    assert daily_count >= 1  # At least the prediction we just created
    
    # Get weekly count (should also include the new prediction)
    weekly_count = await PredictionDB.get_user_prediction_count(
        db,
        user_id=test_user["id"],
        time_period="week"
    )
    assert weekly_count >= daily_count
    
    # Get monthly count (should also include the new prediction)
    monthly_count = await PredictionDB.get_user_prediction_count(
        db,
        user_id=test_user["id"],
        time_period="month"
    )
    assert monthly_count >= weekly_count
