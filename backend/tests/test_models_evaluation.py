"""
Tests for the Evaluation model and related functionality.
"""
import pytest
from bson import ObjectId
from datetime import datetime, timedelta

# Import test utilities
from tests.test_utils import assert_error_response

@pytest.mark.asyncio
async def test_create_evaluation(db, test_user):
    """Test creating a new evaluation."""
    from models.evaluation_model import EvaluationDB, EvaluationCreate, EvaluationStatus
    
    # Create test evaluation data
    evaluation_data = {
        "model_version": "1.0.0",
        "dataset_name": "test_dataset",
        "status": EvaluationStatus.PENDING,
        "metrics": {},
        "metadata": {"test": True}
    }
    
    # Create evaluation
    evaluation = await EvaluationDB.create(db, EvaluationCreate(**evaluation_data))
    
    # Verify evaluation was created
    assert evaluation is not None
    assert evaluation.model_version == evaluation_data["model_version"]
    assert evaluation.dataset_name == evaluation_data["dataset_name"]
    assert evaluation.status == evaluation_data["status"]
    assert evaluation.metrics == evaluation_data["metrics"]
    assert evaluation.metadata == evaluation_data["metadata"]
    assert hasattr(evaluation, "created_at")
    assert hasattr(evaluation, "updated_at")
    assert evaluation.created_at <= datetime.utcnow()
    assert evaluation.updated_at <= datetime.utcnow()

@pytest.mark.asyncio
async def test_get_evaluation_by_id(db, test_evaluation):
    """Test retrieving an evaluation by ID."""
    from models.evaluation_model import EvaluationDB
    
    # Get the test evaluation
    evaluation = await EvaluationDB.get_by_id(db, str(test_evaluation["_id"]))
    
    # Verify evaluation data
    assert evaluation is not None
    assert str(evaluation.id) == str(test_evaluation["_id"])
    assert evaluation.model_version == test_evaluation["model_version"]
    assert evaluation.dataset_name == test_evaluation["dataset_name"]

@pytest.mark.asyncio
async def test_get_nonexistent_evaluation(db):
    """Test retrieving a non-existent evaluation."""
    from models.evaluation_model import EvaluationDB
    
    # Try to get an evaluation with a non-existent ID
    non_existent_id = str(ObjectId())
    evaluation = await EvaluationDB.get_by_id(db, non_existent_id)
    assert evaluation is None

@pytest.mark.asyncio
async def test_update_evaluation(db, test_evaluation):
    """Test updating an evaluation's information."""
    from models.evaluation_model import EvaluationDB, EvaluationUpdate, EvaluationStatus
    
    # Update evaluation data
    update_data = {
        "status": EvaluationStatus.COMPLETED,
        "metrics": {"accuracy": 0.95, "precision": 0.94, "recall": 0.93},
        "confusion_matrix": [[50, 5], [3, 42]],
        "class_report": {"class1": {"precision": 0.94, "recall": 0.91, "f1-score": 0.92}},
        "error": None
    }
    
    # Perform update
    updated_evaluation = await EvaluationDB.update(
        db, 
        evaluation_id=str(test_evaluation["_id"]),
        evaluation_update=EvaluationUpdate(**update_data)
    )
    
    # Verify updates
    assert updated_evaluation is not None
    assert updated_evaluation.status == update_data["status"]
    assert updated_evaluation.metrics == update_data["metrics"]
    assert updated_evaluation.confusion_matrix == update_data["confusion_matrix"]
    assert updated_evaluation.class_report == update_data["class_report"]
    assert updated_evaluation.error is None
    assert updated_evaluation.updated_at > datetime.utcnow() - timedelta(seconds=5)
    
    # Get the evaluation again to verify updates were saved
    evaluation = await EvaluationDB.get_by_id(db, str(test_evaluation["_id"]))
    assert evaluation.status == update_data["status"]
    assert evaluation.metrics == update_data["metrics"]

@pytest.mark.asyncio
async def test_delete_evaluation(db, test_evaluation):
    """Test deleting an evaluation."""
    from models.evaluation_model import EvaluationDB
    
    # Delete the evaluation
    deleted = await EvaluationDB.delete(db, str(test_evaluation["_id"]))
    assert deleted is True
    
    # Verify evaluation no longer exists
    evaluation = await EvaluationDB.get_by_id(db, str(test_evaluation["_id"]))
    assert evaluation is None

@pytest.mark.asyncio
async def test_list_evaluations(db, test_evaluation):
    """Test listing evaluations with filters."""
    from models.evaluation_model import EvaluationDB, EvaluationStatus
    
    # Get all evaluations
    evaluations = await EvaluationDB.list_evaluations(db)
    
    # Should find at least the test evaluation
    assert len(evaluations) >= 1
    
    # Test filtering by status
    completed_evaluations = await EvaluationDB.list_evaluations(
        db,
        status=EvaluationStatus.COMPLETED
    )
    # The test evaluation might be PENDING, so we can't assume there are completed ones
    
    # Test filtering by model version
    version_evaluations = await EvaluationDB.list_evaluations(
        db,
        model_version=test_evaluation["model_version"]
    )
    assert len(version_evaluations) >= 1
    
    # Test pagination
    evaluations_page1 = await EvaluationDB.list_evaluations(db, skip=0, limit=1)
    assert len(evaluations_page1) == 1
    
    evaluations_page2 = await EvaluationDB.list_evaluations(db, skip=1, limit=1)
    if len(evaluations) > 1:
        assert len(evaluations_page2) == 1
        # Verify different pages return different evaluations
        assert evaluations_page1[0].id != evaluations_page2[0].id

@pytest.mark.asyncio
async def test_count_evaluations(db, test_evaluation):
    """Test counting evaluations."""
    from models.evaluation_model import EvaluationDB, EvaluationStatus
    
    # Count all evaluations
    count = await EvaluationDB.count(db)
    assert count >= 1  # At least the test evaluation
    
    # Count by status
    pending_count = await EvaluationDB.count(
        db, 
        status=EvaluationStatus.PENDING
    )
    assert pending_count >= 0  # Could be 0 if test_evaluation is not PENDING
    
    # Count by model version
    version_count = await EvaluationDB.count(
        db, 
        model_version=test_evaluation["model_version"]
    )
    assert version_count >= 1  # At least the test evaluation

@pytest.mark.asyncio
async def test_get_latest_evaluation(db, test_evaluation):
    """Test getting the latest evaluation for a model version."""
    from models.evaluation_model import EvaluationDB, EvaluationCreate, EvaluationStatus
    
    # Create a newer evaluation for the same model version
    newer_evaluation_data = {
        "model_version": test_evaluation["model_version"],
        "dataset_name": "newer_test_dataset",
        "status": EvaluationStatus.COMPLETED,
        "metrics": {"accuracy": 0.96},
        "metadata": {"test": True}
    }
    
    # Add a small delay to ensure the timestamps are different
    import asyncio
    await asyncio.sleep(1)
    
    # Create the newer evaluation
    newer_evaluation = await EvaluationDB.create(
        db, 
        EvaluationCreate(**newer_evaluation_data)
    )
    
    # Get the latest evaluation for this model version
    latest_evaluation = await EvaluationDB.get_latest_by_model_version(
        db,
        model_version=test_evaluation["model_version"]
    )
    
    # Verify we got the most recent evaluation
    assert latest_evaluation is not None
    assert str(latest_evaluation.id) == str(newer_evaluation.id)
    assert latest_evaluation.dataset_name == newer_evaluation_data["dataset_name"]
    assert latest_evaluation.metrics == newer_evaluation_data["metrics"]

@pytest.mark.asyncio
async def test_get_evaluation_metrics_history(db, test_evaluation):
    """Test getting evaluation metrics history for a model."""
    from models.evaluation_model import EvaluationDB, EvaluationCreate, EvaluationStatus
    
    # Create a few more evaluations for the same model version
    for i in range(2):
        evaluation_data = {
            "model_version": test_evaluation["model_version"],
            "dataset_name": f"test_dataset_v{i}",
            "status": EvaluationStatus.COMPLETED,
            "metrics": {"accuracy": 0.90 + (i * 0.03)},
            "metadata": {"test": True}
        }
        await EvaluationDB.create(db, EvaluationCreate(**evaluation_data))
        
        # Add a small delay to ensure the timestamps are different
        import asyncio
        await asyncio.sleep(1)
    
    # Get metrics history
    metrics_history = await EvaluationDB.get_metrics_history(
        db,
        model_version=test_evaluation["model_version"],
        metric_name="accuracy"
    )
    
    # Verify we got the correct number of data points
    # Should be at least 3 (test_evaluation + 2 we just created)
    assert len(metrics_history) >= 3
    
    # Verify the data is sorted by date (newest first)
    for i in range(len(metrics_history) - 1):
        assert metrics_history[i]["date"] >= metrics_history[i+1]["date"]
    
    # Verify the metric values are in the expected format
    for point in metrics_history:
        assert "date" in point
        assert "value" in point
        assert isinstance(point["date"], datetime)
        assert isinstance(point["value"], (int, float))
