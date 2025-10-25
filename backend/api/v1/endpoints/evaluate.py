"""
Model evaluation endpoints for the Crop Health Prediction API.

This module provides endpoints for evaluating model performance.
"""
import os
import uuid
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional

from fastapi import (
    APIRouter, 
    Depends, 
    HTTPException, 
    status,
    BackgroundTasks,
    Query
)
from fastapi.responses import FileResponse

from ....core.security import get_current_user, get_current_active_superuser
from ....models.schemas import (
    EvaluationRequest,
    EvaluationResponse,
    ErrorResponse,
    BaseResponse
)
from ....models.models import Evaluation, User
from ....core.config import settings
from ....db.mongodb import db_manager
from ....ml.evaluate import evaluate_model as ml_evaluate_model

router = APIRouter()
logger = logging.getLogger(__name__)

# Global state for evaluation
_evaluation_lock = asyncio.Lock()
_evaluation_status = {
    "status": "idle",  # 'idle', 'running', 'completed', 'failed'
    "progress": 0.0,
    "start_time": None,
    "end_time": None,
    "error": None,
    "result": None
}

@router.post("/evaluate", response_model=EvaluationResponse, status_code=status.HTTP_202_ACCEPTED)
async def evaluate_model(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_superuser)
) -> Dict[str, Any]:
    """
    Start a model evaluation job.
    
    - **test_data_dir**: Path to the directory containing test images
    - **batch_size**: Batch size for evaluation (default: 32)
    - **save_to_db**: Whether to save the evaluation results to the database (default: True)
    
    Starts an asynchronous evaluation job and returns immediately with a job ID.
    """
    global _evaluation_status
    
    # Check if an evaluation is already running
    async with _evaluation_lock:
        if _evaluation_status["status"] == "running":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An evaluation is already in progress"
            )
        
        # Reset status
        _evaluation_status = {
            "status": "running",
            "progress": 0.0,
            "start_time": datetime.utcnow(),
            "end_time": None,
            "error": None,
            "result": None
        }
    
    # Start the evaluation in the background
    background_tasks.add_task(
        run_evaluation,
        test_data_dir=request.test_data_dir,
        batch_size=request.batch_size,
        save_to_db=request.save_to_db,
        user_id=str(current_user.id)
    )
    
    return {
        "success": True,
        "data": {
            "status": "started",
            "message": "Evaluation job started successfully",
            "start_time": _evaluation_status["start_time"].isoformat()
        }
    }

@router.get("/evaluate/status", response_model=Dict[str, Any])
async def get_evaluation_status() -> Dict[str, Any]:
    """
    Get the status of the current or most recent evaluation job.
    
    Returns the current status, progress, and any results or errors.
    """
    global _evaluation_status
    
    status_data = {
        "status": _evaluation_status["status"],
        "progress": _evaluation_status["progress"],
        "start_time": _evaluation_status["start_time"].isoformat() if _evaluation_status["start_time"] else None,
        "end_time": _evaluation_status["end_time"].isoformat() if _evaluation_status["end_time"] else None,
        "duration_seconds": (
            (_evaluation_status["end_time"] or datetime.utcnow()) - _evaluation_status["start_time"].total_seconds()
            if _evaluation_status["start_time"] else None
        ),
        "error": _evaluation_status.get("error"),
        "has_result": _evaluation_status.get("result") is not None
    }
    
    # Include result summary if available
    if _evaluation_status.get("result"):
        result = _evaluation_status["result"]
        status_data["result_summary"] = {
            "accuracy": result.get("accuracy"),
            "f1_score": result.get("f1_score"),
            "precision": result.get("precision"),
            "recall": result.get("recall"),
            "num_samples": result.get("num_samples")
        }
    
    return {
        "success": True,
        "data": status_data
    }

@router.get("/evaluate/results", response_model=Dict[str, Any])
async def get_evaluation_results() -> Dict[str, Any]:
    """
    Get the detailed results of the most recent evaluation.
    
    Returns the full evaluation metrics, including per-class statistics.
    """
    global _evaluation_status
    
    if not _evaluation_status.get("result"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No evaluation results available"
        )
    
    return {
        "success": True,
        "data": _evaluation_status["result"]
    }

@router.get("/evaluate/history", response_model=Dict[str, Any])
async def get_evaluation_history(
    skip: int = 0,
    limit: int = 10,
    current_user: User = Depends(get_current_active_superuser)
) -> Dict[str, Any]:
    """
    Get a history of past evaluations.
    
    - **skip**: Number of records to skip (for pagination)
    - **limit**: Maximum number of records to return (default: 10, max: 100)
    
    Returns a paginated list of past evaluations.
    """
    # Normalize parameters
    limit = max(1, min(100, limit))  # Ensure limit is between 1 and 100
    skip = max(0, skip)  # Ensure skip is not negative
    
    db = await db_manager.get_database()
    
    # Get total count
    total = await db["evaluations"].count_documents({})
    
    # Get paginated results
    cursor = db["evaluations"].find().sort("created_at", -1).skip(skip).limit(limit)
    
    evaluations = []
    async for eval_doc in cursor:
        # Extract summary metrics
        metrics = eval_doc.get("metrics", {})
        eval_summary = {
            "id": str(eval_doc["_id"]),
            "model_version": eval_doc.get("model_version", "unknown"),
            "created_at": eval_doc["created_at"].isoformat(),
            "created_by": str(eval_doc.get("created_by")),
            "metrics": {
                "accuracy": metrics.get("accuracy"),
                "f1_score": metrics.get("f1_score"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "num_samples": metrics.get("num_samples")
            }
        }
        evaluations.append(eval_summary)
    
    return {
        "success": True,
        "data": evaluations,
        "total": total,
        "skip": skip,
        "limit": limit,
        "has_more": (skip + len(evaluations)) < total
    }

@router.get("/evaluate/results/{evaluation_id}", response_model=Dict[str, Any])
async def get_evaluation_result(
    evaluation_id: str,
    current_user: User = Depends(get_current_active_superuser)
) -> Dict[str, Any]:
    """
    Get detailed results for a specific evaluation.
    
    - **evaluation_id**: The ID of the evaluation to retrieve
    
    Returns the full evaluation metrics for the specified evaluation.
    """
    from bson import ObjectId
    
    try:
        db = await db_manager.get_database()
        eval_doc = await db["evaluations"].find_one({"_id": ObjectId(evaluation_id)})
        
        if not eval_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Evaluation not found"
            )
        
        # Convert ObjectId to string for JSON serialization
        eval_doc["id"] = str(eval_doc.pop("_id"))
        eval_doc["created_by"] = str(eval_doc.get("created_by"))
        
        return {
            "success": True,
            "data": eval_doc
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting evaluation result: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving evaluation result"
        )

@router.get("/evaluate/export/{evaluation_id}", response_class=FileResponse)
async def export_evaluation_results(
    evaluation_id: str,
    format: str = "json",
    current_user: User = Depends(get_current_active_superuser)
):
    """
    Export evaluation results in the specified format.
    
    - **evaluation_id**: The ID of the evaluation to export
    - **format**: Export format (json, csv) - defaults to json
    
    Returns the evaluation results in the requested format.
    """
    from bson import ObjectId
    import csv
    import tempfile
    import os
    
    try:
        db = await db_manager.get_database()
        eval_doc = await db["evaluations"].find_one({"_id": ObjectId(evaluation_id)})
        
        if not eval_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Evaluation not found"
            )
        
        # Prepare data for export
        metrics = eval_doc.get("metrics", {})
        class_metrics = metrics.get("class_metrics", {})
        
        if format.lower() == "csv":
            # Create a temporary file for CSV export
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow(["Metric", "Value"])
                
                # Write overall metrics
                writer.writerow(["Overall Metrics", ""])
                writer.writerow(["Accuracy", metrics.get("accuracy", "")])
                writer.writerow(["Precision", metrics.get("precision", "")])
                writer.writerow(["Recall", metrics.get("recall", "")])
                writer.writerow(["F1 Score", metrics.get("f1_score", "")])
                writer.writerow(["Number of Samples", metrics.get("num_samples", "")])
                
                # Write per-class metrics
                writer.writerow(["", ""])
                writer.writerow(["Per-Class Metrics", ""])
                writer.writerow(["Class", "Precision", "Recall", "F1 Score", "Support"])
                
                for class_name, class_metric in class_metrics.items():
                    writer.writerow([
                        class_name,
                        class_metric.get("precision", ""),
                        class_metric.get("recall", ""),
                        class_metric.get("f1_score", ""),
                        class_metric.get("support", "")
                    ])
                
                temp_path = f.name
            
            # Return the file for download
            return FileResponse(
                temp_path,
                media_type="text/csv",
                filename=f"evaluation_{evaluation_id}.csv",
                background=BackgroundTask(lambda: os.unlink(temp_path) if os.path.exists(temp_path) else None)
            )
            
        else:  # Default to JSON
            # Create a temporary file for JSON export
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
                json.dump(metrics, f, indent=2)
                temp_path = f.name
            
            # Return the file for download
            return FileResponse(
                temp_path,
                media_type="application/json",
                filename=f"evaluation_{evaluation_id}.json",
                background=BackgroundTask(lambda: os.unlink(temp_path) if os.path.exists(temp_path) else None)
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting evaluation results: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error exporting evaluation results"
        )

async def run_evaluation(
    test_data_dir: str,
    batch_size: int = 32,
    save_to_db: bool = True,
    user_id: Optional[str] = None
) -> None:
    """
    Run model evaluation and update the global status.
    
    This function is meant to be run in a background task.
    """
    global _evaluation_status
    
    try:
        # Check if test data directory exists
        if not os.path.isdir(test_data_dir):
            raise ValueError(f"Test data directory not found: {test_data_dir}")
        
        # Initialize the model
        from ....ml.model import load_model
        from ....ml.data_loader import PlantVillageDataLoader
        
        model_path = os.path.join(settings.MODEL_DIR, settings.MODEL_NAME)
        if not os.path.exists(model_path):
            raise ValueError(f"Model not found at {model_path}")
        
        # Load the model
        model = load_model(model_path)
        
        # Update status
        _evaluation_status["status"] = "running"
        _evaluation_status["progress"] = 0.1
        
        # Load test data
        data_loader = PlantVillageDataLoader(
            data_dir=test_data_dir,
            batch_size=batch_size,
            img_size=(224, 224)  # Default image size
        )
        
        # Get test data generator
        _, _, test_gen = data_loader.get_data_generators()
        
        # Update status
        _evaluation_status["progress"] = 0.2
        
        # Run evaluation
        def progress_callback(progress: float, message: str = "") -> None:
            # Update progress (scale to 0.2-0.9 range to account for setup/teardown)
            _evaluation_status["progress"] = 0.2 + (0.7 * progress)
            logger.info(f"Evaluation progress: {_evaluation_status['progress']*100:.1f}% - {message}")
        
        # Run evaluation
        results = ml_evaluate_model(
            model=model,
            test_gen=test_gen,
            class_names=data_loader.get_class_names(),
            progress_callback=progress_callback
        )
        
        # Prepare the result
        result = {
            "accuracy": results.get("accuracy"),
            "precision": results.get("precision"),
            "recall": results.get("recall"),
            "f1_score": results.get("f1_score"),
            "num_samples": results.get("num_samples"),
            "class_metrics": results.get("class_metrics", {}),
            "confusion_matrix": results.get("confusion_matrix"),
            "timestamp": datetime.utcnow().isoformat(),
            "test_data_dir": test_data_dir,
            "batch_size": batch_size
        }
        
        # Save to database if requested
        if save_to_db:
            try:
                db = await db_manager.get_database()
                await Evaluation.create(
                    db=db,
                    metrics=result,
                    test_data_path=test_data_dir,
                    model_version=settings.MODEL_VERSION,
                    user_id=user_id
                )
            except Exception as e:
                logger.error(f"Error saving evaluation to database: {str(e)}", exc_info=True)
        
        # Update status
        _evaluation_status["status"] = "completed"
        _evaluation_status["progress"] = 1.0
        _evaluation_status["end_time"] = datetime.utcnow()
        _evaluation_status["result"] = result
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        _evaluation_status["status"] = "failed"
        _evaluation_status["error"] = str(e)
        _evaluation_status["end_time"] = datetime.utcnow()
    
    finally:
        # Ensure we always have an end time
        if not _evaluation_status.get("end_time"):
            _evaluation_status["end_time"] = datetime.utcnow()
        
        logger.info("Evaluation completed")
