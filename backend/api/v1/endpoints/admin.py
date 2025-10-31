"""
Admin API endpoints for the Crop Health Prediction System.
Requires admin privileges to access these endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any
from bson import ObjectId
from datetime import datetime, timedelta
import json

from ....core.security import get_current_active_superuser
from ....models.schemas import UserInDB, UserUpdate, PredictionResponse, EvaluationResult
from ....models.models import User, Prediction, Evaluation, APIKey
from ....core.database import get_db
from ....core.config import settings

router = APIRouter()

@router.get("/users/", response_model=List[UserInDB], tags=["Admin"]
async def list_users(
    skip: int = 0,
    limit: int = 100,
    current_user: UserInDB = Depends(get_current_active_superuser),
    db=Depends(get_db)
):
    """List all users (admin only)."""
    users = await User.find_all(skip=skip, limit=limit)
    return users

@router.get("/users/{user_id}", response_model=UserInDB, tags=["Admin"])
async def get_user(
    user_id: str,
    current_user: UserInDB = Depends(get_current_active_superuser),
    db=Depends(get_db)
):
    """Get a specific user by ID (admin only)."""
    user = await User.find_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.put("/users/{user_id}", response_model=UserInDB, tags=["Admin"])
async def update_user(
    user_id: str,
    user_in: UserUpdate,
    current_user: UserInDB = Depends(get_current_active_superuser),
    db=Depends(get_db)
):
    """Update a user (admin only)."""
    user = await User.find_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    updated_user = await User.update(user_id, user_in.dict(exclude_unset=True))
    return updated_user

@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Admin"])
async def delete_user(
    user_id: str,
    current_user: UserInDB = Depends(get_current_active_superuser),
    db=Depends(get_db)
):
    """Delete a user (admin only)."""
    user = await User.find_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Prevent deleting yourself
    if str(user.id) == str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own user account"
        )
    
    await User.delete(user_id)
    return {"status": "success", "message": "User deleted successfully"}

@router.get("/predictions/", response_model=List[PredictionResponse], tags=["Admin"])
async def list_predictions(
    skip: int = 0,
    limit: int = 100,
    user_id: str = None,
    current_user: UserInDB = Depends(get_current_active_superuser),
    db=Depends(get_db)
):
    """List all predictions (admin only)."""
    query = {}
    if user_id:
        query["user_id"] = ObjectId(user_id)
    
    predictions = await Prediction.find(query, skip=skip, limit=limit)
    return predictions

@router.get("/evaluations/", response_model=List[EvaluationResult], tags=["Admin"])
async def list_evaluations(
    skip: int = 0,
    limit: int = 100,
    current_user: UserInDB = Depends(get_current_active_superuser),
    db=Depends(get_db)
):
    """List all model evaluations (admin only)."""
    evaluations = await Evaluation.find({}, skip=skip, limit=limit)
    return evaluations

@router.get("/metrics/", tags=["Admin"])
async def get_system_metrics(
    current_user: UserInDB = Depends(get_current_active_superuser),
    db=Depends(get_db)
):
    """Get system metrics (admin only)."""
    # Get user statistics
    total_users = await User.count()
    active_users = await User.count({"is_active": True})
    
    # Get prediction statistics
    total_predictions = await Prediction.count()
    predictions_last_24h = await Prediction.count({
        "created_at": {"$gte": datetime.utcnow() - timedelta(days=1)}
    })
    
    # Get evaluation statistics
    total_evaluations = await Evaluation.count()
    
    return {
        "users": {
            "total": total_users,
            "active": active_users,
            "inactive": total_users - active_users
        },
        "predictions": {
            "total": total_predictions,
            "last_24h": predictions_last_24h
        },
        "evaluations": {
            "total": total_evaluations
        }
    }
