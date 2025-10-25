"""
Prediction endpoints for the Crop Health Prediction API.

This module provides endpoints for making predictions on plant images.
"""
import os
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import (
    APIRouter, 
    Depends, 
    File, 
    UploadFile, 
    HTTPException, 
    status,
    BackgroundTasks
)
from fastapi.responses import FileResponse, JSONResponse
import aiofiles

from ....core.security import get_current_user, get_api_key_user
from ....models.schemas import (
    PredictionResponse,
    PredictionListResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse
)
from ....models.models import Prediction, User
from ....core.config import settings
from ....db.mongodb import db_manager
from ....ml.predict import PlantDiseasePredictor

router = APIRouter()
logger = logging.getLogger(__name__)

# Global predictor instance (lazy-loaded)
_predictor = None

async def get_predictor() -> PlantDiseasePredictor:
    """Get the global predictor instance, initializing it if necessary."""
    global _predictor
    
    if _predictor is None:
        try:
            model_path = os.path.join(settings.MODEL_DIR, settings.MODEL_NAME)
            _predictor = PlantDiseasePredictor(
                model_path=model_path,
                class_names_path=settings.CLASS_MAPPING_PATH
            )
            logger.info("Initialized PlantDiseasePredictor")
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not available"
            )
    
    return _predictor

@router.post("/predict", response_model=PredictionResponse, responses={
    400: {"model": ErrorResponse, "description": "Invalid image file"},
    413: {"model": ErrorResponse, "description": "File too large"},
    503: {"model": ErrorResponse, "description": "Model not available"}
})
async def predict(
    file: UploadFile = File(..., description="Image file to analyze"),
    current_user: User = Depends(get_current_user),
    predictor: PlantDiseasePredictor = Depends(get_predictor)
) -> Dict[str, Any]:
    """
    Make a prediction on a single uploaded image.
    
    - **file**: Image file (JPEG, PNG, etc.)
    
    Returns the predicted class and confidence scores.
    """
    # Validate file type
    file_ext = os.path.splitext(file.filename or "")[1].lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not allowed. Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    
    # Create upload directory if it doesn't exist
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    # Generate a unique filename
    file_id = str(uuid.uuid4())
    filename = f"{file_id}{file_ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    
    try:
        # Save the uploaded file
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            
            # Check file size
            if len(content) > settings.MAX_UPLOAD_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE} bytes"
                )
                
            await out_file.write(content)
        
        # Make prediction
        result = await predictor.predict(
            image=content,
            format='bytes',
            top_k=3,
            return_probabilities=True
        )
        
        if not result.get('success', False):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get('error', 'Prediction failed')
            )
        
        # Get the top prediction
        top_pred = result['predictions'][0] if result['predictions'] else None
        
        if not top_pred:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No predictions returned from model"
            )
        
        # Create prediction record
        prediction_data = {
            "user_id": current_user.id,
            "image_path": file_path,
            "predicted_class": top_pred['class_name'],
            "confidence": top_pred['probability'],
            "class_probabilities": result.get('all_probabilities', {}),
            "metadata": {
                "original_filename": file.filename,
                "content_type": file.content_type,
                "file_size": len(content),
            }
        }
        
        # Save to database
        db = await db_manager.get_database()
        prediction = await Prediction.create(db, prediction_data, str(current_user.id))
        
        return {
            "success": True,
            "data": prediction.to_response()
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )
    finally:
        # Clean up the uploaded file in case of errors
        if os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except Exception as e:
                logger.error(f"Error cleaning up file {file_path}: {str(e)}")

@router.post("/predict/batch", response_model=BatchPredictionResponse, responses={
    400: {"model": ErrorResponse, "description": "Invalid request"},
    503: {"model": ErrorResponse, "description": "Model not available"}
})
async def predict_batch(
    request: BatchPredictionRequest,
    current_user: User = Depends(get_current_user),
    predictor: PlantDiseasePredictor = Depends(get_predictor)
) -> Dict[str, Any]:
    """
    Make predictions on multiple images specified by URLs.
    
    - **image_urls**: List of image URLs to analyze
    - **save_to_db**: Whether to save predictions to the database (default: True)
    
    Returns predictions for all images.
    """
    import aiohttp
    from urllib.parse import urlparse
    
    if not request.image_urls:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image URLs provided"
        )
    
    # Limit the number of images per batch
    max_batch_size = 10
    if len(request.image_urls) > max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum batch size is {max_batch_size} images"
        )
    
    results = []
    successful = 0
    failed = 0
    
    async with aiohttp.ClientSession() as session:
        for url in request.image_urls:
            try:
                # Download the image
                async with session.get(str(url)) as response:
                    if response.status != 200:
                        raise ValueError(f"Failed to download image: HTTP {response.status}")
                    
                    content = await response.read()
                    
                    # Make prediction
                    result = await predictor.predict(
                        image=content,
                        format='bytes',
                        top_k=3,
                        return_probabilities=True
                    )
                    
                    if not result.get('success', False):
                        raise ValueError(result.get('error', 'Prediction failed'))
                    
                    # Get the top prediction
                    top_pred = result['predictions'][0] if result['predictions'] else None
                    
                    if not top_pred:
                        raise ValueError("No predictions returned from model")
                    
                    # Create prediction data
                    prediction_data = {
                        "user_id": current_user.id,
                        "image_url": str(url),
                        "predicted_class": top_pred['class_name'],
                        "confidence": top_pred['probability'],
                        "class_probabilities": result.get('all_probabilities', {}),
                        "metadata": {
                            "content_type": response.content_type,
                            "file_size": len(content),
                            "source_url": str(url)
                        }
                    }
                    
                    # Save to database if requested
                    if request.save_to_db:
                        db = await db_manager.get_database()
                        prediction = await Prediction.create(db, prediction_data, str(current_user.id))
                        prediction_data["prediction_id"] = str(prediction.id)
                    
                    results.append({
                        "image_url": str(url),
                        "success": True,
                        "prediction": {
                            "class_name": top_pred['class_name'],
                            "confidence": top_pred['probability'],
                            "all_predictions": [
                                {"class_name": p['class_name'], "probability": p['probability']}
                                for p in result.get('predictions', [])
                            ]
                        },
                        "prediction_id": prediction_data.get("prediction_id")
                    })
                    successful += 1
                    
            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}", exc_info=True)
                results.append({
                    "image_url": str(url),
                    "success": False,
                    "error": str(e)
                })
                failed += 1
    
    return {
        "success": True,
        "predictions": results,
        "total": len(request.image_urls),
        "successful": successful,
        "failed": failed
    }

@router.get("/predictions", response_model=PredictionListResponse)
async def list_predictions(
    skip: int = 0,
    limit: int = 10,
    user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List predictions with pagination.
    
    - **skip**: Number of records to skip (for pagination)
    - **limit**: Maximum number of records to return (default: 10, max: 100)
    - **user_id**: Filter by user ID (admin only)
    
    Returns a paginated list of predictions.
    """
    from bson import ObjectId
    
    # Normalize parameters
    limit = max(1, min(100, limit))  # Ensure limit is between 1 and 100
    skip = max(0, skip)  # Ensure skip is not negative
    
    # Build query
    query = {}
    
    # Non-admin users can only see their own predictions
    if not current_user.is_superuser:
        query["user_id"] = current_user.id
    # Admin can filter by user_id
    elif user_id:
        try:
            query["user_id"] = ObjectId(user_id)
        except:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid user_id"
            )
    
    db = await db_manager.get_database()
    
    # Get total count
    total = await db["predictions"].count_documents(query)
    
    # Get paginated results
    cursor = db["predictions"].find(query).sort("created_at", -1).skip(skip).limit(limit)
    
    predictions = []
    async for pred in cursor:
        predictions.append({
            "id": str(pred["_id"]),
            "user_id": str(pred.get("user_id")),
            "image_url": pred.get("image_url"),
            "image_path": pred.get("image_path"),
            "predicted_class": pred["predicted_class"],
            "confidence": pred["confidence"],
            "created_at": pred["created_at"].isoformat(),
            "metadata": pred.get("metadata", {})
        })
    
    return {
        "success": True,
        "data": predictions,
        "total": total,
        "skip": skip,
        "limit": limit,
        "has_more": (skip + len(predictions)) < total
    }

@router.get("/predictions/{prediction_id}", response_model=PredictionResponse)
async def get_prediction(
    prediction_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get a specific prediction by ID.
    
    - **prediction_id**: The ID of the prediction to retrieve
    
    Returns the prediction details.
    """
    from bson import ObjectId
    
    try:
        db = await db_manager.get_database()
        prediction = await db["predictions"].find_one({"_id": ObjectId(prediction_id)})
        
        if not prediction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Prediction not found"
            )
        
        # Check permissions (user can only see their own predictions unless admin)
        if not current_user.is_superuser and str(prediction.get("user_id")) != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this prediction"
            )
        
        return {
            "success": True,
            "data": {
                "id": str(prediction["_id"]),
                "user_id": str(prediction.get("user_id")),
                "image_url": prediction.get("image_url"),
                "image_path": prediction.get("image_path"),
                "predicted_class": prediction["predicted_class"],
                "confidence": prediction["confidence"],
                "class_probabilities": prediction.get("class_probabilities", {}),
                "created_at": prediction["created_at"].isoformat(),
                "updated_at": prediction["updated_at"].isoformat(),
                "metadata": prediction.get("metadata", {})
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving prediction"
        )

@router.get("/predictions/{prediction_id}/image", response_class=FileResponse)
async def get_prediction_image(
    prediction_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get the image associated with a prediction.
    
    - **prediction_id**: The ID of the prediction
    
    Returns the image file if available.
    """
    from bson import ObjectId
    
    try:
        db = await db_manager.get_database()
        prediction = await db["predictions"].find_one({"_id": ObjectId(prediction_id)})
        
        if not prediction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Prediction not found"
            )
        
        # Check permissions (user can only see their own predictions unless admin)
        if not current_user.is_superuser and str(prediction.get("user_id")) != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this prediction"
            )
        
        # Check if we have a local file path
        image_path = prediction.get("image_path")
        if not image_path or not os.path.exists(image_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Image file not found"
            )
        
        # Determine content type from file extension
        content_type = "image/jpeg"  # default
        if image_path.lower().endswith('.png'):
            content_type = "image/png"
        
        return FileResponse(
            image_path,
            media_type=content_type,
            filename=os.path.basename(image_path)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prediction image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving prediction image"
        )
