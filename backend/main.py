from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import shutil
import uuid
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# Import ML components
from ml.inference import CropHealthPredictor
from ml.model_evaluation import evaluate_model

# Initialize FastAPI app
app = FastAPI(
    title="Crop Health Prediction API",
    description="API for predicting crop health and disease detection",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = "uploads"
MODEL_PATH = "ml/models/crop_health_model"
CLASS_MAPPING_PATH = "ml/class_mapping.json"
EVALUATION_RESULTS_DIR = "evaluation_results"

# Ensure evaluation results directory exists
os.makedirs(EVALUATION_RESULTS_DIR, exist_ok=True)

# Global variable to track evaluation status
evaluation_in_progress = False
evaluation_results = {}

# Lock for thread-safe operations
from threading import Lock
evaluation_lock = Lock()

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize predictor (will be loaded on first request)
predictor = None

def get_predictor():
    """Lazy load the predictor to improve startup time."""
    global predictor
    if predictor is None:
        try:
            # Check if model files exist
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
                
            if not os.path.exists(CLASS_MAPPING_PATH):
                raise FileNotFoundError(f"Class mapping not found at {CLASS_MAPPING_PATH}")
                
            predictor = CropHealthPredictor(MODEL_PATH, CLASS_MAPPING_PATH)
            
            # Verify model is properly loaded
            if not hasattr(predictor, 'model') or predictor.model is None:
                raise ValueError("Failed to load model: model attribute is None")
                
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model: {str(e)}"
            )
    return predictor

# Models
class PredictionResult(BaseModel):
    image_path: str
    predicted_class: str
    confidence: float
    class_probabilities: dict
    timestamp: str

class HealthCheck(BaseModel):
    status: str = "ok"
    model_loaded: bool = False
    evaluation_in_progress: bool = False

class EvaluationMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    class_metrics: Dict[str, Dict[str, float]]
    confusion_matrix: List[List[int]]
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class EvaluationRequest(BaseModel):
    test_data_dir: str
    batch_size: int = 32

# Routes
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Crop Health Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        get_predictor()
        return {"status": "ok", "model_loaded": True}
    except Exception:
        return {"status": "ok", "model_loaded": False}

@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Make a prediction on an uploaded image.
    
    - **file**: Image file to analyze (JPEG, PNG, etc.)
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    # Generate unique filename
    file_ext = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Make prediction
        predictor = get_predictor()
        result = predictor.predict(file_path)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Add timestamp
        result["timestamp"] = datetime.utcnow().isoformat()
        
        return result
        
    except Exception as e:
        # Clean up file if there was an error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/classes", response_model=List[str])
async def list_classes():
    """List all available crop disease classes."""
    try:
        predictor = get_predictor()
        return list(predictor.class_mapping.values())
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list classes: {str(e)}"
        )

@app.post("/api/evaluate", response_model=dict)
async def evaluate_model_endpoint(background_tasks: BackgroundTasks, request: EvaluationRequest):
    """
    Evaluate the model on the provided test dataset.
    
    - **test_data_dir**: Path to the directory containing test images
    - **batch_size**: Batch size for evaluation (default: 32)
    """
    global evaluation_in_progress, evaluation_results
    
    if evaluation_in_progress:
        raise HTTPException(
            status_code=400,
            detail="Evaluation is already in progress"
        )
    
    if not os.path.isdir(request.test_data_dir):
        raise HTTPException(
            status_code=400,
            detail=f"Test data directory not found: {request.test_data_dir}"
        )
    
    # Start evaluation in background
    background_tasks.add_task(
        run_model_evaluation,
        request.test_data_dir,
        request.batch_size
    )
    
    return {"status": "evaluation_started", "message": "Model evaluation has started in the background"}

@app.get("/api/evaluation/status", response_model=dict)
async def get_evaluation_status():
    """Get the status of the model evaluation."""
    global evaluation_in_progress, evaluation_results
    return {
        "evaluation_in_progress": evaluation_in_progress,
        "results_available": bool(evaluation_results),
        "last_updated": evaluation_results.get("timestamp") if evaluation_results else None
    }

@app.get("/api/evaluation/results", response_model=EvaluationMetrics)
async def get_evaluation_results():
    """Get the results of the last model evaluation."""
    global evaluation_results
    if not evaluation_results:
        raise HTTPException(
            status_code=404,
            detail="No evaluation results available. Run an evaluation first."
        )
    return evaluation_results

def run_model_evaluation(test_data_dir: str, batch_size: int):
    """Run model evaluation and store results."""
    global evaluation_in_progress, evaluation_results
    
    with evaluation_lock:
        if evaluation_in_progress:
            return
        evaluation_in_progress = True
    
    try:
        # Run evaluation
        results = evaluate_model(
            model_path=MODEL_PATH,
            test_dir=test_data_dir,
            batch_size=batch_size
        )
        
        # Save evaluation results
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(EVALUATION_RESULTS_DIR, f"evaluation_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Update global results
        with evaluation_lock:
            evaluation_results = results
            evaluation_results["timestamp"] = datetime.utcnow().isoformat()
            
    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")
        # Optionally save error information
        with evaluation_lock:
            evaluation_results = {
                "error": str(e),
                "status": "error",
                "timestamp": datetime.utcnow().isoformat()
            }
    finally:
        with evaluation_lock:
            evaluation_in_progress = False

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
