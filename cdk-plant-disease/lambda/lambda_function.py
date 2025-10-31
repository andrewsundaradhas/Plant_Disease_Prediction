import json
import boto3
import os
import tempfile
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import tensorflow as tf
from datetime import datetime
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['PREDICTIONS_TABLE_NAME'])

# Constants
BUCKET_NAME = os.environ['MODEL_BUCKET']
MODEL_KEY = 'models/final_model.h5'
CLASS_INDICES_KEY = 'models/class_indices.json'

# Load model and class indices (cached)
model = None
class_indices = {}

def load_model():
    """Load the model and class indices from S3"""
    global model, class_indices
    
    if model is None or not class_indices:
        with tempfile.NamedTemporaryFile() as tmp1, tempfile.NamedTemporaryFile() as tmp2:
            try:
                # Download model and class indices
                s3.download_file(BUCKET_NAME, MODEL_KEY, tmp1.name)
                s3.download_file(BUCKET_NAME, CLASS_INDICES_KEY, tmp2.name)
                
                # Load model
                model = tf.keras.models.load_model(tmp1.name)
                
                # Load class indices
                with open(tmp2.name, 'r') as f:
                    loaded_indices = json.load(f)
                    class_indices = {int(k): v for k, v in loaded_indices.items()}
                    
                logger.info("Model and class indices loaded successfully")
                
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise
    
    return model, class_indices

def preprocess_image(image_bytes, img_size=(224, 224)):
    """Preprocess the image for prediction"""
    try:
        img = Image.open(BytesIO(image_bytes))
        img = img.convert('RGB')
        img = img.resize(img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def save_prediction_to_dynamo(prediction_id, predicted_class, confidence):
    """Save prediction results to DynamoDB"""
    try:
        table.put_item(Item={
            'prediction_id': prediction_id,
            'predicted_class': predicted_class,
            'confidence': str(confidence),
            'timestamp': datetime.utcnow().isoformat(),
            'ttl': int(datetime.now().timestamp()) + (30 * 24 * 60 * 60)  # 30 days TTL
        })
        logger.info(f"Saved prediction to DynamoDB: {prediction_id}")
    except Exception as e:
        logger.error(f"Error saving to DynamoDB: {str(e)}")
        raise

def lambda_handler(event, context):
    try:
        # Load model (will be cached between invocations)
        model, class_indices = load_model()
        
        # Parse the image from the request
        if 'body' in event:
            if event.get('isBase64Encoded', False):
                image_bytes = base64.b64decode(event['body'])
            else:
                image_bytes = event['body'].encode('utf-8')
        else:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'No image data provided'})
            }
        
        # Preprocess the image
        img_array = preprocess_image(image_bytes)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_idx = int(np.argmax(predictions[0]))
        predicted_class = class_indices.get(predicted_class_idx, "Unknown")
        confidence = float(np.max(predictions[0]))
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = [
            {
                'class': class_indices.get(i, "Unknown"),
                'confidence': float(predictions[0][i])
            }
            for i in top_indices
        ]
        
        # Save to DynamoDB
        prediction_id = context.aws_request_id
        save_prediction_to_dynamo(prediction_id, predicted_class, confidence)
        
        # Format the response
        response = {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST,OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps({
                'prediction_id': prediction_id,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'top_predictions': top_predictions
            })
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }
