"""
AWS Lambda function for handling crop health prediction requests.
"""
import os
import json
import boto3
import tempfile
import urllib.parse
from datetime import datetime
from typing import Dict, Any, Optional
import base64

# Initialize AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
sagemaker_runtime = boto3.client('sagemaker-runtime')

# Environment variables
MODEL_BUCKET = os.environ['MODEL_BUCKET']
DATA_BUCKET = os.environ['DATA_BUCKET']
TABLE_NAME = os.environ['TABLE_NAME']

# Constants
MODEL_NAME = 'crop-health-model'
MODEL_VERSION = '1.0.0'
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Handle incoming API Gateway requests for image prediction.
    """
    try:
        # Parse request
        http_method = event.get('httpMethod', '')
        
        if http_method == 'POST':
            return handle_prediction_request(event)
        elif http_method == 'GET':
            return handle_status_request(event)
        else:
            return create_response(405, {'error': 'Method not allowed'})
            
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return create_response(500, {'error': 'Internal server error'})

def handle_prediction_request(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle prediction request with image upload."""
    try:
        # Parse request body
        if event.get('isBase64Encoded', False):
            body = base64.b64decode(event['body']).decode('utf-8')
        else:
            body = event.get('body', '{}')
            
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return create_response(400, {'error': 'Invalid JSON in request body'})
        
        # Check for image data
        if 'image' not in data:
            return create_response(400, {'error': 'No image data provided'})
            
        # Process image
        try:
            image_data = base64.b64decode(data['image'])
            
            # Validate image size
            if len(image_data) > MAX_IMAGE_SIZE:
                return create_response(400, {'error': f'Image size exceeds {MAX_IMAGE_SIZE} bytes limit'})
                
            # Generate unique filename
            timestamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
            file_extension = '.jpg'  # Default to jpg
            file_key = f'uploads/{timestamp}{file_extension}'
            
            # Upload to S3
            s3_client.put_object(
                Bucket=DATA_BUCKET,
                Key=file_key,
                Body=image_data,
                ContentType='image/jpeg'
            )
            
            # Call SageMaker endpoint for prediction
            prediction = predict_image(image_data)
            
            # Save to DynamoDB
            save_prediction(file_key, prediction)
            
            return create_response(200, {
                'status': 'success',
                'prediction': prediction,
                'image_url': f's3://{DATA_BUCKET}/{file_key}'
            })
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return create_response(400, {'error': 'Failed to process image'})
            
    except Exception as e:
        print(f"Error in prediction handler: {str(e)}")
        return create_response(500, {'error': 'Internal server error'})

def predict_image(image_data: bytes) -> Dict[str, Any]:
    """
    Call SageMaker endpoint to get prediction for the image.
    """
    try:
        # Call SageMaker endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=MODEL_NAME,
            ContentType='application/x-image',
            Body=image_data
        )
        
        # Parse prediction result
        result = json.loads(response['Body'].read().decode('utf-8'))
        
        # Format prediction result
        return {
            'class_id': int(result.get('predictions', [{}])[0].get('predicted_class', -1)),
            'class_name': result.get('predictions', [{}])[0].get('class_name', 'unknown'),
            'confidence': float(result.get('predictions', [{}])[0].get('confidence', 0.0)),
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        print(f"Error calling SageMaker endpoint: {str(e)}")
        raise

def save_prediction(image_key: str, prediction: Dict[str, Any]) -> None:
    """Save prediction result to DynamoDB."""
    try:
        table = dynamodb.Table(TABLE_NAME)
        
        item = {
            'prediction_id': f"pred-{datetime.utcnow().timestamp()}",
            'image_key': image_key,
            'timestamp': int(datetime.utcnow().timestamp()),
            'prediction': prediction,
            'status': 'completed',
            'expiry_time': int((datetime.utcnow() + timedelta(days=30)).timestamp())
        }
        
        table.put_item(Item=item)
        
    except Exception as e:
        print(f"Error saving to DynamoDB: {str(e)}")
        raise

def handle_status_request(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle status check request."""
    try:
        # Get prediction ID from query string
        prediction_id = event.get('queryStringParameters', {}).get('id')
        
        if not prediction_id:
            return create_response(400, {'error': 'Missing prediction ID'})
            
        # Query DynamoDB
        table = dynamodb.Table(TABLE_NAME)
        response = table.get_item(Key={'prediction_id': prediction_id})
        
        if 'Item' not in response:
            return create_response(404, {'error': 'Prediction not found'})
            
        item = response['Item']
        
        # Generate pre-signed URL for the image
        try:
            image_url = s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': DATA_BUCKET,
                    'Key': item['image_key']
                },
                ExpiresIn=3600  # 1 hour
            )
            item['image_url'] = image_url
        except Exception as e:
            print(f"Error generating pre-signed URL: {str(e)}")
            item['image_url'] = None
        
        return create_response(200, item)
        
    except Exception as e:
        print(f"Error in status handler: {str(e)}")
        return create_response(500, {'error': 'Internal server error'})

def create_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    """Create API Gateway response."""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization'
        },
        'body': json.dumps(body),
        'isBase64Encoded': False
    }
