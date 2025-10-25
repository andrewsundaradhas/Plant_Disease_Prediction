""
AWS utility functions for the Crop Health Prediction System.
"""
import os
import json
import boto3
from botocore.exceptions import ClientError
from typing import Dict, Optional, Any, List, BinaryIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3Manager:
    """Manager for AWS S3 operations."""
    
    def __init__(self, bucket_name: str, region_name: str = None):
        """Initialize the S3 manager with the specified bucket.
        
        Args:
            bucket_name: Name of the S3 bucket
            region_name: AWS region name (default: None, uses default region)
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.s3_resource = boto3.resource('s3', region_name=region_name)
        
    def upload_file(
        self, 
        file_obj: BinaryIO, 
        object_name: str, 
        content_type: str = 'application/octet-stream',
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Upload a file to S3.
        
        Args:
            file_obj: File-like object to upload
            object_name: S3 object name (key)
            content_type: MIME type of the file
            metadata: Optional metadata to attach to the object
            
        Returns:
            str: URL of the uploaded file
        """
        try:
            extra_args = {'ContentType': content_type}
            if metadata:
                extra_args['Metadata'] = metadata
                
            self.s3_client.upload_fileobj(
                file_obj,
                self.bucket_name,
                object_name,
                ExtraArgs=extra_args
            )
            
            url = f"https://{self.bucket_name}.s3.amazonaws.com/{object_name}"
            logger.info(f"File uploaded to {url}")
            return url
            
        except ClientError as e:
            logger.error(f"Error uploading file to S3: {str(e)}")
            raise
    
    def download_file(self, object_name: str, file_path: str) -> str:
        """Download a file from S3.
        
        Args:
            object_name: S3 object name (key)
            file_path: Local path to save the file
            
        Returns:
            str: Path to the downloaded file
        """
        try:
            self.s3_client.download_file(
                self.bucket_name, 
                object_name, 
                file_path
            )
            logger.info(f"File downloaded to {file_path}")
            return file_path
            
        except ClientError as e:
            logger.error(f"Error downloading file from S3: {str(e)}")
            raise
    
    def delete_file(self, object_name: str) -> bool:
        """Delete a file from S3.
        
        Args:
            object_name: S3 object name (key)
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=object_name
            )
            logger.info(f"Deleted S3 object: {object_name}")
            return True
            
        except ClientError as e:
            logger.error(f"Error deleting file from S3: {str(e)}")
            return False


class DynamoDBManager:
    """Manager for AWS DynamoDB operations."""
    
    def __init__(self, table_name: str, region_name: str = None):
        """Initialize the DynamoDB manager with the specified table.
        
        Args:
            table_name: Name of the DynamoDB table
            region_name: AWS region name (default: None, uses default region)
        """
        self.table_name = table_name
        self.dynamodb = boto3.resource('dynamodb', region_name=region_name)
        self.table = self.dynamodb.Table(table_name)
    
    def put_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Put an item in the DynamoDB table.
        
        Args:
            item: Dictionary containing the item to put
            
        Returns:
            Dict: The response from DynamoDB
        """
        try:
            response = self.table.put_item(Item=item)
            logger.info(f"Item added to {self.table_name}")
            return response
            
        except ClientError as e:
            logger.error(f"Error putting item in DynamoDB: {str(e)}")
            raise
    
    def get_item(self, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get an item from the DynamoDB table.
        
        Args:
            key: Dictionary containing the primary key of the item to get
            
        Returns:
            Optional[Dict]: The item if found, None otherwise
        """
        try:
            response = self.table.get_item(Key=key)
            return response.get('Item')
            
        except ClientError as e:
            logger.error(f"Error getting item from DynamoDB: {str(e)}")
            raise
    
    def query_items(
        self, 
        key_condition_expression: str, 
        expression_attribute_values: Dict[str, Any],
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query items from the DynamoDB table.
        
        Args:
            key_condition_expression: The condition that specifies the key value(s) for items to be retrieved
            expression_attribute_values: One or more values that can be substituted in the expression
            limit: Maximum number of items to return
            
        Returns:
            List[Dict]: List of items matching the query
        """
        try:
            response = self.table.query(
                KeyConditionExpression=key_condition_expression,
                ExpressionAttributeValues=expression_attribute_values,
                Limit=limit
            )
            return response.get('Items', [])
            
        except ClientError as e:
            logger.error(f"Error querying items from DynamoDB: {str(e)}")
            raise


class SageMakerManager:
    """Manager for AWS SageMaker operations."""
    
    def __init__(self, endpoint_name: str, region_name: str = None):
        """Initialize the SageMaker manager with the specified endpoint.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            region_name: AWS region name (default: None, uses default region)
        """
        self.endpoint_name = endpoint_name
        self.sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=region_name)
    
    def invoke_endpoint(
        self, 
        body: bytes, 
        content_type: str = 'application/x-image',
        accept: str = 'application/json'
    ) -> Dict[str, Any]:
        """Invoke a SageMaker endpoint.
        
        Args:
            body: The input data for the endpoint
            content_type: The MIME type of the input data
            accept: The MIME type of the response
            
        Returns:
            Dict: The response from the endpoint
        """
        try:
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                Body=body,
                ContentType=content_type,
                Accept=accept
            )
            
            # Parse the response body
            result = response['Body'].read().decode('utf-8')
            return json.loads(result)
            
        except ClientError as e:
            logger.error(f"Error invoking SageMaker endpoint: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing SageMaker response: {str(e)}")
            raise


def get_aws_region() -> str:
    """Get the AWS region from environment variables or config."""
    return os.environ.get('AWS_REGION', 'us-east-1')

def get_aws_credentials() -> Dict[str, str]:
    """Get AWS credentials from environment variables."""
    return {
        'aws_access_key_id': os.environ.get('AWS_ACCESS_KEY_ID', ''),
        'aws_secret_access_key': os.environ.get('AWS_SECRET_ACCESS_KEY', ''),
        'region_name': get_aws_region()
    }

def is_aws_configured() -> bool:
    """Check if AWS credentials are configured."""
    return all([
        os.environ.get('AWS_ACCESS_KEY_ID'),
        os.environ.get('AWS_SECRET_ACCESS_KEY'),
        os.environ.get('AWS_REGION')
    ])
