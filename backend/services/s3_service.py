"""
S3 Service for handling file storage in AWS S3.
"""
import os
import logging
from typing import Optional, BinaryIO, Dict, Any, List
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from fastapi import UploadFile
import aiofiles
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3Service:
    """Service for interacting with AWS S3."""
    
    def __init__(self, bucket_name: str, region_name: Optional[str] = None):
        """Initialize the S3 service.
        
        Args:
            bucket_name: Name of the S3 bucket
            region_name: AWS region name (default: None, uses default region)
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.s3_resource = boto3.resource('s3', region_name=region_name)
    
    async def upload_file(
        self, 
        file_obj: UploadFile, 
        object_name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None
    ) -> str:
        """Upload a file to S3.
        
        Args:
            file_obj: File-like object to upload
            object_name: S3 object name (key). If not provided, a UUID will be generated.
            metadata: Optional metadata to attach to the object
            content_type: MIME type of the file
            
        Returns:
            str: S3 object key
        """
        import uuid
        from datetime import datetime
        
        try:
            # Generate object name if not provided
            if not object_name:
                file_extension = Path(file_obj.filename).suffix if file_obj.filename else ''
                object_name = f"uploads/{datetime.utcnow().strftime('%Y/%m/%d')}/{uuid.uuid4()}{file_extension}"
            
            # Determine content type
            if not content_type:
                content_type = file_obj.content_type or 'application/octet-stream'
            
            # Prepare upload parameters
            extra_args = {
                'ContentType': content_type,
            }
            
            if metadata:
                extra_args['Metadata'] = metadata
            
            # Read file content
            file_content = await file_obj.read()
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=object_name,
                Body=file_content,
                **extra_args
            )
            
            logger.info(f"File uploaded to S3: {object_name}")
            return object_name
            
        except ClientError as e:
            logger.error(f"Error uploading file to S3: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error uploading file: {str(e)}")
            raise
    
    async def download_file(
        self, 
        object_name: str, 
        local_path: Optional[str] = None
    ) -> str:
        """Download a file from S3.
        
        Args:
            object_name: S3 object key
            local_path: Local path to save the file. If not provided, a temporary file will be created.
            
        Returns:
            str: Path to the downloaded file
        """
        try:
            # Create a temporary file if no path is provided
            if not local_path:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(object_name).suffix)
                local_path = temp_file.name
                temp_file.close()
            
            # Ensure directory exists
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            self.s3_client.download_file(
                Bucket=self.bucket_name,
                Key=object_name,
                Filename=local_path
            )
            
            logger.info(f"File downloaded from S3: {object_name} -> {local_path}")
            return local_path
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.error(f"File not found in S3: {object_name}")
                raise FileNotFoundError(f"File not found in S3: {object_name}")
            logger.error(f"Error downloading file from S3: {str(e)}")
            raise
    
    def get_presigned_url(
        self, 
        object_name: str, 
        expiration: int = 3600
    ) -> str:
        """Generate a presigned URL for an S3 object.
        
        Args:
            object_name: S3 object key
            expiration: Expiration time in seconds (default: 1 hour)
            
        Returns:
            str: Presigned URL
        """
        try:
            response = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': object_name
                },
                ExpiresIn=expiration
            )
            
            logger.debug(f"Generated presigned URL for {object_name}")
            return response
            
        except ClientError as e:
            logger.error(f"Error generating presigned URL: {str(e)}")
            raise
    
    def delete_file(self, object_name: str) -> bool:
        """Delete a file from S3.
        
        Args:
            object_name: S3 object key
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=object_name
            )
            logger.info(f"Deleted file from S3: {object_name}")
            return True
            
        except ClientError as e:
            logger.error(f"Error deleting file from S3: {str(e)}")
            return False
    
    def list_objects(
        self, 
        prefix: str = '', 
        max_keys: int = 1000
    ) -> List[Dict[str, Any]]:
        """List objects in the S3 bucket.
        
        Args:
            prefix: Filter objects with this prefix
            max_keys: Maximum number of objects to return
            
        Returns:
            List of object metadata dictionaries
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            return response.get('Contents', [])
            
        except ClientError as e:
            logger.error(f"Error listing objects in S3: {str(e)}")
            raise
    
    def get_object_metadata(self, object_name: str) -> Dict[str, Any]:
        """Get metadata for an S3 object.
        
        Args:
            object_name: S3 object key
            
        Returns:
            Dictionary containing object metadata
        """
        try:
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=object_name
            )
            
            return {
                'content_length': response.get('ContentLength'),
                'content_type': response.get('ContentType'),
                'last_modified': response.get('LastModified'),
                'metadata': response.get('Metadata', {}),
                'etag': response.get('ETag')
            }
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.error(f"Object not found in S3: {object_name}")
                raise FileNotFoundError(f"Object not found in S3: {object_name}")
            logger.error(f"Error getting object metadata: {str(e)}")
            raise

# Global S3 service instance
_s3_service = None

def get_s3_service() -> S3Service:
    """Get the global S3 service instance (singleton pattern)."""
    global _s3_service
    if _s3_service is None:
        from ...core.config import settings
        _s3_service = S3Service(
            bucket_name=settings.AWS_S3_BUCKET_NAME,
            region_name=settings.AWS_REGION
        )
    return _s3_service
