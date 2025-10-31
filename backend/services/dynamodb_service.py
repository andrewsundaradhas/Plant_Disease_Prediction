"""
DynamoDB Service for storing and retrieving metadata.
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import boto3
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamoDBService:
    """Service for interacting with AWS DynamoDB."""
    
    def __init__(self, table_name: str, region_name: Optional[str] = None):
        """Initialize the DynamoDB service.
        
        Args:
            table_name: Name of the DynamoDB table
            region_name: AWS region name (default: None, uses default region)
        """
        self.table_name = table_name
        self.dynamodb = boto3.resource('dynamodb', region_name=region_name)
        self.table = self.dynamodb.Table(table_name)
    
    async def put_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Put an item in the DynamoDB table.
        
        Args:
            item: Dictionary containing the item to put
            
        Returns:
            Dictionary containing the response from DynamoDB
        """
        try:
            # Ensure timestamps are set
            current_time = datetime.utcnow().isoformat()
            if 'created_at' not in item:
                item['created_at'] = current_time
            item['updated_at'] = current_time
            
            # Put item in DynamoDB
            response = self.table.put_item(Item=item)
            logger.info(f"Item added to {self.table_name}: {item.get('id', 'unknown')}")
            return response
            
        except ClientError as e:
            logger.error(f"Error putting item in DynamoDB: {str(e)}")
            raise
    
    async def get_item(self, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get an item from the DynamoDB table.
        
        Args:
            key: Dictionary containing the primary key of the item to get
            
        Returns:
            Dictionary containing the item if found, None otherwise
        """
        try:
            response = self.table.get_item(Key=key)
            return response.get('Item')
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                logger.error(f"Table not found: {self.table_name}")
                raise
            logger.error(f"Error getting item from DynamoDB: {str(e)}")
            return None
    
    async def update_item(
        self, 
        key: Dict[str, Any], 
        update_expression: str,
        expression_attribute_values: Dict[str, Any],
        return_values: str = 'UPDATED_NEW'
    ) -> Optional[Dict[str, Any]]:
        """Update an item in the DynamoDB table.
        
        Args:
            key: Dictionary containing the primary key of the item to update
            update_expression: Update expression (e.g., 'SET #name = :val1, #status = :val2')
            expression_attribute_values: Values for the update expression
            return_values: What to return after update (default: 'UPDATED_NEW')
            
        Returns:
            Dictionary containing the updated attributes if return_values is specified
        """
        try:
            # Add updated_at timestamp
            expression_attribute_values[':updated_at'] = datetime.utcnow().isoformat()
            if 'SET' in update_expression:
                update_expression += ', updated_at = :updated_at'
            else:
                update_expression = f"{update_expression} SET updated_at = :updated_at"
            
            response = self.table.update_item(
                Key=key,
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_attribute_values,
                ReturnValues=return_values
            )
            
            logger.info(f"Updated item in {self.table_name}: {key}")
            return response.get('Attributes')
            
        except ClientError as e:
            logger.error(f"Error updating item in DynamoDB: {str(e)}")
            raise
    
    async def delete_item(self, key: Dict[str, Any]) -> bool:
        """Delete an item from the DynamoDB table.
        
        Args:
            key: Dictionary containing the primary key of the item to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            response = self.table.delete_item(Key=key)
            logger.info(f"Deleted item from {self.table_name}: {key}")
            return response.get('ResponseMetadata', {}).get('HTTPStatusCode') == 200
            
        except ClientError as e:
            logger.error(f"Error deleting item from DynamoDB: {str(e)}")
            return False
    
    async def query(
        self,
        key_condition_expression: str,
        expression_attribute_values: Dict[str, Any],
        filter_expression: Optional[str] = None,
        index_name: Optional[str] = None,
        limit: int = 100,
        scan_index_forward: bool = True,
        exclusive_start_key: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query items from the DynamoDB table.
        
        Args:
            key_condition_expression: Key condition expression for the query
            expression_attribute_values: Values for the key condition expression
            filter_expression: Optional filter expression
            index_name: Name of the index to query (if using a secondary index)
            limit: Maximum number of items to return
            scan_index_forward: Whether to sort the results in ascending order (default: True)
            exclusive_start_key: Primary key of the first item to evaluate (for pagination)
            
        Returns:
            Dictionary containing the query results and pagination information
        """
        try:
            query_params = {
                'KeyConditionExpression': key_condition_expression,
                'ExpressionAttributeValues': expression_attribute_values,
                'Limit': limit,
                'ScanIndexForward': scan_index_forward
            }
            
            if filter_expression:
                query_params['FilterExpression'] = filter_expression
                
            if index_name:
                query_params['IndexName'] = index_name
                
            if exclusive_start_key:
                query_params['ExclusiveStartKey'] = exclusive_start_key
            
            response = self.table.query(**query_params)
            
            return {
                'items': response.get('Items', []),
                'count': response.get('Count', 0),
                'scanned_count': response.get('ScannedCount', 0),
                'last_evaluated_key': response.get('LastEvaluatedKey')
            }
            
        except ClientError as e:
            logger.error(f"Error querying DynamoDB: {str(e)}")
            raise
    
    async def scan(
        self,
        filter_expression: Optional[str] = None,
        expression_attribute_values: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        exclusive_start_key: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Scan items from the DynamoDB table.
        
        Args:
            filter_expression: Filter expression for the scan
            expression_attribute_values: Values for the filter expression
            limit: Maximum number of items to return
            exclusive_start_key: Primary key of the first item to evaluate (for pagination)
            
        Returns:
            Dictionary containing the scan results and pagination information
        """
        try:
            scan_params = {
                'Limit': limit
            }
            
            if filter_expression and expression_attribute_values:
                scan_params['FilterExpression'] = filter_expression
                scan_params['ExpressionAttributeValues'] = expression_attribute_values
                
            if exclusive_start_key:
                scan_params['ExclusiveStartKey'] = exclusive_start_key
            
            response = self.table.scan(**scan_params)
            
            return {
                'items': response.get('Items', []),
                'count': response.get('Count', 0),
                'scanned_count': response.get('ScannedCount', 0),
                'last_evaluated_key': response.get('LastEvaluatedKey')
            }
            
        except ClientError as e:
            logger.error(f"Error scanning DynamoDB: {str(e)}")
            raise

# Global DynamoDB service instance
_dynamodb_service = None

def get_dynamodb_service() -> DynamoDBService:
    """Get the global DynamoDB service instance (singleton pattern)."""
    global _dynamodb_service
    if _dynamodb_service is None:
        from ...core.config import settings
        _dynamodb_service = DynamoDBService(
            table_name=settings.AWS_DYNAMODB_TABLE_NAME,
            region_name=settings.AWS_REGION
        )
    return _dynamodb_service
