"""
SQS Service for handling background tasks with AWS SQS.
"""
import json
import logging
from typing import Dict, Any, Optional, List
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQSService:
    """Service for interacting with AWS SQS."""
    
    def __init__(self, queue_url: str, region_name: Optional[str] = None):
        """Initialize the SQS service.
        
        Args:
            queue_url: URL of the SQS queue
            region_name: AWS region name (default: None, uses default region)
        """
        self.queue_url = queue_url
        self.sqs_client = boto3.client('sqs', region_name=region_name)
    
    async def send_message(
        self, 
        message_body: Dict[str, Any],
        message_group_id: Optional[str] = None,
        message_deduplication_id: Optional[str] = None,
        delay_seconds: int = 0
    ) -> str:
        """Send a message to the SQS queue.
        
        Args:
            message_body: Dictionary containing the message data
            message_group_id: Message group ID (for FIFO queues)
            message_deduplication_id: Message deduplication ID (for FIFO queues)
            delay_seconds: Delay in seconds before the message becomes available (0-900)
            
        Returns:
            str: Message ID of the sent message
        """
        try:
            message_attributes = {
                'ContentType': {
                    'DataType': 'String',
                    'StringValue': 'application/json'
                }
            }
            
            params = {
                'QueueUrl': self.queue_url,
                'MessageBody': json.dumps(message_body),
                'MessageAttributes': message_attributes,
                'DelaySeconds': delay_seconds
            }
            
            # Add FIFO-specific parameters if provided
            if message_group_id:
                params['MessageGroupId'] = message_group_id
                
                if message_deduplication_id:
                    params['MessageDeduplicationId'] = message_deduplication_id
                else:
                    # If no deduplication ID is provided, use a hash of the message body
                    import hashlib
                    dedup_id = hashlib.md5(json.dumps(message_body).encode()).hexdigest()
                    params['MessageDeduplicationId'] = dedup_id
            
            response = self.sqs_client.send_message(**params)
            message_id = response.get('MessageId')
            
            logger.info(f"Sent message to SQS: {message_id}")
            return message_id
            
        except ClientError as e:
            logger.error(f"Error sending message to SQS: {str(e)}")
            raise
    
    async def receive_messages(
        self,
        max_number_of_messages: int = 1,
        wait_time_seconds: int = 20,
        visibility_timeout: int = 30,
        attribute_names: Optional[List[str]] = None,
        message_attribute_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Receive messages from the SQS queue.
        
        Args:
            max_number_of_messages: Maximum number of messages to retrieve (1-10)
            wait_time_seconds: Long polling wait time in seconds (0-20)
            visibility_timeout: Visibility timeout in seconds (0-43200)
            attribute_names: List of message attributes to retrieve
            message_attribute_names: List of message attribute names to retrieve
            
        Returns:
            List of message dictionaries
        """
        try:
            params = {
                'QueueUrl': self.queue_url,
                'MaxNumberOfMessages': min(10, max(1, max_number_of_messages)),
                'WaitTimeSeconds': min(20, max(0, wait_time_seconds)),
                'VisibilityTimeout': min(43200, max(0, visibility_timeout))
            }
            
            if attribute_names:
                params['AttributeNames'] = attribute_names
                
            if message_attribute_names:
                params['MessageAttributeNames'] = message_attribute_names
            
            response = self.sqs_client.receive_message(**params)
            messages = response.get('Messages', [])
            
            # Parse message bodies
            for message in messages:
                if 'Body' in message:
                    try:
                        message['Body'] = json.loads(message['Body'])
                    except json.JSONDecodeError:
                        pass  # Keep as string if not JSON
            
            logger.info(f"Received {len(messages)} messages from SQS")
            return messages
            
        except ClientError as e:
            logger.error(f"Error receiving messages from SQS: {str(e)}")
            raise
    
    async def delete_message(self, receipt_handle: str) -> bool:
        """Delete a message from the SQS queue.
        
        Args:
            receipt_handle: Receipt handle of the message to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            self.sqs_client.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle
            )
            logger.info(f"Deleted message from SQS: {receipt_handle[:20]}...")
            return True
            
        except ClientError as e:
            logger.error(f"Error deleting message from SQS: {str(e)}")
            return False
    
    async def change_message_visibility(
        self, 
        receipt_handle: str, 
        visibility_timeout: int
    ) -> bool:
        """Change the visibility timeout of a message.
        
        Args:
            receipt_handle: Receipt handle of the message
            visibility_timeout: New visibility timeout in seconds (0-43200)
            
        Returns:
            bool: True if the visibility timeout was changed successfully
        """
        try:
            self.sqs_client.change_message_visibility(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle,
                VisibilityTimeout=visibility_timeout
            )
            logger.info(f"Changed visibility of message {receipt_handle[:20]}... to {visibility_timeout}s")
            return True
            
        except ClientError as e:
            logger.error(f"Error changing message visibility: {str(e)}")
            return False
    
    async def get_queue_attributes(self, attribute_names: List[str] = None) -> Dict[str, str]:
        """Get attributes of the SQS queue.
        
        Args:
            attribute_names: List of attribute names to retrieve (default: All)
            
        Returns:
            Dictionary of queue attributes
        """
        try:
            params = {'QueueUrl': self.queue_url}
            if attribute_names:
                params['AttributeNames'] = attribute_names
                
            response = self.sqs_client.get_queue_attributes(**params)
            return response.get('Attributes', {})
            
        except ClientError as e:
            logger.error(f"Error getting queue attributes: {str(e)}")
            raise
    
    async def purge_queue(self) -> bool:
        """Delete all messages in the queue.
        
        Returns:
            bool: True if the queue was purged successfully
        """
        try:
            self.sqs_client.purge_queue(QueueUrl=self.queue_url)
            logger.info(f"Purged queue: {self.queue_url}")
            return True
            
        except ClientError as e:
            logger.error(f"Error purging queue: {str(e)}")
            return False

# Global SQS service instance
_sqs_service = None

def get_sqs_service() -> SQSService:
    """Get the global SQS service instance (singleton pattern)."""
    global _sqs_service
    if _sqs_service is None:
        from ...core.config import settings
        _sqs_service = SQSService(
            queue_url=settings.AWS_SQS_QUEUE_URL,
            region_name=settings.AWS_REGION
        )
    return _sqs_service

# Background task processor
class BackgroundTaskProcessor:
    """Process background tasks from SQS."""
    
    def __init__(self, sqs_service: SQSService):
        """Initialize the background task processor."""
        self.sqs_service = sqs_service
        self.running = False
    
    async def start(self):
        """Start the background task processor."""
        from ...core.config import settings
        
        self.running = True
        logger.info("Starting background task processor...")
        
        while self.running:
            try:
                # Receive messages from SQS
                messages = await self.sqs_service.receive_messages(
                    max_number_of_messages=10,
                    wait_time_seconds=20,
                    visibility_timeout=300  # 5 minutes visibility timeout
                )
                
                # Process each message
                for message in messages:
                    try:
                        await self.process_message(message)
                        # Delete the message if processing was successful
                        await self.sqs_service.delete_message(message['ReceiptHandle'])
                    except Exception as e:
                        logger.error(f"Error processing message {message.get('MessageId')}: {str(e)}")
                        # Don't delete the message so it can be retried
                        continue
                        
            except Exception as e:
                logger.error(f"Error in background task processor: {str(e)}")
                import asyncio
                await asyncio.sleep(5)  # Wait before retrying
    
    async def stop(self):
        """Stop the background task processor."""
        self.running = False
        logger.info("Stopped background task processor")
    
    async def process_message(self, message: Dict[str, Any]):
        """Process a single message from the queue.
        
        Args:
            message: Message dictionary from SQS
        """
        from ...core.config import settings
        
        message_body = message.get('Body', {})
        message_id = message.get('MessageId')
        receipt_handle = message.get('ReceiptHandle')
        
        logger.info(f"Processing message {message_id}: {message_body.get('task_type', 'unknown')}")
        
        # Extract task information
        task_type = message_body.get('task_type')
        task_data = message_body.get('data', {})
        
        try:
            # Route to the appropriate task handler
            if task_type == 'process_prediction':
                await self._process_prediction_task(task_data)
            elif task_type == 'evaluate_model':
                await self._evaluate_model_task(task_data)
            elif task_type == 'send_notification':
                await self._send_notification_task(task_data)
            else:
                logger.warning(f"Unknown task type: {task_type}")
                
            logger.info(f"Successfully processed message {message_id}")
            
        except Exception as e:
            logger.error(f"Error processing task {task_type}: {str(e)}")
            raise
    
    async def _process_prediction_task(self, task_data: Dict[str, Any]):
        """Process a prediction task."""
        from ...services.model_service import predict_image
        from ...models.prediction_model import PredictionDB
        from ...core.database import get_db
        
        prediction_id = task_data.get('prediction_id')
        if not prediction_id:
            raise ValueError("Missing prediction_id in task data")
        
        # Get database connection
        db = await get_db()
        
        try:
            # Get the prediction from the database
            prediction = await PredictionDB.get_by_id(db, prediction_id)
            if not prediction:
                raise ValueError(f"Prediction not found: {prediction_id}")
            
            # Update status to processing
            await PredictionDB.update(
                db,
                prediction_id,
                {"status": "processing"}
            )
            
            # Process the prediction (simplified example)
            # In a real application, you would load the image and run the model
            result = {
                "class": "healthy",
                "confidence": 0.95,
                "metadata": {"model_version": "1.0.0"}
            }
            
            # Update the prediction with the result
            await PredictionDB.update(
                db,
                prediction_id,
                {
                    "status": "completed",
                    "result": result,
                    "completed_at": datetime.utcnow()
                }
            )
            
        except Exception as e:
            # Update status to failed
            await PredictionDB.update(
                db,
                prediction_id,
                {
                    "status": "failed",
                    "error": str(e),
                    "completed_at": datetime.utcnow()
                }
            )
            raise
    
    async def _evaluate_model_task(self, task_data: Dict[str, Any]):
        """Process a model evaluation task."""
        from ...services.evaluation_service import get_evaluation_service
        from ...models.evaluation_model import EvaluationDB
        from ...core.database import get_db
        
        evaluation_id = task_data.get('evaluation_id')
        if not evaluation_id:
            raise ValueError("Missing evaluation_id in task data")
        
        # Get database connection
        db = await get_db()
        
        try:
            # Get the evaluation from the database
            evaluation = await EvaluationDB.get_by_id(db, evaluation_id)
            if not evaluation:
                raise ValueError(f"Evaluation not found: {evaluation_id}")
            
            # Update status to running
            await EvaluationDB.update(
                db,
                evaluation_id,
                {
                    "status": "running",
                    "started_at": datetime.utcnow()
                }
            )
            
            # Get the evaluation service
            evaluation_service = get_evaluation_service()
            
            # Run the evaluation (simplified example)
            # In a real application, you would evaluate the model on a test dataset
            metrics = {
                "accuracy": 0.95,
                "precision": 0.94,
                "recall": 0.95,
                "f1_score": 0.945
            }
            
            # Update the evaluation with the results
            await EvaluationDB.update(
                db,
                evaluation_id,
                {
                    "status": "completed",
                    "metrics": metrics,
                    "completed_at": datetime.utcnow()
                }
            )
            
        except Exception as e:
            # Update status to failed
            await EvaluationDB.update(
                db,
                evaluation_id,
                {
                    "status": "failed",
                    "error": str(e),
                    "completed_at": datetime.utcnow()
                }
            )
            raise
    
    async def _send_notification_task(self, task_data: Dict[str, Any]):
        """Send a notification."""
        # This is a placeholder for a notification task
        # In a real application, you would send an email, push notification, etc.
        user_id = task_data.get('user_id')
        message = task_data.get('message')
        
        logger.info(f"Sending notification to user {user_id}: {message}")
        
        # Simulate some work
        import asyncio
        await asyncio.sleep(1)

# Global background task processor
_background_processor = None

def get_background_processor() -> BackgroundTaskProcessor:
    """Get the global background task processor (singleton pattern)."""
    global _background_processor
    if _background_processor is None:
        _background_processor = BackgroundTaskProcessor(get_sqs_service())
    return _background_processor

async def start_background_processor():
    """Start the background task processor."""
    processor = get_background_processor()
    await processor.start()

async def stop_background_processor():
    """Stop the background task processor."""
    processor = get_background_processor()
    await processor.stop()
