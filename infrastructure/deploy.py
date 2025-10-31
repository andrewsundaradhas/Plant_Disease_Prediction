#!/usr/bin/env python3
"""
AWS Infrastructure Deployment Script for Crop Health Prediction System
"""
import os
import boto3
import time
import subprocess
from typing import Dict, Any
from botocore.exceptions import ClientError

# Constants
STACK_NAME = 'crop-health-stack'
TEMPLATE_FILE = 'cloudformation.yaml'
REGION = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')

# Initialize AWS clients
cf_client = boto3.client('cloudformation', region_name=REGION)
s3_client = boto3.client('s3', region_name=REGION)
sagemaker_client = boto3.client('sagemaker', region_name=REGION)

def validate_template() -> bool:
    """Validate the CloudFormation template."""
    try:
        with open(TEMPLATE_FILE, 'r') as f:
            template_body = f.read()
        
        cf_client.validate_template(TemplateBody=template_body)
        print("✅ Template validation successful")
        return True
    except ClientError as e:
        print(f"❌ Template validation failed: {e}")
        return False

def deploy_stack() -> bool:
    """Deploy or update the CloudFormation stack."""
    with open(TEMPLATE_FILE, 'r') as f:
        template_body = f.read()
    
    parameters = [
        {
            'ParameterKey': 'EnvironmentName',
            'ParameterValue': 'dev',
            'UsePreviousValue': False
        },
        {
            'ParameterKey': 'ProjectName',
            'ParameterValue': 'crop-health',
            'UsePreviousValue': False
        },
        {
            'ParameterKey': 'SageMakerInstanceType',
            'ParameterValue': 'ml.m5.large',
            'UsePreviousValue': False
        }
    ]
    
    try:
        # Check if stack exists
        cf_client.describe_stacks(StackName=STACK_NAME)
        print(f"Updating stack: {STACK_NAME}")
        response = cf_client.update_stack(
            StackName=STACK_NAME,
            TemplateBody=template_body,
            Parameters=parameters,
            Capabilities=['CAPABILITY_NAMED_IAM'],
            OnFailure='ROLLBACK'
        )
    except cf_client.exceptions.ClientError as e:
        if 'does not exist' in str(e):
            print(f"Creating stack: {STACK_NAME}")
            response = cf_client.create_stack(
                StackName=STACK_NAME,
                TemplateBody=template_body,
                Parameters=parameters,
                Capabilities=['CAPABILITY_NAMED_IAM'],
                OnFailure='ROLLBACK',
                EnableTerminationProtection=False
            )
        else:
            raise
    
    # Wait for stack to be created/updated
    print("Waiting for stack deployment to complete...")
    waiter = cf_client.get_waiter('stack_create_complete')
    waiter.wait(StackName=STACK_NAME, WaiterConfig={'Delay': 30, 'MaxAttempts': 60})
    
    # Get stack outputs
    response = cf_client.describe_stacks(StackName=STACK_NAME)
    outputs = {o['OutputKey']: o['OutputValue'] for o in response['Stacks'][0]['Outputs']}
    
    print("\n🚀 Stack deployment complete!")
    print("\nStack Outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")
    
    return True

def setup_s3_buckets() -> None:
    """Set up S3 buckets with proper configurations."""
    try:
        # Get stack outputs to find bucket names
        response = cf_client.describe_stacks(StackName=STACK_NAME)
        outputs = {o['OutputKey']: o['OutputValue'] for o in response['Stacks'][0]['Outputs']}
        
        model_bucket = outputs.get('ModelBucketName')
        data_bucket = outputs.get('DataBucketName')
        
        if not model_bucket or not data_bucket:
            print("❌ Could not find bucket names in stack outputs")
            return
        
        # Enable CORS on the data bucket
        cors_config = {
            'CORSRules': [
                {
                    'AllowedHeaders': ['*'],
                    'AllowedMethods': ['GET', 'PUT', 'POST', 'DELETE', 'HEAD'],
                    'AllowedOrigins': ['*'],
                    'ExposeHeaders': ['ETag'],
                    'MaxAgeSeconds': 3000
                }
            ]
        }
        
        s3_client.put_bucket_cors(
            Bucket=data_bucket,
            CORSConfiguration=cors_config
        )
        
        # Enable versioning on model bucket
        s3_client.put_bucket_versioning(
            Bucket=model_bucket,
            VersioningConfiguration={'Status': 'Enabled'}
        )
        
        print(f"✅ Configured S3 buckets: {model_bucket}, {data_bucket}")
        
    except Exception as e:
        print(f"❌ Error setting up S3 buckets: {e}")

def setup_dynamodb_table() -> None:
    """Set up DynamoDB table with proper configurations."""
    try:
        # Get stack outputs to find table name
        response = cf_client.describe_stacks(StackName=STACK_NAME)
        outputs = {o['OutputKey']: o['OutputValue'] for o in response['Stacks'][0]['Outputs']}
        
        table_name = outputs.get('PredictionsTableName')
        
        if not table_name:
            print("❌ Could not find DynamoDB table name in stack outputs")
            return
        
        # Add a global secondary index for querying by timestamp
        dynamodb = boto3.resource('dynamodb', region_name=REGION)
        table = dynamodb.Table(table_name)
        
        # Check if the table exists and has the required attributes
        try:
            table.load()
            print(f"✅ DynamoDB table {table_name} is ready")
        except Exception as e:
            print(f"❌ Error accessing DynamoDB table: {e}")
            
    except Exception as e:
        print(f"❌ Error setting up DynamoDB table: {e}")

def deploy_lambda_functions() -> None:
    """Package and deploy Lambda functions."""
    try:
        # Create a temporary directory for packaging
        os.makedirs('.build', exist_ok=True)
        
        # Install dependencies
        print("📦 Installing Python dependencies...")
        subprocess.run(
            ["pip", "install", "-r", "../requirements.txt", "-t", ".build/lambda"],
            check=True
        )
        
        # Copy Lambda function code
        print("📝 Copying Lambda function code...")
        subprocess.run(
            ["cp", "-r", "../backend/app/lambda/", ".build/lambda/"],
            check=True
        )
        
        # Create ZIP archive
        print("📦 Creating deployment package...")
        subprocess.run(
            ["zip", "-r", "function.zip", "."],
            cwd=".build/lambda",
            check=True
        )
        
        # Upload to S3
        print("⬆️  Uploading Lambda function...")
        s3_client.upload_file(
            ".build/lambda/function.zip",
            "crop-health-lambda-code",
            "dev/predict-lambda.zip"
        )
        
        print("✅ Lambda function deployed successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error deploying Lambda function: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        # Clean up
        if os.path.exists(".build"):
            subprocess.run(["rm", "-rf", ".build"], check=True)

def main() -> None:
    """Main deployment function."""
    print("🚀 Starting deployment of Crop Health Prediction System\n")
    
    try:
        # Validate CloudFormation template
        print("🔍 Validating CloudFormation template...")
        if not validate_template():
            return
        
        # Deploy CloudFormation stack
        print("\n☁️  Deploying AWS infrastructure...")
        if not deploy_stack():
            return
        
        # Set up additional resources
        print("\n⚙️  Configuring additional resources...")
        setup_s3_buckets()
        setup_dynamodb_table()
        
        # Deploy Lambda functions
        print("\nλ Deploying Lambda functions...")
        deploy_lambda_functions()
        
        print("\n✨ Deployment completed successfully! ✨")
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        raise

if __name__ == "__main__":
    main()
