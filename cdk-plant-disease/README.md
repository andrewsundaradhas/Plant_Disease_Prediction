# Plant Disease Detection - AWS Infrastructure

This project sets up a serverless infrastructure on AWS for plant disease detection using machine learning.

## Prerequisites

1. AWS Account with appropriate permissions
2. AWS CLI configured with credentials
3. Node.js (v14 or later)
4. Python 3.8+
5. AWS CDK Toolkit (npm install -g aws-cdk)

## Setup

1. **Install Dependencies**
   ```bash
   # Create and activate a virtual environment
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   
   # Install Python dependencies
   pip install -r requirements.txt
   
   # Install AWS CDK
   npm install -g aws-cdk
   
   # Bootstrap CDK (only needed once per AWS account/region)
   cdk bootstrap
   ```

2. **Deploy the Stack**
   ```bash
   # Synthesize the CloudFormation template
   cdk synth
   
   # Deploy the stack
   cdk deploy
   ```

3. **After Deployment**
   - The API endpoint will be displayed in the output
   - Model files will be automatically uploaded to S3
   - The DynamoDB table will be created

## Architecture

- **API Gateway**: Handles incoming HTTP requests
- **Lambda**: Processes images and runs predictions
- **S3**: Stores the ML model and uploaded images
- **DynamoDB**: Stores prediction history

## Testing

1. Use the provided `test_api.py` script to test the API:
   ```bash
   python test_api.py path/to/your/image.jpg
   ```

2. Or use curl:
   ```bash
   curl -X POST -H "Content-Type: image/jpeg" --data-binary @path/to/your/image.jpg YOUR_API_ENDPOINT/predict
   ```

## Cleanup

To remove all resources:
```bash
cdk destroy
```

## Cost Considerations

This setup uses AWS Free Tier eligible services:
- Lambda: 1M requests/month
- API Gateway: 1M API calls/month
- S3: 5GB storage
- DynamoDB: 25GB storage

Monitor your usage to stay within free tier limits.
