from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_lambda as _lambda,
    aws_apigateway as apigw,
    aws_dynamodb as dynamodb,
    aws_iam as iam,
    aws_s3_deployment as s3deploy,
    RemovalPolicy,
    Duration,
)
from constructs import Construct
import os

class PlantDiseaseStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Create S3 bucket for model and images
        bucket = s3.Bucket(
            self, "PlantDiseaseBucket",
            versioned=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            cors=[
                s3.CorsRule(
                    allowed_methods=[s3.HttpMethods.GET, s3.HttpMethods.PUT, s3.HttpMethods.POST],
                    allowed_origins=["*"],
                    allowed_headers=["*"]
                )
            ]
        )

        # Create DynamoDB table for predictions
        predictions_table = dynamodb.Table(
            self, "PredictionsTable",
            partition_key={"name": "prediction_id", "type": dynamodb.AttributeType.STRING},
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY
        )

        # Create Lambda execution role
        lambda_role = iam.Role(
            self, "LambdaExecutionRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
            ]
        )

        # Add permissions
        bucket.grant_read_write(lambda_role)
        predictions_table.grant_read_write_data(lambda_role)

        # Create Lambda function
        lambda_fn = _lambda.Function(
            self, "PlantDiseasePredictor",
            runtime=_lambda.Runtime.PYTHON_3_9,
            code=_lambda.Code.from_asset("lambda"),
            handler="lambda_function.lambda_handler",
            role=lambda_role,
            environment={
                "MODEL_BUCKET": bucket.bucket_name,
                "PREDICTIONS_TABLE_NAME": predictions_table.table_name
            },
            timeout=Duration.seconds(30),
            memory_size=1024
        )

        # Create API Gateway
        api = apigw.LambdaRestApi(
            self, "PlantDiseaseAPI",
            handler=lambda_fn,
            proxy=False,
            default_cors_preflight_options={
                "allow_origins": apigw.Cors.ALL_ORIGINS,
                "allow_methods": apigw.Cors.ALL_METHODS
            }
        )

        # Add resource and method
        predict = api.root.add_resource("predict")
        predict.add_method("POST")

        # Upload model files to S3
        s3deploy.BucketDeployment(
            self, "DeployModel",
            sources=[s3deploy.Source.asset("../output")],
            destination_bucket=bucket,
            destination_key_prefix="models"
        )

        # Output the API endpoint
        self.api_endpoint = api.url
