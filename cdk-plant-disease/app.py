#!/usr/bin/env python3
from aws_cdk import App, Environment
from stacks.plant_disease_stack import PlantDiseaseStack

app = App()

# Create the stack
PlantDiseaseStack(
    app, "PlantDiseaseStack",
    env=Environment(
        account='YOUR_AWS_ACCOUNT_ID',  # Will be replaced during deployment
        region='us-east-1'  # Change to your preferred region
    )
)

app.synth()
