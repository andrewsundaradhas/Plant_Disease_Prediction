import aws_cdk as core
import aws_cdk.assertions as assertions

from plant_disease_detection.plant_disease_detection_stack import PlantDiseaseDetectionStack

# example tests. To run these tests, uncomment this file along with the example
# resource in plant_disease_detection/plant_disease_detection_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = PlantDiseaseDetectionStack(app, "plant-disease-detection")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
