import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import argparse

# Constants
MODEL_PATH = 'output/final_model.h5'
CLASS_INDICES_PATH = 'output/class_indices.json'
IMG_SIZE = (224, 224)

def load_class_indices():
    """Load class indices from JSON file."""
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    # Invert the dictionary to get class names from indices
    return {v: k for k, v in class_indices.items()}

def preprocess_image(img_path):
    """Preprocess the image for prediction."""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def predict_disease(image_path, model, class_names):
    """Make prediction on a single image."""
    # Preprocess the image
    processed_img = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(processed_img)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    # Get class name
    predicted_class = class_names.get(predicted_class_idx, "Unknown")
    
    # Format the class name for better display
    formatted_name = predicted_class.replace('___', ' - ').replace('_', ' ').title()
    
    # Get top 3 predictions
    top_indices = np.argsort(predictions[0])[-3:][::-1]
    top_predictions = [
        (class_names.get(i, "Unknown"), float(predictions[0][i])) 
        for i in top_indices
    ]
    
    return {
        'predicted_class': formatted_name,
        'confidence': float(confidence),
        'top_predictions': [
            {
                'class': name.replace('___', ' - ').replace('_', ' ').title(),
                'confidence': float(conf)
            } 
            for name, conf in top_predictions
        ]
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict plant disease from an image')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found.")
        return
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print("Error: Model file not found. Please train the model first.")
        return
    
    # Load model and class names
    print("Loading model and class mappings...")
    try:
        model = load_model(MODEL_PATH)
        class_names = load_class_indices()
    except Exception as e:
        print(f"Error loading model or class indices: {e}")
        return
    
    # Make prediction
    print(f"\nAnalyzing image: {args.image_path}")
    try:
        result = predict_disease(args.image_path, model, class_names)
        
        # Print results
        print("\nPrediction Results:")
        print("-" * 50)
        print(f"Most likely disease: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        
        print("\nTop 3 Predictions:")
        for i, pred in enumerate(result['top_predictions'], 1):
            print(f"{i}. {pred['class']}: {pred['confidence']*100:.2f}%")
            
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
