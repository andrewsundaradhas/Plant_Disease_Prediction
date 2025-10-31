import requests
import sys
import json

def test_api(image_path, api_endpoint):
    """Test the prediction API with an image."""
    try:
        with open(image_path, 'rb') as img_file:
            files = {'file': img_file}
            response = requests.post(f"{api_endpoint}/predict", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print("\n✅ Prediction successful!")
                print(f"\nPredicted Class: {result.get('predicted_class')}")
                print(f"Confidence: {result.get('confidence')*100:.2f}%")
                
                print("\nTop Predictions:")
                for i, pred in enumerate(result.get('top_predictions', []), 1):
                    print(f"{i}. {pred.get('class')}: {pred.get('confidence')*100:.2f}%")
                
                return True
            else:
                print(f"\n❌ Error: {response.status_code}")
                print(response.text)
                return False
                
    except Exception as e:
        print(f"\n❌ Error testing API: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <path_to_image> [api_endpoint]")
        print("Example: python test_api.py test_image.jpg https://your-api.execute-api.region.amazonaws.com/prod")
        sys.exit(1)
        
    image_path = sys.argv[1]
    api_endpoint = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:3000"
    
    print(f"\nTesting API with image: {image_path}")
    print(f"API Endpoint: {api_endpoint}")
    
    test_api(image_path, api_endpoint)
