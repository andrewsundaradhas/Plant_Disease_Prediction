import os
import io
import json
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

# Constants
MODEL_PATH = 'output/final_model.h5'
CLASS_INDICES_PATH = 'output/class_indices.json'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = (224, 224)

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and class indices
model = None
class_names = {}

def load_model_and_classes():
    global model, class_names
    try:
        model = load_model(MODEL_PATH)
        with open(CLASS_INDICES_PATH, 'r') as f:
            class_indices = json.load(f)
        class_names = {v: k for k, v in class_indices.items()}
        print("Model and class mappings loaded successfully!")
    except Exception as e:
        print(f"Error loading model or class indices: {e}")
        raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img):
    """Preprocess the image for prediction."""
    if isinstance(img, str):
        img = image.load_img(img, target_size=IMG_SIZE)
    img = img.resize(IMG_SIZE) if not isinstance(img, Image.Image) else img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            try:
                # Read image file
                img = Image.open(io.BytesIO(file.read()))
                
                # Preprocess and predict
                processed_img = preprocess_image(img)
                predictions = model.predict(processed_img)
                
                # Get top 3 predictions
                top_indices = np.argsort(predictions[0])[-3:][::-1]
                top_predictions = [
                    {
                        'class': class_names.get(i, 'Unknown').replace('___', ' - ').replace('_', ' ').title(),
                        'confidence': float(predictions[0][i])
                    }
                    for i in top_indices
                ]
                
                # Save the uploaded file
                img_path = os.path.join(UPLOAD_FOLDER, file.filename)
                img.save(img_path)
                
                return jsonify({
                    'status': 'success',
                    'image': file.filename,
                    'predictions': top_predictions
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    return render_template('index.html')

@app.route('/uploaded/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    # Load model and class indices when the app starts
    load_model_and_classes()
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create a simple HTML template if it doesn't exist
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Plant Disease Classifier</title>
                <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
            </head>
            <body class="bg-gray-100 min-h-screen">
                <div class="container mx-auto px-4 py-8">
                    <h1 class="text-3xl font-bold text-center mb-8">Plant Disease Classifier</h1>
                    
                    <div class="max-w-2xl mx-auto bg-white rounded-lg shadow-md p-6">
                        <div class="text-center mb-6">
                            <h2 class="text-xl font-semibold mb-2">Upload a plant leaf image</h2>
                            <p class="text-gray-600 mb-4">Supported formats: JPG, JPEG, PNG</p>
                        </div>
                        
                        <form id="uploadForm" class="mb-6">
                            <div class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-blue-400 transition-colors" id="dropZone">
                                <input type="file" id="fileInput" class="hidden" accept="image/*">
                                <div id="uploadContent">
                                    <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
                                    </svg>
                                    <p class="mt-2 text-sm text-gray-600">Click to upload or drag and drop</p>
                                    <p class="text-xs text-gray-500 mt-1">PNG, JPG, JPEG up to 5MB</p>
                                </div>
                            </div>
                            <div class="mt-4 flex justify-center">
                                <button type="submit" class="bg-blue-500 hover:bg-blue-600 text-white font-medium py-2 px-6 rounded-md transition-colors" id="predictBtn">
                                    Predict Disease
                                </button>
                            </div>
                        </form>
                        
                        <div id="result" class="hidden">
                            <h3 class="text-lg font-medium mb-3">Results</h3>
                            <div class="space-y-4">
                                <div id="imagePreview" class="max-w-sm mx-auto mb-4"></div>
                                <div id="predictions" class="space-y-2"></div>
                            </div>
                        </div>
                        
                        <div id="loading" class="hidden text-center py-8">
                            <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
                            <p class="mt-2 text-gray-600">Analyzing image...</p>
                        </div>
                        
                        <div id="error" class="hidden bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
                            <strong class="font-bold">Error! </strong>
                            <span class="block sm:inline" id="errorMessage"></span>
                        </div>
                    </div>
                </div>
                
                <script>
                    const dropZone = document.getElementById('dropZone');
                    const fileInput = document.getElementById('fileInput');
                    const uploadForm = document.getElementById('uploadForm');
                    const uploadContent = document.getElementById('uploadContent');
                    const resultDiv = document.getElementById('result');
                    const loadingDiv = document.getElementById('loading');
                    const errorDiv = document.getElementById('error');
                    const errorMessage = document.getElementById('errorMessage');
                    const predictionsDiv = document.getElementById('predictions');
                    const imagePreview = document.getElementById('imagePreview');
                    const predictBtn = document.getElementById('predictBtn');
                    
                    // Prevent default drag behaviors
                    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                        dropZone.addEventListener(eventName, preventDefaults, false);
                        document.body.addEventListener(eventName, preventDefaults, false);
                    });
                    
                    // Highlight drop zone when item is dragged over it
                    ['dragenter', 'dragover'].forEach(eventName => {
                        dropZone.addEventListener(eventName, highlight, false);
                    });
                    
                    ['dragleave', 'drop'].forEach(eventName => {
                        dropZone.addEventListener(eventName, unhighlight, false);
                    });
                    
                    // Handle dropped files
                    dropZone.addEventListener('drop', handleDrop, false);
                    
                    // Handle click on drop zone
                    dropZone.addEventListener('click', () => {
                        fileInput.click();
                    });
                    
                    // Handle file selection
                    fileInput.addEventListener('change', handleFiles);
                    
                    // Handle form submission
                    uploadForm.addEventListener('submit', handleSubmit);
                    
                    function preventDefaults(e) {
                        e.preventDefault();
                        e.stopPropagation();
                    }
                    
                    function highlight() {
                        dropZone.classList.add('border-blue-400', 'bg-blue-50');
                    }
                    
                    function unhighlight() {
                        dropZone.classList.remove('border-blue-400', 'bg-blue-50');
                    }
                    
                    function handleDrop(e) {
                        const dt = e.dataTransfer;
                        const files = dt.files;
                        handleFiles({ target: { files } });
                    }
                    
                    function handleFiles(e) {
                        const files = e.target.files;
                        if (files.length > 0) {
                            const file = files[0];
                            if (file.type.match('image.*')) {
                                // Show preview
                                const reader = new FileReader();
                                reader.onload = function(e) {
                                    uploadContent.innerHTML = `
                                        <img src="${e.target.result}" class="max-h-40 mx-auto mb-2 rounded">
                                        <p class="text-sm text-gray-600">${file.name}</p>
                                        <p class="text-xs text-gray-500">${(file.size / 1024).toFixed(1)} KB</p>
                                    `;
                                };
                                reader.readAsDataURL(file);
                            } else {
                                showError('Please upload an image file (JPG, JPEG, PNG)');
                            }
                        }
                    }
                    
                    async function handleSubmit(e) {
                        e.preventDefault();
                        
                        const file = fileInput.files[0];
                        if (!file) {
                            showError('Please select an image file first');
                            return;
                        }
                        
                        if (!file.type.match('image.*')) {
                            showError('Please upload an image file (JPG, JPEG, PNG)');
                            return;
                        }
                        
                        // Show loading state
                        loadingDiv.classList.remove('hidden');
                        resultDiv.classList.add('hidden');
                        errorDiv.classList.add('hidden');
                        predictBtn.disabled = true;
                        
                        const formData = new FormData();
                        formData.append('file', file);
                        
                        try {
                            const response = await fetch('/', {
                                method: 'POST',
                                body: formData
                            });
                            
                            const data = await response.json();
                            
                            if (response.ok) {
                                // Show results
                                showResults(data);
                            } else {
                                throw new Error(data.error || 'Failed to process image');
                            }
                        } catch (error) {
                            console.error('Error:', error);
                            showError(error.message || 'An error occurred while processing your request');
                        } finally {
                            loadingDiv.classList.add('hidden');
                            predictBtn.disabled = false;
                        }
                    }
                    
                    function showResults(data) {
                        // Show image preview
                        imagePreview.innerHTML = `
                            <img src="/uploaded/${data.image}" class="max-w-full h-auto rounded-lg shadow-md">
                        `;
                        
                        // Show predictions
                        predictionsDiv.innerHTML = data.predictions.map((pred, index) => `
                            <div class="bg-gray-50 p-3 rounded-md">
                                <div class="flex justify-between items-center mb-1">
                                    <span class="font-medium ${index === 0 ? 'text-green-600' : 'text-gray-700'}">
                                        ${index + 1}. ${pred.class}
                                    </span>
                                    <span class="text-sm font-medium ${index === 0 ? 'text-green-600' : 'text-gray-500'}">
                                        ${(pred.confidence * 100).toFixed(2)}%
                                    </span>
                                </div>
                                <div class="w-full bg-gray-200 rounded-full h-2">
                                    <div class="bg-${index === 0 ? 'green' : 'blue'}-500 h-2 rounded-full" 
                                         style="width: ${pred.confidence * 100}%">
                                    </div>
                                </div>
                            </div>
                        `).join('');
                        
                        // Show result section
                        resultDiv.classList.remove('hidden');
                    }
                    
                    function showError(message) {
                        errorMessage.textContent = message;
                        errorDiv.classList.remove('hidden');
                        setTimeout(() => {
                            errorDiv.classList.add('hidden');
                        }, 5000);
                    }
                </script>
            </body>
            </html>
            """)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
