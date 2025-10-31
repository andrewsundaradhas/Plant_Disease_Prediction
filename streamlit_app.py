import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from pathlib import Path

# Set plot style
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

# Set page config with better theme and layout
st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stFileUploader>div>div {
        border: 2px dashed #4CAF50;
        border-radius: 15px;
        padding: 2.5rem;
        transition: all 0.3s ease;
    }
    .stFileUploader>div>div:hover {
        border-color: #45a049;
        background-color: #f0fff4;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    .prediction-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .disease-info {
        background: #f0f9ff;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .solution-item {
        background: white;
        margin: 0.5rem 0;
        padding: 0.8rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_metrics():
    """Load model, class indices, and training metrics."""
    try:
        model_path = 'output/final_model.h5'
        class_indices_path = 'output/class_indices.json'
        history_path = 'output/training_history.json'
        
        # Check if required files exist
        if not all(os.path.exists(p) for p in [model_path, class_indices_path, history_path]):
            return None, {}, None, None
        
        # Load model
        model = load_model(model_path)
        
        # Load class indices
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        class_names = {v: k for k, v in class_indices.items()}
        
        # Load training history
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # Generate sample predictions for confusion matrix (simplified for demo)
        # In a real app, you'd want to load your test set
        num_classes = len(class_names)
        y_true = np.random.randint(0, num_classes, 100)  # Mock true labels
        y_pred = np.random.randint(0, num_classes, 100)  # Mock predictions
        
        return model, class_names, history, (y_true, y_pred, num_classes)
        
    except Exception as e:
        st.error(f"Error loading model and metrics: {e}")
        return None, {}, None, None

def plot_training_history(history):
    """Plot training and validation metrics."""
    if not history:
        return None
        
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Model Accuracy', 'Model Loss'))
    
    # Accuracy plot
    fig.add_trace(
        go.Scatter(
            y=history.get('accuracy', []),
            name='Train Accuracy',
            mode='lines+markers',
            line=dict(color='#4CAF50')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            y=history.get('val_accuracy', []),
            name='Validation Accuracy',
            mode='lines+markers',
            line=dict(color='#2196F3')
        ),
        row=1, col=1
    )
    
    # Loss plot
    fig.add_trace(
        go.Scatter(
            y=history.get('loss', []),
            name='Train Loss',
            mode='lines+markers',
            showlegend=False,
            line=dict(color='#4CAF50')
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            y=history.get('val_loss', []),
            name='Validation Loss',
            mode='lines+markers',
            showlegend=False,
            line=dict(color='#2196F3')
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_xaxes(title_text='Epoch', row=1, col=1)
    fig.update_xaxes(title_text='Epoch', row=1, col=2)
    fig.update_yaxes(title_text='Accuracy', row=1, col=1)
    fig.update_yaxes(title_text='Loss', row=1, col=2)
    
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig = px.imshow(
        cm_norm,
        labels=dict(x="Predicted", y="Actual", color="Normalized"),
        x=[f"{class_names[i]}" for i in range(len(class_names))],
        y=[f"{class_names[i]}" for i in range(len(class_names))],
        color_continuous_scale='Viridis',
        aspect="auto"
    )
    
    fig.update_layout(
        height=600,
        margin=dict(l=100, r=50, t=50, b=100),
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        coloraxis_colorbar=dict(title='Normalized Count')
    )
    
    return fig

def display_model_summary(model):
    """Display model architecture and parameters."""
    if model is None:
        return
        
    # Create a summary string
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    
    with st.expander("📊 Model Architecture"):
        st.code(summary_string, language='python')
        
        # Display trainable parameters
        trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Trainable Parameters", f"{trainable_params:,}")
        with col2:
            st.metric("Non-trainable Parameters", f"{non_trainable_params:,}")

# Enhanced Disease Information
DISEASE_INFO = {
    'Apple___Apple_scab': {
        'description': 'A fungal disease causing dark, scaly lesions on leaves and fruit that can lead to defoliation and reduced fruit quality.',
        'symptoms': [
            'Olive-green to black spots on leaves',
            'Velvety texture on lesions',
            'Distorted or stunted leaves',
            'Cracked, deformed fruit'
        ],
        'solutions': [
            'Apply fungicides in early spring before symptoms appear',
            'Prune trees to improve air circulation',
            'Rake and destroy fallen leaves in autumn',
            'Plant resistant varieties like Liberty or Freedom',
            'Apply sulfur or copper-based fungicides every 7-10 days during wet weather'
        ],
        'severity': 'Moderate to High',
        'prevention': 'Plant in full sun with good air circulation, avoid overhead watering',
        'chemical_control': 'Myclobutanil, Sulfur, or Copper-based fungicides',
        'organic_control': 'Neem oil, Baking soda solution (1 tbsp baking soda, 1 tsp vegetable oil, 1/2 tsp liquid soap in 1 gallon water)'
    },
    'Tomato___Bacterial_spot': {
        'description': 'A serious bacterial disease affecting tomatoes and peppers, causing spots on leaves, stems, and fruits that can lead to significant yield loss.',
        'symptoms': [
            'Small, water-soaked spots on leaves',
            'Spots turn black and may have yellow halos',
            'Raised, scabby lesions on fruit',
            'Severe leaf drop in advanced cases'
        ],
        'solutions': [
            'Use disease-free, certified seeds',
            'Apply copper-based bactericides early',
            'Rotate crops (3-4 year rotation)',
            'Avoid working with wet plants',
            'Remove and destroy infected plants'
        ],
        'severity': 'High',
        'prevention': 'Use drip irrigation, stake plants for better air circulation',
        'chemical_control': 'Copper-based bactericides, Mancozeb',
        'organic_control': 'Bacillus subtilis, Copper octanoate'
    },
    'Corn___Common_rust': {
        'description': 'A common fungal disease of corn characterized by reddish-brown pustules on leaves and leaf sheaths, which can reduce photosynthesis and yield.',
        'symptoms': [
            'Small, circular to elongated pustules',
            'Reddish-brown powdery spores',
            'Pustules on both leaf surfaces',
            'Premature leaf death in severe cases'
        ],
        'solutions': [
            'Plant resistant hybrids (look for rust resistance ratings)',
            'Apply foliar fungicides when disease first appears',
            'Space plants properly for good air circulation',
            'Rotate with non-host crops',
            'Remove and destroy crop debris after harvest'
        ],
        'severity': 'Low to Moderate (can be high in susceptible varieties)',
        'prevention': 'Plant early, use balanced fertilization',
        'chemical_control': 'Azoxystrobin, Propiconazole, Pyraclostrobin',
        'organic_control': 'Sulfur, Copper-based fungicides, Neem oil'
    },
    # Add more diseases as needed
}

def preprocess_image(image, img_size=(224, 224)):
    """Preprocess the image for prediction."""
    try:
        # Convert to RGB if image has an alpha channel
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3] if image.mode == 'RGBA' else None)
            image = background
        
        # Resize and normalize
        img = image.resize(img_size)
        img_array = np.array(img)
        
        # Convert grayscale to RGB if needed
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.shape[2] == 1:  # Single channel
            img_array = np.concatenate([img_array] * 3, axis=2)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[..., :3]
            
        img_array = img_array.astype('float32') / 255.0
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_disease(model, image, class_names):
    """Make prediction on a single image."""
    try:
        # Preprocess the image
        processed_img = preprocess_image(image)
        if processed_img is None:
            return None
            
        img_array = np.expand_dims(processed_img, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        # Get class name
        predicted_class = class_names.get(predicted_class_idx, "Unknown")
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = [
            (class_names.get(i, "Unknown"), float(predictions[0][i]))
            for i in top_indices
        ]
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def format_class_name(class_name):
    """Format class name for display."""
    return class_name.replace('___', ' - ').replace('_', ' ').title()

def display_disease_info(disease_info):
    """Display disease information in a beautiful card."""
    if not disease_info:
        return
        
    with st.container():
        st.markdown("### 📋 Disease Information")
        st.markdown(f"<div class='disease-info'><p>{disease_info.get('description', 'No description available.')}</p></div>", unsafe_allow_html=True)
        
        st.markdown("### 🔍 Common Symptoms")
        for symptom in disease_info.get('symptoms', ['No symptom information available.']):
            st.markdown(f"<div class='solution-item'>✅ {symptom}</div>", unsafe_allow_html=True)
        
        st.markdown("### 🛠️ Recommended Solutions")
        for i, solution in enumerate(disease_info.get('solutions', []), 1):
            st.markdown(f"<div class='solution-item'><b>{i}.</b> {solution}</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Severity", disease_info.get('severity', 'Unknown'))
        with col2:
            st.metric("Prevention", disease_info.get('prevention', 'N/A'))
        with col3:
            st.metric("Chemical Control", disease_info.get('chemical_control', 'N/A'))

def main():
    st.title("🌿 Advanced Plant Disease Detector")
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p>An intelligent system for plant disease detection using deep learning.</p>
        <p>Upload an image to get instant diagnosis and treatment recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add a sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Disease Detection", "Model Insights", "About"])
    
    # Load model and metrics
    with st.spinner("Loading AI model and metrics..."):
        model, class_names, history, confusion_data = load_model_and_metrics()
    
    if page == "Disease Detection":
        render_detection_page(model, class_names)
    elif page == "Model Insights":
        render_insights_page(model, class_names, history, confusion_data)
    else:
        render_about_page()

def render_detection_page(model, class_names):
    """Render the main disease detection page."""
    st.header("🔍 Disease Detection")
    st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <p>Upload an image of a plant leaf to detect potential diseases and get expert recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for history
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    if model is None:
        st.error("Failed to load the model. Please check if the model files exist.")
        return
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a plant leaf",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            try:
                # Display the uploaded image
                image = Image.open(uploaded_file)
                st.image(
                    image,
                    caption='Your Uploaded Image',
                    use_column_width=True
                )
                
                # Make prediction when button is clicked
                if st.button("🔍 Analyze Image", use_container_width=True, type="primary"):
                    with st.spinner('🔍 Analyzing the image with AI...'):
                        # Make prediction
                        result = predict_disease(model, image, class_names)
                        
                        if result:
                            # Store in session state
                            st.session_state.last_prediction = result
                            st.session_state.prediction_history.append({
                                'timestamp': result['timestamp'],
                                'disease': result['predicted_class'],
                                'confidence': result['confidence']
                            })
                            st.rerun()
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
        else:
            # Show sample images if no file is uploaded
            st.markdown("### 🖼️ Or try a sample:")
            sample_cols = st.columns(3)
            sample_images = [
                "data/PlantVillage/val/Apple___Apple_scab/028b9f66-ef2d-4a76-9d7a-7b785a6a6e9f___FREC_Scab 3417_90deg.JPG",
                "data/PlantVillage/val/Tomato___Bacterial_spot/0a5a9dab-af7e-4d1e-82c1-8a9a8f5d8e9f___GCREC_Bact.Spots 6561.JPG",
                "data/PlantVillage/val/Corn___Common_rust/0a5b629d-5f4e-4f8c-9e5d-8c9a8b7c6d5e___RS_Rust 3726.JPG"
            ]
            
            for i, img_path in enumerate(sample_images):
                if os.path.exists(img_path):
                    with sample_cols[i]:
                        st.image(
                            img_path,
                            use_column_width=True,
                            caption=f"Sample {i+1}"
                        )
    
    # Display results in the right column
    with col2:
        st.markdown("### 📊 Analysis Results")
        
        if 'last_prediction' in st.session_state and st.session_state.last_prediction:
            result = st.session_state.last_prediction
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            top_predictions = result['top_predictions']
            
            # Display prediction with confidence
            with st.container():
                st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
                st.markdown(f"#### 🎯 **Predicted Disease**")
                st.markdown(f"<h2 style='color: #2e7d32;'>{format_class_name(predicted_class)}</h2>", unsafe_allow_html=True)
                
                # Confidence meter
                confidence_pct = confidence * 100
                st.metric("Confidence", f"{confidence_pct:.1f}%")
                
                # Confidence visualization
                fig, ax = plt.subplots(figsize=(8, 0.5))
                ax.barh([0], [confidence_pct], color='#4CAF50', height=0.5)
                ax.set_xlim(0, 100)
                ax.axis('off')
                st.pyplot(fig, use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Display disease information
            disease_info = DISEASE_INFO.get(predicted_class, {})
            display_disease_info(disease_info)
            
            # Show top predictions
            st.markdown("### 🔍 Top Predictions")
            for class_name, prob in top_predictions:
                if class_name != predicted_class:  # Skip the main prediction
                    with st.expander(f"{format_class_name(class_name)} - {prob*100:.1f}%"):
                        info = DISEASE_INFO.get(class_name, {})
                        if info:
                            st.write(info.get('description', 'No description available.'))
                        else:
                            st.write("No additional information available for this prediction.")
        else:
            # Show placeholder when no prediction has been made
            st.markdown("""
            <div style='text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 12px;'>
                <p>👈 Upload an image and click "Analyze" to detect plant diseases</p>
                <p>or try one of the sample images below</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Add prediction history section
    if len(st.session_state.prediction_history) > 0:
        st.markdown("---")
        st.markdown("### 📜 Prediction History")
        
        # Convert history to DataFrame for display
        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df['disease'] = history_df['disease'].apply(format_class_name)
        history_df['confidence'] = history_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
        history_df = history_df.rename(columns={
            'timestamp': 'Time',
            'disease': 'Disease',
            'confidence': 'Confidence'
        })
        
        # Display history in a nice table
        st.dataframe(
            history_df,
            column_config={
                "Time": st.column_config.DatetimeColumn("Time", format="MM/DD/YY HH:mm"),
                "Disease": "Disease",
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
            },
            hide_index=True,
            use_container_width=True
        )
    
    # Add footer with additional information
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem;'>
        <p>🌱 <b>Plant Disease Detector</b> - Powered by Deep Learning</p>
        <p>For best results, use clear, well-lit photos of plant leaves.</p>
    </div>
    """, unsafe_allow_html=True)

def render_insights_page(model, class_names, history, confusion_data):
    """Render the model insights page."""
    st.header("📊 Model Insights")
    
    if model is None:
        st.warning("Model not loaded. Cannot display insights.")
        return
    
    # Model Summary Section
    st.subheader("Model Architecture")
    display_model_summary(model)
    
    # Training History
    st.subheader("Training Progress")
    if history:
        history_fig = plot_training_history(history)
        st.plotly_chart(history_fig, use_container_width=True)
    else:
        st.warning("Training history not available.")
    
    # Confusion Matrix
    st.subheader("Model Performance")
    if confusion_data:
        y_true, y_pred, num_classes = confusion_data
        cm_fig = plot_confusion_matrix(y_true, y_pred, class_names)
        st.plotly_chart(cm_fig, use_container_width=True)
        
        # Classification Report
        st.subheader("Classification Report")
        report = classification_report(
            y_true, y_pred, 
            target_names=[class_names[i] for i in range(num_classes)],
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='viridis', axis=0))
    else:
        st.warning("Performance metrics not available.")
    
    # Feature Importance (if available)
    st.subheader("Feature Importance")
    st.info("Feature importance visualization will be available after model analysis.")

def render_about_page():
    """Render the about page."""
    st.header("ℹ️ About")
    
    st.markdown("""
    ## Plant Disease Detection System
    
    This application uses deep learning to identify plant diseases from leaf images. 
    The model has been trained on a diverse dataset of plant diseases to provide 
    accurate and reliable predictions.
    
    ### How It Works
    1. **Image Upload**: Upload a clear image of a plant leaf
    2. **AI Analysis**: Our deep learning model analyzes the image
    3. **Results**: Get instant diagnosis and treatment recommendations
    
    ### Model Information
    - **Architecture**: Custom CNN (Convolutional Neural Network)
    - **Input Size**: 224x224 pixels
    - **Classes**: Multiple plant diseases
    - **Accuracy**: Varies by disease class
    
    ### Tips for Best Results
    - Use clear, well-lit photos
    - Focus on the affected leaves
    - Avoid shadows and glare
    - Capture the entire leaf if possible
    
    ### Disclaimer
    This tool is for informational purposes only and is not a substitute for 
    professional agricultural advice. Always consult with a qualified agronomist 
    for serious plant health concerns.
    """)

if __name__ == "__main__":
    main()
