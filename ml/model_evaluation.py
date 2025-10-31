import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_recall_fscore_support
)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_data(test_dir: str, img_size: Tuple[int, int] = (224, 224), batch_size: int = 32) -> Tuple[tf.keras.preprocessing.image.DirectoryIterator, Dict[int, str]]:
    """
    Load and preprocess test data from directory.
    
    Args:
        test_dir: Directory containing test images organized in subdirectories by class
        img_size: Target image size (height, width)
        batch_size: Batch size for data generator
        
    Returns:
        test_generator: Data generator for test set
        class_indices: Mapping from class indices to class names
    """
    # Create data generator for test data (only rescaling, no augmentation)
    test_datagen = ImageDataGenerator(
        rescale=1./255
    )
    
    # Load test data
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # Important for consistent evaluation
    )
    
    # Get class indices (numeric to class name mapping)
    class_indices = {v: k for k, v in test_generator.class_indices.items()}
    
    return test_generator, class_indices

def evaluate_model(
    model_path: str, 
    test_dir: str, 
    batch_size: int = 32,
    img_size: Tuple[int, int] = (224, 224)
) -> Dict[str, Any]:
    """
    Evaluate the trained model on the test set and generate comprehensive performance metrics.
    
    Args:
        model_path: Path to the saved model
        test_dir: Directory containing test images organized in subdirectories by class
        batch_size: Batch size for evaluation
        img_size: Target image size (height, width)
        
    Returns:
        Dictionary containing evaluation metrics and results
    """
    logger.info("🔍 Loading model for evaluation...")
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"✅ Successfully loaded model from {model_path}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise
        
    # Load and preprocess test data
    logger.info("📂 Loading test data...")
    test_generator, class_indices = load_test_data(test_dir, img_size, batch_size)
    num_classes = len(class_indices)
    
    # Get true labels and predictions
    logger.info("🔮 Running predictions on test set...")
    y_true = test_generator.classes
    y_pred_proba = model.predict(test_generator, steps=len(test_generator), verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    logger.info("📊 Calculating evaluation metrics...")
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Generate classification report
    class_report = classification_report(
        y_true, 
        y_pred, 
        target_names=list(class_indices.values()),
        output_dict=True
    )
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate per-class metrics
    class_metrics = {}
    for i, class_name in class_indices.items():
        class_metrics[class_name] = {
            'precision': class_report[class_name]['precision'],
            'recall': class_report[class_name]['recall'],
            'f1_score': class_report[class_name]['f1-score'],
            'support': class_report[class_name]['support']
        }
    
    # Generate visualization paths
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    cm_plot_path = f"confusion_matrix_{timestamp}.png"
    
    # Save confusion matrix plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        cmap='Blues',
        xticklabels=class_indices.values(),
        yticklabels=class_indices.values()
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(cm_plot_path)
    plt.close()
    
    # Prepare results
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'class_metrics': class_metrics,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_plot': cm_plot_path,
        'num_samples': len(y_true),
        'num_classes': num_classes,
        'class_indices': class_indices,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    logger.info("✅ Evaluation completed successfully")
    return results
    
    # Create test data generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Get class names
    class_names = list(test_generator.class_indices.keys())
    
    # Evaluate model
    print("📊 Evaluating model on test set...")
    loss, accuracy = model.evaluate(test_generator)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    print(f"Test Loss: {loss:.4f}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes
    
    # Generate classification report
    print("\n📝 Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    
    # Generate and plot confusion matrix
    print("\n📈 Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred_classes)
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save the confusion matrix
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    print("✅ Evaluation results saved to 'results/' directory")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate crop health prediction model')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the saved model directory')
    parser.add_argument('--test_dir', type=str, required=True,
                       help='Directory containing test images organized by class')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--output_file', type=str, default='evaluation_results.json',
                       help='Path to save evaluation results (JSON)')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_model(
        model_path=args.model_path,
        test_dir=args.test_dir,
        batch_size=args.batch_size
    )
    
    # Save results to file
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Evaluation completed! Results saved to {args.output_file}")
    print(f"\n📋 Summary:")
    print(f"- Accuracy: {results['accuracy']:.4f}")
    print(f"- Precision: {results['precision']:.4f}")
    print(f"- Recall: {results['recall']:.4f}")
    print(f"- F1 Score: {results['f1_score']:.4f}")
    print(f"- # Classes: {results['num_classes']}")
    print(f"- # Samples: {results['num_samples']}")
    )
