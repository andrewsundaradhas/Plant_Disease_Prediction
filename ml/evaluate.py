import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc
)
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

# Import local modules
from .data_loader import PlantVillageDataLoader
from .model import load_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(
    model_path: str,
    data_dir: str,
    output_dir: str = 'evaluation',
    batch_size: int = 32,
    img_size: Tuple[int, int] = (224, 224),
    test_split: float = 0.2,
    val_split: float = 0.2,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluate a trained model on the test set.
    
    Args:
        model_path: Path to the trained model
        data_dir: Directory containing the test data
        output_dir: Directory to save evaluation results
        batch_size: Batch size for evaluation
        img_size: Image size (height, width)
        test_split: Fraction of data to use for testing
        val_split: Fraction of training data to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing evaluation metrics and plots
    """
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_dir = Path(output_dir) / f"evaluation_{timestamp}"
    plots_dir = eval_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Evaluation results will be saved to: {eval_dir}")
    
    # Load the model
    logger.info(f"Loading model from {model_path}...")
    try:
        model = load_model(model_path)
        model.summary()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    # Initialize data loader
    logger.info("Loading test data...")
    data_loader = PlantVillageDataLoader(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        test_split=test_split,
        val_split=val_split,
        seed=seed
    )
    
    # Get test data generator
    _, _, test_gen = data_loader.get_data_generators()
    class_names = data_loader.get_class_names()
    num_classes = len(class_names)
    
    # Evaluate the model
    logger.info("Evaluating model on test set...")
    evaluation = model.evaluate(test_gen, return_dict=True)
    
    # Generate predictions
    logger.info("Generating predictions...")
    y_pred_proba = model.predict(test_gen)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Get true labels
    y_true = test_gen.classes
    
    # Calculate additional metrics
    logger.info("Calculating metrics...")
    
    # Classification report
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    # Calculate ROC AUC (one-vs-rest)
    roc_auc = None
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    try:
        # Convert true labels to one-hot encoding
        y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)
        
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Calculate micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_onehot.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
    except Exception as e:
        logger.warning(f"Could not calculate ROC AUC: {e}")
    
    # Save metrics to a dictionary
    metrics = {
        'overall_metrics': evaluation,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1.tolist(),
            'support': support.tolist(),
            'roc_auc': {str(k): v for k, v in roc_auc.items()} if roc_auc else None
        },
        'class_names': class_names,
        'timestamp': timestamp,
        'model_path': model_path,
        'data_dir': data_dir,
        'test_size': len(y_true)
    }
    
    # Save metrics to JSON
    metrics_path = eval_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Generate and save plots
    logger.info("Generating plots...")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    cm_plot_path = plots_dir / 'confusion_matrix.png'
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curve (if available)
    if roc_auc:
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for i, class_name in enumerate(class_names):
            plt.plot(
                fpr[i], 
                tpr[i], 
                lw=2, 
                label=f'{class_name} (AUC = {roc_auc[i]:.2f})'
            )
        
        # Plot micro-average ROC curve
        plt.plot(
            fpr["micro"], 
            tpr["micro"],
            label=f'micro-average (AUC = {roc_auc["micro"]:.2f})',
            linestyle='--',
            color='deeppink'
        )
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        roc_plot_path = plots_dir / 'roc_curve.png'
        plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        metrics['roc_curve_plot'] = str(roc_plot_path)
    
    # 3. Class Distribution
    plt.figure(figsize=(12, 6))
    class_counts = np.bincount(y_true)
    sns.barplot(x=class_names, y=class_counts)
    plt.title('Class Distribution in Test Set')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    dist_plot_path = plots_dir / 'class_distribution.png'
    plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Per-class metrics
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support,
        'ROC AUC': [roc_auc.get(i, None) for i in range(num_classes)] if roc_auc else [None] * num_classes
    })
    
    metrics_csv_path = eval_dir / 'per_class_metrics.csv'
    metrics_df.to_csv(metrics_csv_path, index=False)
    
    # Save paths to plots in metrics
    metrics['plots'] = {
        'confusion_matrix': str(cm_plot_path),
        'class_distribution': str(dist_plot_path),
        'per_class_metrics': str(metrics_csv_path)
    }
    
    if roc_auc:
        metrics['plots']['roc_curve'] = str(roc_plot_path)
    
    # Save updated metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Evaluation completed. Results saved to {eval_dir}")
    
    return metrics

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate a plant disease classification model.')
    
    # Required arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing the test data')
    
    # Optional arguments
    parser.add_argument('--output-dir', type=str, default='evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--img-size', type=int, nargs=2, default=[224, 224],
                        metavar=('HEIGHT', 'WIDTH'),
                        help='Image size (height, width)')
    parser.add_argument('--test-split', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Fraction of training data to use for validation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('evaluation.log')
        ]
    )
    
    try:
        # Evaluate the model
        metrics = evaluate_model(
            model_path=args.model_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            img_size=tuple(args.img_size),
            test_split=args.test_split,
            val_split=args.val_split,
            seed=args.seed
        )
        
        # Print summary
        print("\n" + "="*50)
        print("Evaluation Summary")
        print("="*50)
        print(f"Model: {args.model_path}")
        print(f"Test accuracy: {metrics['overall_metrics']['accuracy']:.4f}")
        print(f"Test loss: {metrics['overall_metrics']['loss']:.4f}")
        
        if 'roc_auc' in metrics['per_class_metrics'] and metrics['per_class_metrics']['roc_auc']:
            print(f"Micro-average ROC AUC: {metrics['per_class_metrics']['roc_auc'].get('micro', 'N/A'):.4f}")
        
        print("\nPer-class metrics:")
        metrics_df = pd.DataFrame({
            'Class': metrics['class_names'],
            'Precision': metrics['per_class_metrics']['precision'],
            'Recall': metrics['per_class_metrics']['recall'],
            'F1-Score': metrics['per_class_metrics']['f1_score'],
            'Support': metrics['per_class_metrics']['support']
        })
        
        if 'roc_auc' in metrics['per_class_metrics'] and metrics['per_class_metrics']['roc_auc']:
            metrics_df['ROC AUC'] = [metrics['per_class_metrics']['roc_auc'].get(str(i), None) for i in range(len(metrics['class_names']))]
        
        print(metrics_df.to_string())
        print("\nPlots and detailed metrics saved to:")
        for name, path in metrics['plots'].items():
            print(f"- {name}: {path}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise
