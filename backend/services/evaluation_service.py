"""
Model evaluation service for evaluating model performance.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationService:
    """Service for evaluating model performance."""
    
    def __init__(self, model_service, output_dir: str = "evaluation_results"):
        """Initialize the evaluation service.
        
        Args:
            model_service: Instance of the model service
            output_dir: Directory to save evaluation results
        """
        self.model_service = model_service
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def evaluate_on_test_set(
        self, 
        test_data_dir: str,
        batch_size: int = 32,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Evaluate the model on a test set.
        
        Args:
            test_data_dir: Directory containing test data (organized by class)
            batch_size: Batch size for evaluation
            save_results: Whether to save the evaluation results
            
        Returns:
            Dictionary containing evaluation metrics and plots
        """
        try:
            logger.info("Starting model evaluation on test set")
            
            # Load test data
            test_ds, class_names = self._load_test_data(test_data_dir)
            
            # Make predictions
            logger.info("Making predictions on test set")
            y_true = []
            y_pred = []
            y_scores = []
            
            for images, labels in test_ds:
                # Get predictions
                predictions = self.model_service.model.predict(images)
                
                # Store true and predicted labels
                y_true.extend(labels.numpy())
                y_pred.extend(np.argmax(predictions, axis=1))
                y_scores.extend(predictions)
            
            # Convert to numpy arrays
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_scores = np.array(y_scores)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_true, y_pred, y_scores, class_names)
            
            # Generate plots
            plots = self._generate_plots(y_true, y_pred, y_scores, class_names)
            
            # Save results if requested
            if save_results:
                self._save_results(metrics, plots, class_names)
            
            return {
                'metrics': metrics,
                'plots': plots,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise
    
    def _load_test_data(self, test_data_dir: str) -> Tuple[Any, List[str]]:
        """Load test data from directory.
        
        Args:
            test_data_dir: Directory containing test data
            
        Returns:
            Tuple of (test_dataset, class_names)
        """
        try:
            # Use the same preprocessing as during training
            img_height, img_width = self.model_service.input_shape
            
            test_ds = tf.keras.preprocessing.image_dataset_from_directory(
                test_data_dir,
                image_size=(img_height, img_width),
                batch_size=32,
                shuffle=False
            )
            
            # Get class names
            class_names = test_ds.class_names
            
            # Normalize pixel values
            normalization_layer = tf.keras.layers.Rescaling(1./255)
            test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
            
            return test_ds, class_names
            
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            raise
    
    def _calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_scores: np.ndarray,
        class_names: List[str]
    ) -> Dict[str, Any]:
        """Calculate evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Prediction scores/probabilities
            class_names: List of class names
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, confusion_matrix, classification_report,
            roc_auc_score
        )
        
        # Calculate overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Calculate per-class metrics
        class_report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate ROC AUC (one-vs-rest)
        try:
            if len(class_names) == 2:
                roc_auc = roc_auc_score(y_true, y_scores[:, 1])
            else:
                roc_auc = roc_auc_score(y_true, y_scores, multi_class='ovr')
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {str(e)}")
            roc_auc = None
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'class_report': class_report,
            'confusion_matrix': cm.tolist()
        }
    
    def _generate_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        class_names: List[str]
    ) -> Dict[str, str]:
        """Generate evaluation plots.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Prediction scores/probabilities
            class_names: List of class names
            
        Returns:
            Dictionary of base64-encoded plot images
        """
        plots = {}
        
        # Generate confusion matrix plot
        plots['confusion_matrix'] = self._plot_confusion_matrix(
            y_true, y_pred, class_names
        )
        
        # Generate ROC curve (for binary classification)
        if len(class_names) == 2:
            plots['roc_curve'] = self._plot_roc_curve(y_true, y_scores[:, 1])
        
        # Generate precision-recall curve
        plots['pr_curve'] = self._plot_precision_recall_curve(y_true, y_scores, class_names)
        
        return plots
    
    def _plot_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        class_names: List[str]
    ) -> str:
        """Generate and save confusion matrix plot."""
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create plot
        plt.figure(figsize=(10, 8))
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
        
        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        
        # Convert to base64
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    def _plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray) -> str:
        """Generate and save ROC curve plot (for binary classification)."""
        from sklearn.metrics import roc_curve, auc
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Create plot
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        
        # Convert to base64
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    def _plot_precision_recall_curve(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray,
        class_names: List[str]
    ) -> str:
        """Generate and save precision-recall curve."""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        # Calculate precision-recall curve
        if len(class_names) == 2:
            # Binary classification
            precision, recall, _ = precision_recall_curve(y_true, y_scores[:, 1])
            avg_precision = average_precision_score(y_true, y_scores[:, 1])
            
            # Create plot
            plt.figure()
            plt.plot(recall, precision, lw=2, 
                    label=f'Precision-Recall (AP = {avg_precision:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
        else:
            # Multi-class classification
            from sklearn.preprocessing import label_binarize
            
            # Binarize the output
            y_test_bin = label_binarize(y_true, classes=range(len(class_names)))
            n_classes = y_test_bin.shape[1]
            
            # Compute PR curve and PR AUC for each class
            plt.figure(figsize=(10, 8))
            
            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(
                    y_test_bin[:, i], y_scores[:, i])
                avg_precision = average_precision_score(
                    y_test_bin[:, i], y_scores[:, i])
                
                plt.plot(
                    recall, 
                    precision, 
                    lw=2,
                    label=f'{class_names[i]} (AP = {avg_precision:.2f})'
                )
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('Precision-Recall Curve (One-vs-Rest)')
            plt.legend(loc="best")
        
        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        
        # Convert to base64
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    def _save_results(
        self, 
        metrics: Dict[str, Any], 
        plots: Dict[str, str],
        class_names: List[str]
    ) -> str:
        """Save evaluation results to disk.
        
        Args:
            metrics: Dictionary of evaluation metrics
            plots: Dictionary of base64-encoded plot images
            class_names: List of class names
            
        Returns:
            Path to the saved results directory
        """
        try:
            # Create timestamped directory
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_dir = self.output_dir / f"evaluation_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metrics as JSON
            metrics_file = output_dir / "metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save plots as images
            for plot_name, plot_data in plots.items():
                plot_file = output_dir / f"{plot_name}.png"
                with open(plot_file, 'wb') as f:
                    f.write(base64.b64decode(plot_data))
            
            # Save class mapping
            class_mapping = {name: i for i, name in enumerate(class_names)}
            with open(output_dir / "class_mapping.json", 'w') as f:
                json.dump(class_mapping, f, indent=2)
            
            logger.info(f"Evaluation results saved to {output_dir}")
            return str(output_dir)
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {str(e)}")
            raise

# Global evaluation service instance
_evaluation_service = None

def get_evaluation_service() -> EvaluationService:
    """Get the global evaluation service instance (singleton pattern)."""
    global _evaluation_service
    if _evaluation_service is None:
        from ..services.model_service import get_model
        _evaluation_service = EvaluationService(get_model())
    return _evaluation_service
