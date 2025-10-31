import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_training():
    """Run the complete training and evaluation pipeline."""
    try:
        from ml.train import train_model
        from ml.model_evaluation import evaluate_model
        
        # Configuration
        config = {
            'data_dir': 'data/plantvillage',
            'output_dir': 'output',
            'model_name': 'plant_disease_model',
            'base_model': 'EfficientNetB0',
            'img_size': (224, 224),
            'batch_size': 32,
            'epochs': 30,  # Reduced for testing, increase for better results
            'learning_rate': 1e-4,
            'freeze_base': True,
            'dropout_rate': 0.5,
            'dense_units': 1024,
            'test_split': 0.2,
            'val_split': 0.2,
            'seed': 42,
            'use_class_weights': True
        }
        
        # Create output directories
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # Run training
        logger.info("🚀 Starting model training...")
        history = train_model(**config)
        
        # Run evaluation
        logger.info("\n🔍 Running model evaluation...")
        model_path = output_dir / 'final_model.h5'
        test_dir = output_dir / 'test_data'  # Update this path if your test data is elsewhere
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Training may have failed.")
            
        if not test_dir.exists():
            logger.warning(f"Test directory not found at {test_dir}. Skipping evaluation.")
        else:
            eval_results = evaluate_model(
                model_path=str(model_path),
                test_dir=str(test_dir),
                batch_size=config['batch_size'],
                img_size=config['img_size']
            )
            
            # Save evaluation results
            eval_path = output_dir / 'evaluation_results.json'
            with open(eval_path, 'w') as f:
                import json
                json.dump(eval_results, f, indent=2)
            logger.info(f"✅ Evaluation results saved to {eval_path}")
            
    except Exception as e:
        logger.error(f"❌ Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    logger.info("🔄 Starting training pipeline...")
    run_training()
    logger.info("✨ Training pipeline completed successfully!")
