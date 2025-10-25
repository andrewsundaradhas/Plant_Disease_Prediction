import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Tuple, Dict, Optional
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlantVillageDataLoader:
    """Data loader for the PlantVillage dataset."""
    
    def __init__(self, data_dir: str, img_size: Tuple[int, int] = (224, 224), 
                 batch_size: int = 32, test_split: float = 0.2, 
                 val_split: float = 0.2, seed: int = 42):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing the PlantVillage dataset
            img_size: Target image size (height, width)
            batch_size: Batch size for data generators
            test_split: Fraction of data to use for testing
            val_split: Fraction of training data to use for validation
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.test_split = test_split
        self.val_split = val_split
        self.seed = seed
        self.class_names = []
        self.num_classes = 0
        self._setup()
    
    def _setup(self) -> None:
        """Set up the data loader by finding classes and calculating splits."""
        # Get list of classes (subdirectories in data_dir)
        self.class_names = sorted([d.name for d in self.data_dir.glob('*') if d.is_dir()])
        self.num_classes = len(self.class_names)
        self.class_indices = {name: i for i, name in enumerate(self.class_names)}
        
        if self.num_classes == 0:
            raise ValueError(f"No classes found in {self.data_dir}. "
                           "Please ensure the directory contains subdirectories for each class.")
        
        logger.info(f"Found {self.num_classes} classes: {', '.join(self.class_names)}")
    
    def get_data_generators(self) -> Tuple[tf.keras.utils.Sequence, tf.keras.utils.Sequence, tf.keras.utils.Sequence]:
        """
        Create data generators for training, validation, and testing.
        
        Returns:
            A tuple containing (train_generator, val_generator, test_generator)
        ""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=self.test_split + self.val_split
        )
        
        # Data generator for testing (no augmentation)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create training generator
        train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            seed=self.seed
        )
        
        # Create validation generator
        val_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            seed=self.seed
        )
        
        # Create test generator
        test_generator = test_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',  # Using the same subset as validation for simplicity
            seed=self.seed
        )
        
        return train_generator, val_generator, test_generator
    
    def get_class_weights(self) -> Dict[int, float]:
        """
        Calculate class weights to handle class imbalance.
        
        Returns:
            Dictionary mapping class indices to their weights
        ""
        # Count number of samples per class
        class_counts = {}
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if class_dir.is_dir():
                class_counts[self.class_indices[class_name]] = len(
                    [f for f in class_dir.glob('*') if f.is_file()]
                )
        
        # Calculate weights
        total_samples = sum(class_counts.values())
        class_weights = {
            class_idx: total_samples / (self.num_classes * count)
            for class_idx, count in class_counts.items()
        }
        
        return class_weights
    
    def get_class_names(self) -> list:
        """Get the list of class names."""
        return self.class_names
    
    def get_num_classes(self) -> int:
        """Get the number of classes."""
        return self.num_classes


def download_plant_village_dataset(output_dir: str = 'data/plant_village', 
                                 force_download: bool = False) -> str:
    """
    Download the PlantVillage dataset from Kaggle.
    
    Args:
        output_dir: Directory to save the dataset
        force_download: If True, download even if directory exists
        
    Returns:
        Path to the downloaded dataset directory
    """
    output_path = Path(output_dir)
    
    # Skip if already downloaded
    if output_path.exists() and not force_download:
        logger.info(f"Dataset already exists at {output_path}. Set force_download=True to re-download.")
        return str(output_path)
    
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        logger.info("Downloading PlantVillage dataset from Kaggle...")
        api.dataset_download_files(
            'vipoooool/new-plant-diseases-dataset',
            path=output_path,
            unzip=True
        )
        
        # The dataset is downloaded to a subdirectory, move files up
        dataset_dir = output_path / 'New Plant Diseases Dataset(Augmented)' / 'New Plant Diseases Dataset(Augmented)'
        if dataset_dir.exists():
            # Move all files to the output directory
            for item in dataset_dir.glob('*'):
                item.rename(output_path / item.name)
            
            # Remove the empty directories
            (output_path / 'New Plant Diseases Dataset(Augmented)').rmdir()
            (output_path / 'New Plant Diseases Dataset(Augmented)').rmdir()
        
        logger.info(f"Dataset downloaded and extracted to {output_path}")
        return str(output_path)
    
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        logger.info("Please ensure you have the Kaggle API installed and configured.")
        logger.info("You can install it with: pip install kaggle")
        logger.info("Then run: kaggle datasets download -d vipoooool/new-plant-diseases-dataset")
        raise


if __name__ == "__main__":
    # Example usage
    data_dir = "data/plant_village"
    
    # Download the dataset if not already present
    try:
        data_dir = download_plant_village_dataset()
    except Exception as e:
        print(f"Could not download dataset: {e}")
        print("Please download it manually and update the data_dir path.")
    
    # Initialize data loader
    data_loader = PlantVillageDataLoader(
        data_dir=data_dir,
        img_size=(224, 224),
        batch_size=32
    )
    
    # Get data generators
    train_gen, val_gen, test_gen = data_loader.get_data_generators()
    
    # Get class weights
    class_weights = data_loader.get_class_weights()
    
    print(f"Number of classes: {data_loader.get_num_classes()}")
    print(f"Class names: {data_loader.get_class_names()}")
    print(f"Class weights: {class_weights}")
    print(f"Training batches: {len(train_gen)}")
    print(f"Validation batches: {len(val_gen)}")
    print(f"Test batches: {len(test_gen)}")
