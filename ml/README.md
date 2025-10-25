# Crop Health Prediction - ML Pipeline

This directory contains the machine learning pipeline for the Crop Health Prediction System. The pipeline includes data downloading, preprocessing, model training, evaluation, and inference components.

## Project Structure

```
ml/
├── data/                  # Data directory
│   ├── raw/               # Raw dataset
│   ├── processed/         # Processed dataset (train/val/test splits)
│   └── samples/           # Sample images for testing
├── models/                # Saved models
├── results/               # Evaluation results and visualizations
├── data_downloader.py     # Script to download the dataset
├── preprocessor.py        # Data preprocessing and splitting
├── model_training.py      # Model training script
├── model_evaluation.py    # Model evaluation and metrics
├── inference.py           # Inference script for making predictions
├── generate_class_mapping.py  # Generate class mapping from dataset
└── requirements.txt       # Python dependencies
```

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up Kaggle API (for data downloading):
   - Get your Kaggle API key from [Kaggle Account Settings](https://www.kaggle.com/account)
   - Create a `kaggle.json` file in `~/.kaggle/` with your API credentials
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## Usage

### 1. Download the Dataset

```bash
python data_downloader.py
```

This will download the PlantVillage dataset to `data/raw/`.

### 2. Preprocess the Data

```bash
python preprocessor.py
```

This will:
- Process and resize images
- Split data into train/validation/test sets
- Save processed data to `data/processed/`

### 3. Generate Class Mapping

```bash
python generate_class_mapping.py
```

This creates a `class_mapping.json` file that maps class indices to class names.

### 4. Train the Model

```bash
python model_training.py
```

This will:
- Load the preprocessed data
- Train a ResNet50 model with transfer learning
- Save the trained model to `models/crop_health_model/`

### 5. Evaluate the Model

```bash
python model_evaluation.py
```

This will:
- Evaluate the model on the test set
- Generate a classification report
- Save a confusion matrix to `results/confusion_matrix.png`

### 6. Make Predictions

To make predictions on a single image:

```bash
python inference.py path/to/your/image.jpg
```

For more options:
```bash
python inference.py --help
```

## Customization

### Model Architecture

You can modify the model architecture in `model_training.py`. The default uses ResNet50 with a custom head for transfer learning.

### Training Parameters

Adjust hyperparameters like batch size, learning rate, and number of epochs in `model_training.py`.

### Data Augmentation

Data augmentation is applied during training. You can modify the augmentation settings in `model_training.py`.

## Troubleshooting

- **CUDA Out of Memory**: Reduce the batch size in `model_training.py`
- **Kaggle API Errors**: Verify your `kaggle.json` file and API key
- **Import Errors**: Make sure all dependencies are installed from `requirements.txt`

## Next Steps

- Deploy the model using the FastAPI backend
- Create a web interface using React
- Set up AWS infrastructure with CDK
