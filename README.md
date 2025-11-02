# Crop Health Prediction System

A comprehensive system for predicting crop health using machine learning and deep learning techniques.

##  Features

- **Machine Learning Model**: Trained on plant disease datasets for accurate predictions
- **RESTful API**: FastAPI backend with comprehensive documentation
- **Modern Frontend**: React-based user interface with responsive design
- **Database**: MongoDB for data persistence
- **Scalable**: Designed to handle multiple concurrent requests
- **Easy Setup**: One-command setup and deployment

##  Prerequisites

- Python 3.8+
- Node.js 16+
- npm 8+
- MongoDB 5.0+
- Git

##  Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/crop-health-prediction.git
cd crop-health-prediction
```

### 2. Run Setup Script

```bash
python setup.py
```

This will:
- Create a Python virtual environment
- Install all required dependencies
- Set up the database
- Configure environment variables

### 3. Configure Environment Variables

Edit the `.env` file in the project root with your configuration:

```env
DEBUG=True
MONGODB_URI=mongodb://localhost:27017/crop_health
SECRET_KEY=your-secret-key
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1
AWS_S3_BUCKET=your-bucket-name
```

### 4. Start the Backend Server

```bash
cd backend
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

### 5. Start the Frontend Development Server

```bash
cd frontend
npm start
```

The application will open in your default browser at `http://localhost:3000`

##  Machine Learning Model

The machine learning model is located in the `ml/` directory. To train a new model:

```bash
cd ml
python train.py
```

### Model Architecture

- **Base Model**: EfficientNetB0 (pre-trained on ImageNet)
- **Input Size**: 224x224 pixels
- **Output**: Probability distribution over disease classes
- **Training**: Transfer learning with fine-tuning

##  API Documentation

Once the backend server is running, access the interactive API documentation at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

##  Deployment

### Backend (Production)

```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app
```

### Frontend (Production Build)

```bash
cd frontend
npm run build
```

##  Testing

### Backend Tests

```bash
cd backend
pytest
```

### Frontend Tests

```bash
cd frontend
npm test
```


