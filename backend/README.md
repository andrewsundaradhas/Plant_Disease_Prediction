# Crop Health Prediction - Backend API

This is the FastAPI backend for the Crop Health Prediction System. It provides a RESTful API for making predictions on crop health using the trained ML model.

## Features

- Upload images for crop health prediction
- Get prediction results with confidence scores
- List all available crop disease classes
- Health check endpoint
- CORS enabled for frontend integration
- Async file handling
- Input validation
- Error handling

## API Documentation

Once the server is running, visit `/docs` for interactive API documentation (Swagger UI) or `/redoc` for ReDoc documentation.

### Endpoints

- `GET /`: API information
- `GET /health`: Health check
- `POST /predict`: Upload an image for prediction
- `GET /classes`: List all available crop disease classes

## Setup

### Prerequisites

- Python 3.8+
- pip
- Docker (optional)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd aws-crop/backend
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Make sure you have the trained model and class mapping file in place:
   ```
   backend/
   ├── ml/
   │   ├── models/
   │   │   └── crop_health_model/  # Your trained model
   │   └── class_mapping.json      # Generated class mapping
   ```

### Running the Server

#### Development Mode

```bash
uvicorn main:app --reload
```

#### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

#### Using Docker

1. Build the Docker image:
   ```bash
   docker build -t crop-health-backend .
   ```

2. Run the container:
   ```bash
   docker run -d --name crop-health-backend -p 8000:8000 crop-health-backend
   ```

## Environment Variables

Create a `.env` file in the backend directory with the following variables:

```
# Server configuration
HOST=0.0.0.0
PORT=8000

# File uploads
UPLOAD_DIR=uploads
MAX_UPLOAD_SIZE=10485760  # 10MB

# Model paths
MODEL_PATH=ml/models/crop_health_model
CLASS_MAPPING_PATH=ml/class_mapping.json
```

## Testing

You can test the API using tools like `curl` or Postman.

### Example using curl:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/your/image.jpg;type=image/jpeg'
```

## Development

### Code Style

This project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting

### Pre-commit Hooks

Install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

### Testing

Run tests:

```bash
pytest
```

## Deployment

For production deployment, consider using:
- Gunicorn with Uvicorn workers
- Nginx as a reverse proxy
- Docker and Docker Compose
- Kubernetes (for scaling)

## License

[Your License Here]
