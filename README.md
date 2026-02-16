# CIFAR-10 Neural Network API

A Docker-containerized FastAPI backend for CIFAR-10 image classification with a web frontend.

## Overview

This project provides:
- **FastAPI Backend**: REST API for image classification
- **PyTorch Model**: Pre-trained CIFAR-10 neural network
- **Web Frontend**: Interactive canvas drawing interface
- **Docker Support**: Ready for containerization
- **Render Deployment**: Configuration for Render.com hosting

## Project Structure

```
.
├── app.py                 # FastAPI application
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── render.yaml            # Render.com deployment config
├── .dockerignore          # Files to exclude from Docker image
├── model.pth              # Trained PyTorch model (REQUIRED)
├── static/
│   └── index.html         # Web interface
└── trained_neural_CIFAR10.ipynb  # Model training notebook
```

## Getting Started

### Prerequisites
- Python 3.11+
- Docker (optional, for containerization)
- The trained model file `model.pth`

### Local Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Ensure model file exists**:
   - Place your `model.pth` file in the project root
   - If missing, the API will run with an untrained model

3. **Run the API**:
```bash
python app.py
```
   - API available at: `http://localhost:8000`
   - Web interface at: `http://localhost:8000`
   - API docs at: `http://localhost:8000/docs`

## API Endpoints

### `GET /`
Root endpoint with API information.

### `GET /health`
Health check endpoint.
```json
{"status": "healthy", "device": "cpu"}
```

### `POST /predict`
Predict class from an uploaded image file.

**Request**: Multipart form with image file
**Response**:
```json
{
  "predicted_class": "plane",
  "confidence": 0.95,
  "all_probabilities": {
    "plane": 0.95,
    "car": 0.03,
    "bird": 0.01,
    ...
  }
}
```

### `POST /predict-base64`
Predict class from base64-encoded image data (useful for canvas drawings).

**Request**:
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSU..."
}
```

**Response**: Same as `/predict`

## Supported Classes

The model recognizes these CIFAR-10 classes:
- plane
- car
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## Docker Usage

### Build Image
```bash
docker build -t cifar10-api .
```

### Run Container
```bash
docker run -p 8000:8000 cifar10-api
```

Visit `http://localhost:8000` in your browser.

## Deployment to Render.com

### Prerequisites
- Render.com account
- GitHub repository with this code
- `model.pth` file committed to the repo

### Deployment Steps

1. **Push to GitHub**:
```bash
git push origin main
```

2. **Create Web Service on Render**:
   - Go to https://dashboard.render.com
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: `cifar10-api`
     - **Runtime**: Python
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
     - **Environment**: Free tier is sufficient for testing

3. **Set Environment Variables** (if needed):
   - No special variables required for basic setup

4. **Deploy**:
   - Click "Create Web Service"
   - Wait for build to complete
   - Your API will be available at: `https://cifar10-api-[random].onrender.com`

### Update API URL in Frontend
If deploying to Render, update the `API_URL` in `static/index.html`:
```javascript
const API_URL = "https://cifar10-api-[random].onrender.com";
```

Or it will auto-detect if served from the same origin.

## Related Jupyter Notebook

The `trained_neural_CIFAR10.ipynb` notebook contains:
- Model architecture definition
- CIFAR-10 dataset loading
- Training code
- Evaluation code
- Custom prediction examples

## Features

✅ Responsive web interface  
✅ Real-time image classification  
✅ Confidence scores and probability distribution  
✅ Canvas drawing support  
✅ Mobile-friendly touch support  
✅ CORS enabled for cross-origin requests  
✅ GPU support (if available)  
✅ Docker ready  
✅ Render.com deployment ready  

## Development

### Adding GPU Support
In Render.com, you can add a GPU for faster inference:
- In Web Service settings
- Add GPU instance type

### Improving Model Accuracy
- Retrain using `trained_neural_CIFAR10.ipynb`
- Save new weights to `model.pth`
- Update code and push to trigger redeploy

## Troubleshooting

### Model not found
- Ensure `model.pth` is in the project root
- API will warn but continue to run with untrained model

### CORS errors
- API already has CORS enabled for all origins
- Check browser console for actual error

### Image not recognized
- Ensure image is resized to 32x32 (API does this automatically)
- Check image color space (API converts to RGB)

### Port already in use
```bash
python app.py  # Uses $PORT env var or default 8000
```

## License

MIT
