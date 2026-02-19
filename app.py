import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import io
from PIL import Image
import numpy as np
from torchvision import transforms
import os

app = FastAPI(title="CIFAR-10 Neural Network API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CIFAR-10 class names
CIFAR10_CLASSES = (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)

# Define the neural network (same as in notebook)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Load model
device = "cpu"
model = Net().to(device)

# Try to load trained weights
model_path = "model.pth"
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Warning: Could not load model weights: {e}")
else:
    print(f"Warning: {model_path} not found. Using untrained model.")

model.eval()

# Image transform (CIFAR-10 normalization)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


@app.get("/")
async def root():
    return {
        "message": "CIFAR-10 Neural Network API",
        "endpoints": {
            "predict": "/predict",
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "device": str(device)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict CIFAR-10 class from an uploaded image.
    Accepts JPG, PNG, etc.
    Returns class name and probabilities for all classes.
    """
    try:
        # Read uploaded file
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        
        # Resize to CIFAR-10 size (32x32)
        img = img.resize((32, 32))
        
        # Transform
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            predicted_class = output.argmax(dim=1).item()
            probabilities = F.softmax(output, dim=1)[0]
        
        # Prepare response
        class_probabilities = {
            CIFAR10_CLASSES[i]: float(probabilities[i].item())
            for i in range(len(CIFAR10_CLASSES))
        }
        
        return {
            "predicted_class": CIFAR10_CLASSES[predicted_class],
            "confidence": float(probabilities[predicted_class].item()),
            "all_probabilities": class_probabilities
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/predict-base64")
async def predict_base64(data: dict):
    """
    Predict from base64 encoded image data.
    Useful for canvas data from HTML frontend.
    """
    try:
        import base64
        
        # Extract base64 string (remove data:image/png;base64, prefix if present)
        img_data = data.get("image")
        if not img_data:
            raise ValueError("No image data provided")
        
        if img_data.startswith("data:"):
            img_data = img_data.split(",")[1]
        
        # Decode base64
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Resize to CIFAR-10 size (32x32)
        img = img.resize((32, 32))
        
        # Transform
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            predicted_class = output.argmax(dim=1).item()
            probabilities = F.softmax(output, dim=1)[0]
        
        # Prepare response
        class_probabilities = {
            CIFAR10_CLASSES[i]: float(probabilities[i].item())
            for i in range(len(CIFAR10_CLASSES))
        }
        
        return {
            "predicted_class": CIFAR10_CLASSES[predicted_class],
            "confidence": float(probabilities[predicted_class].item()),
            "all_probabilities": class_probabilities
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


# Serve static files (HTML, CSS, JS)
try:
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
except Exception as e:
    print(f"Note: Static files directory not found: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
