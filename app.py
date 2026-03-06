import torch
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
import io

from model import Cifar_Net   # my CIFAR model architecture

from mnist_model import Mnist_Net  # my MNIST model architecture

class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

mnist_class_names = [
    "ZERO", "ONE", "TWO", "THREE", "FOUR",
    "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"
]
# -----------------------------
# FastAPI Setup
# -----------------------------
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load Model
# -----------------------------
device = torch.device("cpu")

cifar_model = Cifar_Net().to(device)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pth")

cifar_model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

cifar_model.eval()

# -----------------------------
# Load MNIST model
# -----------------------------

mnist_model = Mnist_Net().to(device)

MNIST_MODEL_PATH = os.path.join(os.path.dirname(__file__), "mnist_model.pth")

mnist_model.load_state_dict(
    torch.load(MNIST_MODEL_PATH, map_location=device)
)

mnist_model.eval()
# -----------------------------
# Preprocessing (CIFAR10)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])
# -----------------------------
# Preprocessing (MNIST)
# -----------------------------
transform_MNIST = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
# -----------------------------
# API Endpoints
# -----------------------------
@app.post("/upload-page")
async def upload_page(image_uploads: UploadFile = File(...)):
    try:
        contents = await image_uploads.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = cifar_model(tensor)  
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item()
            predicted_class = class_names[prediction]

        return JSONResponse({
            "prediction": predicted_class,
            "confidence": float(confidence)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/upload-page-drawing")
async def upload_page_drawing(image_uploads: UploadFile = File(...)):
    try:
        contents = await image_uploads.read()
        image = Image.open(io.BytesIO(contents)).convert("L")

        tensor = transform_MNIST(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = mnist_model(tensor)  
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item()
            predicted_class = mnist_class_names[prediction]

        return JSONResponse({
            "prediction": predicted_class,
            "confidence": float(confidence)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)