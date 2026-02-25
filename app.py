import torch
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from torchvision import transforms

from model import Cifar_Net   # your CIFAR model architecture

from mnist_model import Mnist_Net  # your MNIST model architecture

class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

mnist_class_names = [
    "ZERO", "ONE", "TWO", "THREE", "FOUR",
    "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"
]
# -----------------------------
# Flask Setup
# -----------------------------
app = Flask(__name__)
CORS(app)

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
# API Endpoint
# -----------------------------
@app.route("/upload-page", methods=["POST"])
def upload_page():

    if "image_uploads" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image_uploads"]

    image = Image.open(file).convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = cifar_model(tensor)  
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = torch.max(probabilities).item()
        predicted_class = class_names[prediction]


    return jsonify({
        "prediction": predicted_class,
        "confidence": float(confidence)
    })

@app.route("/upload-page-drawing", methods=["POST"])
def upload_page_drawing():
    file = request.files["image_uploads"]

    image = Image.open(file).convert("L")

    tensor = transform_MNIST(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = mnist_model(tensor)  
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = torch.max(probabilities).item()
        predicted_class = mnist_class_names[prediction]


    return jsonify({
        "prediction": predicted_class,
        "confidence": float(confidence)
    })
    
# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )