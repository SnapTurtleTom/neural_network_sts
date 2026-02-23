import torch
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from torchvision import transforms

from model import Net   # your model architecture

# -----------------------------
# Flask Setup
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# Load Model
# -----------------------------
device = torch.device("cpu")

model = Net().to(device)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pth")

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()

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
        output = model(tensor)  
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = torch.max(probabilities).item()
        

    return jsonify({
        "prediction": prediction,
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