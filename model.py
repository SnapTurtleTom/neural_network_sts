# -----------------------------
# cnn_mnist_full.py
# -----------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# -----------------------------
# Define the neural network
# -----------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, 5)    # input: 1x28x28, output: 6x24x24
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)   # input: 6x12x12, output: 16x8x8

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # after 2 poolings
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)           # CIFAR classes

    def forward(self, x):
        # First conv layer + ReLU + 2x2 max pool
        x = self.pool(F.relu(self.conv1(x)))

        # Second conv layer + ReLU + 2x2 max pool
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten for fully connected layers
        x = torch.flatten(x, 1)

        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output layer (no activation here, CrossEntropyLoss applies softmax internally)
        x = self.fc3(x)
        return x

# -----------------------------
# Load the MNIST dataset
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# -----------------------------
# Setup device, model, loss, optimizer
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Test / evaluation
# -----------------------------
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()  # set model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # no gradients needed for evaluation
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# -----------------------------
# Predict a custom image
# -----------------------------
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")  # Red Green Blue
    img = img.resize((32, 32))                # resize to 32x32
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)  # add batch dim and send to device

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        predicted_digit = output.argmax(dim=1).item()
        probabilities = F.softmax(output, dim=1)
        percentages = probabilities * 100
    return predicted_digit, percentages

# Example usage
if __name__ == "__main__":
    custom_image_path = "lizard.jpg"
    predicted, percentages = predict_image(custom_image_path)
    print(f"Predicted: {predicted}")
    
    percentages_list = percentages.squeeze(0).tolist()  # remove batch dim
    for i, p in enumerate(percentages_list):
        print(f"{i}: {p:.2f}%")