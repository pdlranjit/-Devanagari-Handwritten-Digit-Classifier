import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np


class DevanagariCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2   = nn.Conv2d(32, 64, 3, padding=1)
        self.pool    = nn.MaxPool2d(2, 2)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.fc1     = nn.Linear(64 * 8 * 8, 128)
        self.fc2     = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DevanagariCNN().to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    return model, device

model, device = load_model()
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


st.title("Devanagari Handwritten Digit Classifier")
st.write("Upload a handwritten Devanagari numeral image (0-9) and the model will predict the digit.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=200)

  
    img_tensor = transform(image).unsqueeze(0).to(device)  # add batch dimension

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_digit = class_names[predicted.item()]
    confidence_pct = confidence.item() * 100

    st.success(f"Predicted Digit: **{predicted_digit}**")
    st.write(f"Confidence: {confidence_pct:.2f}%")

    # Show probability breakdown for all classes
    st.subheader("Prediction Probabilities")
    probs_dict = {class_names[i]: float(probabilities[0][i]) for i in range(10)}
    st.bar_chart(probs_dict)
else:
    st.info("Please upload an image to get a prediction.")