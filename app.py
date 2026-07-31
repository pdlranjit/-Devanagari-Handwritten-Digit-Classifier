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


def preprocess_image(image):
    """
    Converts an arbitrary uploaded photo (dark digit on light background,
    off-center, with margins/noise) into the same style as the training
    dataset: white digit on black background, tightly cropped, centered,
    with a small padding margin.
    """
    # Convert to grayscale numpy array
    img = image.convert("L")
    img_array = np.array(img).astype(np.float32)

    # Auto-invert if background is light (dataset expects white digit on black bg)
    if img_array.mean() > 127:
        img_array = 255 - img_array

    # Threshold to remove faint noise/shadows
    img_array = np.where(img_array > 60, img_array, 0)
    img_array = img_array.astype(np.uint8)

    # Find bounding box of the digit (non-zero pixels)
    coords = np.column_stack(np.where(img_array > 20))
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        img_array = img_array[y0:y1 + 1, x0:x1 + 1]

    # Pad to square
    h, w = img_array.shape
    size = max(h, w)
    padded = np.zeros((size, size), dtype=np.uint8)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    padded[y_off:y_off + h, x_off:x_off + w] = img_array

    # Add margin so digit doesn't touch the edges (like typical dataset samples)
    margin = max(size // 5, 4)
    final_size = size + 2 * margin
    final_img = np.zeros((final_size, final_size), dtype=np.uint8)
    final_img[margin:margin + size, margin:margin + size] = padded

    return Image.fromarray(final_img)


st.title("Devanagari Handwritten Digit Classifier")
st.write("Upload a handwritten Devanagari numeral image (0-9) and the model will predict the digit.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=200)

    # Preprocess to match training data style (inverted, cropped, centered)
    processed = preprocess_image(image)
    st.image(processed, caption="Preprocessed (model input)", width=150)

    img_tensor = transform(processed).unsqueeze(0).to(device)

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