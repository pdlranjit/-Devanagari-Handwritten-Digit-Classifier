import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter, ImageChops
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


def otsu_threshold(img_array):
    """Compute an adaptive binarization threshold (Otsu's method) without cv2."""
    hist, _ = np.histogram(img_array, bins=256, range=(0, 256))
    total = img_array.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b, w_b = 0.0, 0
    max_var, threshold = 0.0, 128
    for i in range(256):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = i
    return threshold


def preprocess_image(image):
    """
    Converts an arbitrary uploaded photo (dark digit on paper, with uneven
    lighting/glare/shadows, off-center) into the same style as the training
    dataset: pure white digit on pure black background, tightly cropped,
    centered, with a small padding margin.

    Key idea: instead of a single global brightness threshold (which gets
    fooled by glare or shadows on the paper), we estimate the local paper
    brightness with a heavy blur and subtract it out. What's left over is
    just the ink stroke, regardless of uneven lighting.
    """
    img = ImageOps.grayscale(image)

    # Estimate the smooth local "paper brightness" (includes glare/shadow)
    background = img.filter(ImageFilter.GaussianBlur(radius=21))

    # Subtract it out: result is bright only where the original was
    # noticeably darker than its local surroundings (i.e. the ink stroke)
    diff = ImageChops.subtract(background, img)
    diff_array = np.array(diff).astype(np.float32)

    # Rescale so the strongest stroke pixel hits 255 (boosts faint pencil marks)
    if diff_array.max() > 0:
        diff_array = diff_array * (255.0 / diff_array.max())
    diff_array = diff_array.astype(np.uint8)

    # Adaptive binarization on the cleaned residual -> pure black/white
    thresh = otsu_threshold(diff_array)
    img_array = np.where(diff_array > thresh, 255, 0).astype(np.uint8)

    # Find bounding box of the digit (white pixels only)
    coords = np.column_stack(np.where(img_array == 255))
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