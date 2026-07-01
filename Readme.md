# Devanagari Handwritten Digit Classifier

A Convolutional Neural Network (CNN) built with PyTorch to classify handwritten Devanagari numerals (०-९ / 0-9) from the NHCD (Numeral Handwritten Character Dataset). Trained and tested in Google Colab.

## 📊 Results

- **Best Test Accuracy: 97.92%**
- Dataset size: 2,880 images (10 classes, digits 0-9)
- Train/Test split: 2,448 / 432 images (85/15)
- Trained for 15 epochs

### Classification Report (Test Set)

| Digit | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| 0     | 0.95      | 1.00   | 0.97     |
| 1     | 1.00      | 1.00   | 1.00     |
| 2     | 0.95      | 0.95   | 0.95     |
| 3     | 0.98      | 0.98   | 0.98     |
| 4     | 1.00      | 0.93   | 0.97     |
| 5     | 0.96      | 0.96   | 0.96     |
| 6     | 0.95      | 1.00   | 0.98     |
| 7     | 0.96      | 0.96   | 0.96     |
| 8     | 1.00      | 1.00   | 1.00     |
| 9     | 1.00      | 0.98   | 0.99     |

**Overall accuracy: 97%**

## 🧠 Model Architecture

A simple CNN (`DevanagariCNN`) with:
- 2 convolutional layers (32 → 64 filters) with ReLU activation
- Max pooling after each convolutional layer
- Dropout (0.25) for regularization
- 2 fully connected layers (128 → 10 output classes)

```
Input (1x32x32 grayscale image)
   → Conv2D(1, 32) → ReLU → MaxPool
   → Conv2D(32, 64) → ReLU → MaxPool
   → Dropout(0.25)
   → Flatten
   → Linear(64*8*8, 128) → ReLU
   → Linear(128, 10)
   → Output (10 class scores)
```

## 📁 Dataset

This project uses the **NHCD (Numeral Handwritten Character Dataset)** for Devanagari digits. The dataset is organized using `ImageFolder` format:

```
data/nhcd/nhcd/numerals/
  ├── 0/
  ├── 1/
  ├── 2/
  ...
  └── 9/
```



## ⚙️ How It Works

1. **Data Loading** — Dataset is mounted from Google Drive and unzipped into Colab's local storage.
2. **Preprocessing** — Images are converted to grayscale, resized to 32x32, converted to tensors, and normalized.
3. **Training** — The CNN is trained for 15 epochs using Adam optimizer and Cross-Entropy Loss.
4. **Evaluation** — After each epoch, the model is evaluated on the held-out test set, and the best-performing model (by test accuracy) is saved.
5. **Results** — Final model performance is summarized with a classification report, confusion matrix, and visual examples of correct/incorrect predictions.

## 🚀 How to Run

1. Open the notebook (`DevinagarDigitClassifier.ipynb`) in [Google Colab](https://colab.research.google.com/).
2. Upload your `archive.zip` (NHCD dataset) to your Google Drive root folder (`MyDrive/archive.zip`).
3. Run all cells. The notebook will:
   - Mount your Google Drive
   - Unzip and load the dataset
   - Train the CNN for 15 epochs
   - Save the best model to your Google Drive (`best_model.pth`)
   - Display sample predictions and a full performance report

## 🛠️ Tech Stack

- Python
- PyTorch / torchvision
- scikit-learn (classification report, confusion matrix)
- Matplotlib (visualization)
- Google Colab (GPU training)

## 📌 Future Improvements

- Add data augmentation to improve generalization
- Experiment with deeper architectures (e.g., ResNet-style blocks)
- Add a simple web demo (e.g., Gradio/Streamlit) for live digit recognition
- Expand to full Devanagari character set (not just digits)

## 📄 License

This project is open-source and available for learning/educational purposes.

## Author

Name : Ranjit Damase  Contact No:9864583081