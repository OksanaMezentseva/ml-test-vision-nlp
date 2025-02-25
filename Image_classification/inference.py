import argparse
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from src.mnist_classifier import MnistClassifier
from PIL import Image

def preprocess_image(image_path):
    """
    Loads and preprocesses a single handwritten digit image.
    :param image_path: Path to the image file.
    :return: Processed image as a NumPy array.
    """
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to MNIST format
    img = np.array(img).astype(np.float32)

    print(f"🔍 Image max pixel value: {img.max()}, min: {img.min()}")  # Debug info

    img = img / 255.0  # Normalize pixel values to [0,1]
    img = img.flatten().reshape(1, -1)  # Flatten for model input
    return img


def load_trained_model(algorithm):
    """
    Loads the trained model from the artifacts folder.
    :param algorithm: "rf", "nn", or "cnn"
    :return: Loaded model
    """
    model_path = os.path.join(os.path.dirname(__file__), "artifacts/models", f"{algorithm}.pkl")

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        exit(1)  # Exit with error

    print(f"💾 Loading trained model from {model_path}...")
    return joblib.load(model_path)  # Load model

def main(algorithm, filename):
    """
    Load the trained model and make predictions.
    :param algorithm: "rf", "nn", or "cnn"
    :param filename: Name of the image file (from test_images/)
    """
    # ✅ Ensure filename was provided
    if not filename:
        print("❌ No filename provided. Exiting inference.")
        exit(1)

    # ✅ Construct full path to the image
    image_path = os.path.join("artifacts/test_images", filename)

    # ✅ Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        exit(1)

    print(f"📂 Loading image: {image_path}")
    X_test = preprocess_image(image_path)

    # ✅ Load the trained model
    model = load_trained_model(algorithm)

    # ✅ Predict and display the image
    predictions = model.predict(X_test)
    img = Image.open(image_path)
    plt.imshow(img, cmap="gray")
    plt.title(f"Predicted: {predictions[0]}")
    plt.axis("off")
    plt.show()

    print(f"✅ Model Prediction: {predictions[0]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, choices=["rf", "nn", "cnn"], required=True,
                        help="Choose model: 'rf' for Random Forest, 'nn' for Feed-Forward NN, 'cnn' for Convolutional NN")
    parser.add_argument("--filename", type=str, required=True,
                        help="Name of the image file in test_images/ (required)")
    args = parser.parse_args()
    
    main(args.algorithm, args.filename)
