import pickle
from sklearn.ensemble import RandomForestClassifier
from src.mnist_classifier_interface import MnistClassifierInterface
import os
import joblib

class RandomForestMnistClassifier(MnistClassifierInterface):
    """
    Random Forest classifier for MNIST dataset.
    Implements train() and predict() methods as required by MnistClassifierInterface.
    """

    def __init__(self, n_estimators=100):
        """
        Initialize the Random Forest model.
        :param n_estimators: Number of trees in the forest (default: 100)
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators)
        self.is_trained = False  # Flag to check if the model is trained

    def train(self, X_train, y_train):
        """
        Train the Random Forest model.
        :param X_train: Feature matrix for training
        :param y_train: Labels for training
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True  # Mark model as trained
        print("✅ Random Forest model trained successfully!")

    def predict(self, X_test):
        """
        Make predictions using the trained model.
        :param X_test: Feature matrix for testing
        :return: Predicted labels
        """
        if not self.is_trained:
            raise ValueError("❌ Model is not trained! Call train() before predict().")
        return self.model.predict(X_test)

    def save_model(self, path="Image_classification/artifacts/models/rf.pkl"):
        """
        Save the trained model to a file.
        :param path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)  # Create directory if not exists

        print(f"💾 Attempting to save model to {path}...")

        try:
            joblib.dump(self.model, path)
            print(f"✅ Model successfully saved to {path} using joblib")
        except Exception as e:
            print(f"❌ Error saving model: {e}")

    def load_model(self, path="artifacts/models/rf.pkl"):
        """
        Load a trained model from a file.
        :param path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Model file not found at {path}")
        
        self.model = joblib.load(path)
        self.is_trained = True  # Mark model as trained
        print(f"Model loaded from {path}")
