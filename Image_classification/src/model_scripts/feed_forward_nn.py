import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import os
import numpy as np
from src.mnist_classifier_interface import MnistClassifierInterface

class FeedForwardNN(nn.Module):
    """
    A simple Feed-Forward Neural Network for MNIST classification.
    """
    def __init__(self):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)  # Second fully connected layer
        self.fc3 = nn.Linear(64, 10)  # Output layer for 10 classes
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class FeedForwardNNClassifier(MnistClassifierInterface):
    """
    Class implementing a Feed-Forward Neural Network for MNIST classification.
    """
    def __init__(self, lr=0.001, epochs=200):
        """
        Initializes the FFNN model, optimizer, and loss function.
        :param lr: Learning rate
        :param epochs: Number of training epochs
        """
        self.model = FeedForwardNN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = epochs
        self.is_trained = False  # Flag to track training status

    def train(self, X_train, y_train):
        """
        Trains the Feed-Forward Neural Network.
        :param X_train: Training images
        :param y_train: Training labels
        """
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = self.criterion(outputs, y_train_tensor)
            loss.backward()
            self.optimizer.step()

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")

        self.is_trained = True
        print("✅ Feed-Forward Neural Network model trained successfully!")

    def predict(self, X_test):
        """
        Makes predictions using the trained model.
        :param X_test: Test images
        :return: Predicted labels
        """
        if not self.is_trained:
            raise ValueError("❌ Model is not trained! Call train() before predict().")

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            predictions = torch.argmax(outputs, dim=1).numpy()
        return predictions

    def save_model(self, path="artifacts/models/ff_nn.pth"):
        """
        Saves the trained model to a file.
        :param path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"💾 Model saved to {path}")

    def load_model(self, path="artifacts/models/ff_nn.pth"):
        """
        Loads a trained model from a file.
        :param path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Model file not found at {path}")
        self.model.load_state_dict(torch.load(path))
        self.model.eval()  # Set to evaluation mode
        self.is_trained = True
        print(f"✅ Model loaded from {path}")