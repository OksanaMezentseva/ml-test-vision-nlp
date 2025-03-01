import torch
import torch.nn as nn
import torch.optim as optim
import os
from src.mnist_classifier_interface import MnistClassifierInterface
import torch.utils.data as data

class CNN(nn.Module):
    """
    A Convolutional Neural Network for MNIST classification.
    """
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

        # Activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))  
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)  
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  
        return x
class CNNClassifier(MnistClassifierInterface):
    """
    Class implementing a Convolutional Neural Network for MNIST classification.
    """
    def __init__(self, lr=0.001, epochs=10):
        """
        Initializes the CNN model, optimizer, and loss function.
        :param lr: Learning rate
        :param epochs: Number of training epochs
        """
        self.model = CNN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()  
        self.epochs = epochs
        self.is_trained = False  # Flag to track training status

    def train(self, X_train, y_train, batch_size=16):
        """
        Trains the CNN model with mini-batches.
        :param X_train: Training images (reshaped to 1x28x28)
        :param y_train: Training labels
        :param batch_size: Size of the training batch
        """
        dataset = data.TensorDataset(torch.tensor(X_train, dtype=torch.float32).view(-1, 1, 28, 28),
                                    torch.tensor(y_train, dtype=torch.long))
        dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(self.epochs):
            total_loss = 0
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss / len(dataloader):.4f}")

        self.is_trained = True
        print("✅ CNN model trained successfully!")

    def predict(self, X_test):
        """
        Makes predictions using the trained CNN model.
        :param X_test: Test images (reshaped to 1x28x28)
        :return: Predicted labels
        """
        if not self.is_trained:
            raise ValueError("❌ Model is not trained! Call train() before predict().")

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, 28, 28)
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            predictions = torch.argmax(outputs, dim=1).numpy()
        return predictions

    def save_model(self, path="artifacts/models/cnn.pth"):
        """
        Saves the trained CNN model to a file.
        :param path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"💾 Model saved to {path}")

    def load_model(self, path="artifacts/models/cnn.pth"):
        """
        Loads a trained CNN model from a file.
        :param path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Model file not found at {path}")
        self.model.load_state_dict(torch.load(path))
        self.model.eval()  # Set to evaluation mode
        self.is_trained = True
        print(f"✅ Model loaded from {path}")
