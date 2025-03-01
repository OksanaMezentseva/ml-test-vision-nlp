import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from PIL import Image

class ImageClassifier:

    """
    A deep learning-based image classification model using ResNet18.
    This class handles dataset loading, model training, evaluation, and inference.
    """
    def __init__(self, data_dir=None, model_save_path="models/classifier_model/classifier_model.pth", 
             class_map_path=None, batch_size=32, epochs=10, learning_rate=0.001, predictions_path=None):
        
        """
        Initialize the ImageClassifier.

        Parameters:
        - data_dir (str, optional): Path to the dataset directory. If None, the classifier is used only for inference.
        - model_save_path (str): Path to save/load the trained model.
        - class_map_path (str, optional): Path to class mapping JSON file.
        - batch_size (int): Number of samples per training batch.
        - epochs (int): Number of training epochs.
        - learning_rate (float): Learning rate for optimization.
        - predictions_path (str, optional): Path to save model predictions.
        """

        self.data_dir = data_dir 
        self.model_save_path = model_save_path
        self.class_map_path = class_map_path if class_map_path else model_save_path.replace(".pth", "_class_map.json")
        self.predictions_path = predictions_path if predictions_path else model_save_path.replace(".pth", "_predictions.json")
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.data_dir:
            self.train_loader, self.val_loader, self.test_loader, self.num_classes, self.class_to_idx, self.class_weights = self.load_data()
        else:
            self.train_loader = self.val_loader = self.test_loader = None
            self.num_classes = None
            self.class_to_idx = None
            self.class_weights = None

            if os.path.exists(self.class_map_path):
                with open(self.class_map_path, "r") as f:
                    self.class_to_idx = json.load(f)
                    self.num_classes = len(self.class_to_idx)
            else:
                raise ValueError(f"❌ Class mapping file '{self.class_map_path}' not found. Please provide a valid dataset or class mapping file.")

        self.model = self.build_model()
        
    def load_data(self):

        """
        Loads the dataset, applies transformations, calculates class weights, and splits data into
        training, validation, and test sets in a stratified manner.
        
        Returns:
        - train_loader (DataLoader): DataLoader for training set.
        - val_loader (DataLoader): DataLoader for validation set.
        - test_loader (DataLoader): DataLoader for test set.
        - num_classes (int): Total number of classes in the dataset.
        - class_to_idx (dict): Mapping from class names to indices.
        - class_weights (Tensor): Computed class weights to handle class imbalance.
        """

        transform = transforms.Compose([
            transforms.Resize((224, 224)), # Resize all images to 224x224 pixels
            transforms.ToTensor(),  # Convert images to tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize pixel values
        ])
        
        dataset = ImageFolder(root=self.data_dir, transform=transform)
        class_to_idx = dataset.class_to_idx
        
        # Compute class counts to handle class imbalance
        class_counts = np.bincount([label for _, label in dataset.samples])
        total_samples = sum(class_counts)
        class_weights = total_samples / (len(class_counts) * class_counts)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        
        # Ensure stratified split
        indices = np.arange(len(dataset))
        labels = np.array([label for _, label in dataset.samples])
        
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        from sklearn.model_selection import train_test_split
        train_idx, temp_idx, train_labels, temp_labels = train_test_split(
            indices, labels, stratify=labels, test_size=val_size + test_size, random_state=42
        )
        val_idx, test_idx = train_test_split(
            temp_idx, stratify=temp_labels, test_size=test_size / (val_size + test_size), random_state=42
        )
        
        # Create subset datasets
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
        
        # Initialize DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader, len(dataset.classes), class_to_idx, class_weights
    
    def build_model(self):
        """
        Builds the image classification model.

        Returns:
        - model (torch.nn.Module): The initialized deep learning model.
        """
        import torchvision.models as models
        import torch.nn as nn

        # Load a pre-trained model
        model = models.resnet18(pretrained=True)
        
        # Replace the last layer to match the number of classes
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)

        return model.to(self.device)
    
    def train(self):

        """
        Trains the model using weighted cross-entropy loss to handle class imbalance.
        Prints the loss value for each epoch.
        """

        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        print("Starting training...")
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {running_loss/len(self.train_loader):.4f}")

    def save_model_and_class_map(self):

        """
        Saves the trained model and class-to-index mapping.
        """

        torch.save(self.model.state_dict(), self.model_save_path)
        with open(self.class_map_path, "w") as f:
            json.dump(self.class_to_idx, f)
        print(f"Training complete! Model saved at {self.model_save_path}")
        print(f"Class mapping saved at {self.class_map_path}")
        
    def load_trained_model(self):

        """
        Loads a pre-trained model for inference.
        """
        self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully!")
    
    def predict(self, loader, save_results=True):

        """
        Makes predictions on the given dataset loader.
        Saves predictions to a JSON file if specified.

        Parameters:
        - loader (DataLoader): DataLoader containing images to predict.
        - save_results (bool): Whether to save predictions to a file.
        """

        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().tolist())
        
        if save_results:
            if os.path.isdir(self.predictions_path):
                self.predictions_path = os.path.join(self.predictions_path, "predictions.json")
            with open(self.predictions_path, "w") as f:
                json.dump(predictions, f)
            print(f"Predictions saved at {self.predictions_path}")

    def predict_from_image(self, image_path, idx_to_class):
        """
        Predicts the class of a single image.

        Parameters:
        - image_path (str): Path to the image file.
        - idx_to_class (dict): Mapping of class indices to class names.

        Returns:
        - str: Predicted class name.
        """
        # Define image preprocessing transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image to match model input size
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])

        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")  # Ensure it's an RGB image
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Move image tensor to device (CPU/GPU)
        image = image.to(self.device)

        # Set model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            # Get model predictions
            outputs = self.model(image)
            _, predicted_idx = torch.max(outputs, 1)  # Get index of highest probability

        # Convert index to class name
        predicted_class = idx_to_class[predicted_idx.item()]

        return predicted_class

    def evaluate_model(self):

        """ Evaluate model accuracy on the validation set using saved predictions. """

        try:
            with open(self.predictions_path, "r") as f:
                predictions = json.load(f)
        except FileNotFoundError:
            print("No saved predictions found. Generating new predictions...")
            predictions = self.predict(self.val_loader)

        correct = 0
        total = 0
        for i, (_, labels) in enumerate(self.val_loader):
            total += len(labels)
            correct += sum([pred == label.item() for pred, label in zip(
                predictions[i * self.batch_size:(i + 1) * self.batch_size], labels)])

        accuracy = 100 * correct / total
        print(f"✅ Model Accuracy on Validation Set: {accuracy:.2f}%")
