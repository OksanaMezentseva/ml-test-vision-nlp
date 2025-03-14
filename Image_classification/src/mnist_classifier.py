from src.model_scripts.random_forest import RandomForestMnistClassifier
from src.model_scripts.feed_forward_nn import FeedForwardNNClassifier
from src.model_scripts.feed_forward_nn import FeedForwardNN
from src.model_scripts.cnn import CNNClassifier

class MnistClassifier:
    """
    Wrapper class for selecting and using a specific MNIST classification model.
    """

    def __init__(self, algorithm):
        """
        Initializes the classifier based on the selected algorithm.
        :param algorithm: "rf" (Random Forest), "nn" (Feed-Forward NN), "cnn" (Convolutional NN)
        """
        if algorithm == "rf":
            self.model = RandomForestMnistClassifier()
        elif algorithm == "nn":
            self.model = FeedForwardNNClassifier()
        elif algorithm == "cnn":
            self.model = CNNClassifier()
        else:
            raise ValueError("❌ Unsupported algorithm. Choose from: 'rf', 'nn', 'cnn'.")

    def train(self, X_train, y_train):
        """
        Trains the selected model.
        :param X_train: Feature matrix for training
        :param y_train: Labels for training
        """
        self.model.train(X_train, y_train)

    def predict(self, X_test):
        """
        Makes predictions using the selected model.
        :param X_test: Feature matrix for testing
        :return: Predicted labels
        """
        return self.model.predict(X_test)

    def save_model(self, filepath):
        self.model.save_model(filepath)
    
    def load_model(self, filepath):
        self.model.load_model(filepath)
