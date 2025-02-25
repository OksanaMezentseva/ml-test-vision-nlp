from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    """
    Abstract class for all MNIST classifiers.
    Every model must implement the train() and predict() methods.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model.
        :param X_train: Feature matrix for training
        :param y_train: Labels for training
        """
        pass

    @abstractmethod
    def predict(self, X_test):
        """
        Make predictions on test data.
        :param X_test: Feature matrix for testing
        :return: Predicted labels
        """
        pass
