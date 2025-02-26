import argparse
from src.mnist_classifier import MnistClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_mnist():
    """
    Loads the MNIST dataset and preprocesses it for training.
    :return: X_train, X_test, y_train, y_test
    """
    mnist = fetch_openml("mnist_784", version=1)
    X, y = mnist.data, mnist.target.astype(int)

    # Normalize data for neural networks
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=0.2, random_state=42)

def main(algorithm):
    """
    Train the specified model on the MNIST dataset.
    :param algorithm: "rf", "nn", or "cnn"
    """
    X_train, X_test, y_train, y_test = load_mnist()

    clf = MnistClassifier(algorithm)
    clf.train(X_train, y_train)

    clf.model.save_model()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, choices=["rf", "nn", "cnn"], required=True,
                        help="Choose model: 'rf' for Random Forest, 'nn' for Feed-Forward NN, 'cnn' for Convolutional NN")
    args = parser.parse_args()
    
    main(args.algorithm)
