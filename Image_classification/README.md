# MNIST Classification: Random Forest, Feed-Forward NN, and CNN

## 📌 Project Overview

This project provides an object-oriented approach to solving the MNIST digit classification problem using three different machine learning models:

- **Random Forest (RF)** - A traditional ensemble learning method.
- **Feed-Forward Neural Network (FFNN)** - A basic multi-layer perceptron implemented in PyTorch.
- **Convolutional Neural Network (CNN)** - A deep learning model designed for image recognition.

All models follow a unified interface to ensure consistency in training and inference.

## 🛠 Installation

Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## 📂 Project Structure

```
Image_classification/
│── artifacts/                   # Stores trained models
│   ├── models/                   # Saved models
│── src/                         # Source code
│   ├── models/                   # Model implementations
│   │   ├── random_forest.py      # Random Forest Classifier
│   │   ├── feed_forward_nn.py    # Feed-Forward Neural Network
│   │   ├── cnn.py                # Convolutional Neural Network
│   ├── mnist_classifier.py       # Model selection class
│   ├── mnist_classifier_interface.py  # Interface for classifiers
│── train.py                      # Training script
│── demo.ipynb                     # Jupyter Notebook demo
│── requirements.txt                # Project dependencies
│── README.md                       # Project documentation
```
## Jupyter Notebook Demo

You can explore the project, train models, and run inference using the provided Jupyter Notebook:

 demo.ipynb

## Training the Models

To train a model, run the following command:

```bash
python3 train.py --algorithm <rf|nn|cnn>
```

For example, to train a CNN model:

```bash
python3 train.py --algorithm cnn
```

This will save the trained model inside `artifacts/models/`.

### 2. Download Models
Since the `models/` directories are too large for GitHub, download them from the provided link:
- **[Download Models](https://drive.google.com/file/d/1qzaWCcprs3PXN53g5AXOUHEuqTYZvh1g/view?usp=sharing)**


## 📌 Key Features

- **Object-Oriented Design:** Each model follows a common interface (`MnistClassifierInterface`).
- **Multiple Models Supported:** Choose between Random Forest, Feed-Forward NN, and CNN.
- **Pretrained Model Loading:** Load saved models for fast inference.
- **Modular & Scalable:** Easily extendable to new models.

## 🛠 Dependencies

- Python 3.8+
- Scikit-Learn
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Joblib

## 📝 Future Improvements

- Implement hyperparameter tuning to optimize model performance.
- Use **data augmentation** techniques to improve generalization, especially for misclassified digits.

## ✨ Contributors

- **Oksana Mezentseva** - ML Engineer


