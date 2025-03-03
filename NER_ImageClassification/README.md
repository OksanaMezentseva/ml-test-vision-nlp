# NER_ImageClassification

## Overview
This project implements a multimodal machine learning pipeline that integrates two different models:
1. **Named Entity Recognition (NER) Model** – extracts animal names from the input text.
2. **Image Classification Model** – classifies an image to determine which animal is present.

The pipeline is designed to verify whether a user's textual statement about an image is correct by comparing the extracted animal name(s) from text with the classified object in the image.

## Project Structure
```
NER_ImageClassification/
│── data/                     # Data directory (too large for GitHub, available for download)
│   ├── raw-img/               # Raw images
│   ├── test_images/           # Test images
│   ├── train_ner_dataset.json # Training dataset for NER model
│   ├── val_ner_dataset.json   # Validation dataset for NER model
│   ├── test_ner_dataset.json  # Test dataset for NER model
│
│── models/                    # Pretrained models (available for download)
│   ├── classifier_model/       # Image classifier model
│   ├── ner_model/              # NER model
│
│── src/                        # Source code
│   ├── image_classifier_model.py # Image classification model
│   ├── infer_classifier.py     # Inference script for image classification
│   ├── infer_ner.py            # Inference script for NER model
│   ├── ner_model.py            # NER model implementation
│   ├── pipeline.py             # Multimodal pipeline script
│   ├── train_classifier.py     # Training script for the image classifier
│   ├── train_ner.py            # Training script for the NER model
│
│── demo.ipynb                  # Jupyter Notebook for running the demo
│── EDA.ipynb                    # Exploratory Data Analysis notebook
│── generate_ner_dataset.py      # Script for generating the NER dataset
│── README.md                    # Project documentation
│── requirements.txt              # Python dependencies
```

## Installation

### 1. Install Dependencies
```sh
pip install -r requirements.txt
```

### 2. Download Models and Data
Since the `models/` and `data/` directories are too large for GitHub, download them from the provided link:
- **[Download Models](https://drive.google.com/file/d/1FUwW9qnwdjlq23rY6LW_5K7Xe9U3xdG3/view?usp=sharing)**
- **[Download Data](https://drive.google.com/file/d/1-mBFHxxfikUYI6LupYPPvNsLgO0HoaWX/view?usp=sharing)**

Extract them into the project root. cd NER_ImageClassification

## Training the Models
### Train the NER Model
```sh
python3 src/train_ner.py --data_path data/train_ner_dataset.json --model_save_path models/ner_model --batch_size 16 --epochs 3 --learning_rate 5e-5
```
Parameters:

--data_path (str): Path to the training dataset (default="data").

--model_save_path (str): Path where the trained NER model will be saved (default="models/ner_model").

--batch_size (int): The number of training samples per batch (default: 16).

--epochs (int): The number of training iterations over the dataset (default: 3).

--learning_rate (float): The optimizer's learning rate (default: 5e-5).

--train: Flag to enable training mode.


### Train the Image Classification Model
```sh
python3 src/train_classifier.py --data_dir data/processed-img --model_save_path models/classifier_model/classifier_model.pth --batch_size 32 --epochs 20 --learning_rate 0.001
```
Parameters:

--data_dir (str): Path to the directory containing processed training images(default="data/processed-img").

--model_save_path (str): Path where the trained image classification model will be saved(default="models/classifier_model.pth").

--batch_size (int): The number of images processed in each training batch (default: 32).

--epochs (int): The number of times the model iterates over the entire training dataset (default: 20).

--learning_rate (float): The learning rate for the optimizer, which controls how much to adjust model weights during training (default: 0.001).


## Running the Pipeline
To test the pipeline with a text and an image, run:
```sh
python3 src/pipeline.py --text "There is a cat in the picture." --image data/test_images/cat.jpg
```

## Pipeline Workflow
1. The user provides a text input and an image.
2. The **NER model** extracts potential animal names from the text.
3. The **Image classifier model** predicts the class of the animal in the image.
4. The pipeline checks if the extracted animal name matches the predicted class.
5. The system outputs `True` (if the match is correct) or `False` (if it is incorrect).

## Example
### Input:
```sh
python3 src/pipeline.py --text "I see a cat." --image data/test_images/cow.jpg
```

### Output:
```
Extracted entities: ['cat']
📌 Predicted class: cow
🔍 Matching result: False
```

## Notebooks
- **EDA.ipynb**: Exploratory Data Analysis of the dataset.
- **demo.ipynb**: Demonstrates the usage of the trained models.

## Future Improvements
- Improve NER extraction by fine-tuning with domain-specific data.
- Enhance the classification model with more diverse datasets.
- Add an out-of-distribution (OOD) detection mechanism.

