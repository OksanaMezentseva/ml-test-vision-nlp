import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
from src.ImageClassifier import ImageClassifier

def main(args):
    # Load class mapping
    with open(args.class_map_path, "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Load model
    classifier = ImageClassifier(data_dir=None, model_save_path=args.model_path)
    classifier.load_trained_model()
    
    # Run prediction
    classifier.predict(classifier.val_loader)
    
    # Evaluate model
    classifier.evaluate_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a trained image classifier.")
    
    parser.add_argument("--class_map_path", type=str, default="models/class_map.json", help="Path to class mapping JSON file")
    parser.add_argument("--model_path", type=str, default="models/classifier_model.pth", help="Path to trained model")
    
    args = parser.parse_args()
    main(args)