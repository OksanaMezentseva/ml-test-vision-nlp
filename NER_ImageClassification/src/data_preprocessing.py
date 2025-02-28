import os
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def prepare_directories(raw_dir, processed_dir):
    """Ensure processed directory exists and remove existing files."""
    if os.path.exists(processed_dir):
        for category in os.listdir(processed_dir):
            category_path = os.path.join(processed_dir, category)
            if os.path.isdir(category_path):
                for img_name in os.listdir(category_path):
                    os.remove(os.path.join(category_path, img_name))
                os.rmdir(category_path)
    os.makedirs(processed_dir, exist_ok=True)

def get_transform():
    """Define image transformations (resizing, tensor conversion, normalization)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def process_images(raw_dir, processed_dir, transform):
    """Process images by applying transformations and saving them."""
    for category in os.listdir(raw_dir):
        category_path = os.path.join(raw_dir, category)
        save_category_path = os.path.join(processed_dir, category)
        os.makedirs(save_category_path, exist_ok=True)
        
        for img_name in tqdm(os.listdir(category_path), desc=f"Processing {category}"):
            img_path = os.path.join(category_path, img_name)
            try:
                img = Image.open(img_path).convert("RGB")  # Ensure all images are in RGB format
                img_transformed = transform(img)  # Apply transformations
                img_transformed = transforms.ToPILImage()(img_transformed)  # Convert back to PIL Image
                img_transformed.save(os.path.join(save_category_path, img_name))  # Save processed image
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

def main():
    RAW_IMG_DIR = "data/raw-img"
    PROCESSED_IMG_DIR = "data/processed-img"
    
    prepare_directories(RAW_IMG_DIR, PROCESSED_IMG_DIR)
    transform = get_transform()
    process_images(RAW_IMG_DIR, PROCESSED_IMG_DIR, transform)
    print("Processing complete!")

if __name__ == "__main__":
    main()
