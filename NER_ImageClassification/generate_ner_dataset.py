import json
import random

# List of animals from the image dataset
animals = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]

# Example sentence templates
sentence_templates = [
    "I saw a {} in the park.",
    "There is a {} in the picture.",
    "A {} was running across the road.",
    "Have you ever seen a {} before?",
    "The {} is my favorite animal.",
    "A {} appeared in my backyard yesterday.",
    "Look at that {}! It's so beautiful.",
    "They found a {} near the river.",
    "A {} was spotted in the zoo.",
    "Can you identify this {}?"
]

# Generate dataset
num_samples = 200  # Number of sentences to generate
data = []
for _ in range(num_samples):
    animal = random.choice(animals)
    sentence = random.choice(sentence_templates).format(animal)
    start_idx = sentence.index(animal)
    end_idx = start_idx + len(animal)
    data.append({"text": sentence, "entities": [[start_idx, end_idx, "ANIMAL"]]})

# Save to JSON file
dataset_path = "data/ner_dataset.json"
with open(dataset_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print(f"NER dataset generated and saved to {dataset_path}")