from transformers import pipeline
from PIL import Image
import os

# Load the zero-shot image classification model
checkpoint = "openai/clip-vit-large-patch14"
detector = pipeline(model=checkpoint, task="zero-shot-image-classification")

# Define the path to your dataset
dataset_path = r"C:\Users\qiuyu\OneDrive\Desktop\Food_Safety_AI\Fruit_And_Veg_Dataset\Tomato__Rotten"

# Define candidate labels for vegetable conditions
candidate_labels = ["unripe tomato", "ripe tomato", "rotten tomato"]

# Loop through all subfolders in the dataset
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        # Check if the file is an image
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            # Construct the full image path
            image_path = os.path.join(root, file)
            
            try:
                # Open the image
                image = Image.open(image_path)
                
                # Make predictions
                predictions = detector(image, candidate_labels=candidate_labels)
                
                # Extract the predicted class (label with the highest score)
                predicted_class = predictions[0]["label"]  # The first item has the highest score
                
                # Print only the predicted class
                print(f"Predicted class: {predicted_class}")
                
            except PIL.UnidentifiedImageError as e:
                # Skip images that cannot be opened
                continue