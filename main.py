import fiftyone as fo
import os
from dotenv import load_dotenv

load_dotenv()

dataset_path = os.getenv('DATASET_PATH')

# Create a new dataset for food spoilage
dataset = fo.Dataset(name="fruit_veg_dataset")
categories = ["Healthy", "Rotten"]
samples = []

# Loop through all subfolders in the dataset
for root, dirs, files in os.walk(dataset_path):
    if files:  # If there are image files in this folder
        subfolder = os.path.basename(root)  # Extract folder name
        parts = subfolder.split('__')  # Example: "Apple_Healthy" → ["Apple", "Healthy"]

        if len(parts) == 2:  # Ensure correct format
            food_type = parts[0]  # "Apple", "Banana", etc.
            condition = parts[1].lower()  # Convert to lowercase ("healthy" or "rotten")

            for file in files:
                img_path = os.path.join(root, file)
                
                # Ensure the file is an image before adding it
                if img_path.lower().endswith((".jpg", ".png", ".jpeg")):
                    sample = fo.Sample(filepath=img_path)
                    sample["ground_truth"] = fo.Classification(label=condition)
                    sample["food_type"] = food_type  # Store food type as metadata
                    samples.append(sample)

dataset.add_samples(samples)
# Launch FiftyOne app
if __name__ == "__main__":
    session = fo.launch_app(dataset)
    session.wait()