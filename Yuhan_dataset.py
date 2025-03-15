import fiftyone as fo
import os

# Path to the dataset
dataset_path = r"C:\Users\qiuyu\OneDrive\Desktop\Food_Safety_AI\Fruit_And_Veg_Dataset"

# Check if the dataset path exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist.")

# Delete existing dataset if it exists
dataset_name = "fruit_veg_dataset"
if dataset_name in fo.list_datasets():
    fo.delete_dataset(dataset_name)
    print(f"Deleted existing dataset: {dataset_name}")

# Create a new persistent dataset
dataset = fo.Dataset(name=dataset_name, persistent=True)
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

# Add samples to the dataset
dataset.add_samples(samples)
print(f"Added {len(samples)} images to the dataset.")

# Launch FiftyOne app
if __name__ == "__main__":
    session = fo.launch_app(dataset)
    session.wait()