import fiftyone as fo
import kagglehub as kaggle
import os
import zipfile as zip

# Download latest version
dataset_path = kaggle.dataset_download("muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten")
extract_path = os.path.join(os.getcwd(), "fruit_veg_dataset")

with zip.Zipfile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Create a new dataset for food spoilage
dataset = fo.Dataset(name="food_spoilage")

# Load image samples (fresh and spoiled)
dataset.add_sample(fo.Sample(filepath="fresh_apple.jpg", tags=["fresh"]))
dataset.add_sample(fo.Sample(filepath="rotten_apple.jpg", tags=["spoiled"]))

# Launch FiftyOne app
if __name__ == "__main__":
    session = fo.launch_app(dataset)
    session.wait()