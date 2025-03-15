import fiftyone as fo
import fiftyone.zoo as foz

# Load a sample dataset
dataset = foz.load_zoo_dataset("coco-2017", split="validation")  # Replace with food dataset

# Create a new dataset for food spoilage
food_dataset = fo.Dataset(name="food_spoilage")

# Load image samples (fresh and spoiled)
food_dataset.add_sample(fo.Sample(filepath="fresh_apple.jpg", tags=["fresh"]))
food_dataset.add_sample(fo.Sample(filepath="rotten_apple.jpg", tags=["spoiled"]))

# Launch FiftyOne app
session = fo.launch_app(food_dataset)
session.wait()