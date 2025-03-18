!pip install fiftyone


from google.colab import files
uploaded = files.upload()

import py7zr
import os

# Define paths
seven_z_file = "/content/images.7z"
extract_path = "/content"

# Extract the .7z file
with py7zr.SevenZipFile(seven_z_file, mode='r') as archive:
    archive.extractall(path=extract_path)

print("Files extracted to:", extract_path)

import fiftyone as fo
import fiftyone.zoo as foz


dataset = fo.Dataset.from_images_dir("/content/images") 


model = foz.load_zoo_model("clip-vit-base32-torch")


dataset.apply_model(model, label_field="predictions")

session = fo.launch_app(dataset)

model = foz.load_zoo_model(
    "clip-vit-base32-torch",
    text_prompt="A photo of a",
    classes=["banana", "ripe apple", "brown banana", "old apple", "wrinkly apple", "brown apple"],
)

dataset.apply_model(model, label_field="predictions")
session.refresh()



session = fo.launch_app(dataset)



session.show()


print(session.dataset.first())


from fiftyone import ViewField as F

session.view = (
    dataset
    .sort_by("uniqueness", reverse=True)
    .limit(25)
    .filter_labels("predictions", F("confidence") > 0.5)
)


# Computes the mAP of the predictions in the `predictions` field
# w.r.t. the ground truth labels in the `ground_truth` field
results = dataset.evaluate_detections(
    "predictions",
    gt_field="ground_truth",
    compute_mAP=True,
)

print("\nmAP: %.4f" % results.mAP())

"""evaluate only predictions with confidence greater than 0.75:"""

# Create a view that only contains predictions with confidence > 0.75
high_conf_view = dataset.filter_labels("predictions", F("confidence") > 0.75)

results = high_conf_view.evaluate_detections(
    "predictions",
    gt_field="ground_truth",
    eval_key="eval",
)


counts = dataset.count_values("ground_truth.detections.label")
classes = sorted(counts, key=counts.get, reverse=True)[:10]


results.print_report(classes=classes)

print(dataset)


session.view = high_conf_view.sort_by("eval_fp", reverse=True)


import fiftyone.brain as fob


fob.compute_mistakenness(
    high_conf_view,
    "predictions",
    label_field="ground_truth",
    use_logits=False,
)



print(dataset)



sample = dataset.first()
print(sample.ground_truth.detections[0])



session.view = high_conf_view.filter_labels("ground_truth", F("mistakenness") > 0.95)



session.freeze()
