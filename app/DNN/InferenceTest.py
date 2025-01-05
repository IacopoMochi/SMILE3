import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import torchvision.transforms.functional as F

import numpy as np
import matplotlib.pyplot as plt

# Define the model architecture
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
num_classes = 2  # Adjust to match your number of classes
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load the saved weights
model.load_state_dict(torch.load("faster_rcnn_contacts.pth"))

# Set the model to evaluation mode
model.eval()


# Load an image
image = Image.open("Ctest1.tif").convert("RGB")
image_tensor = F.to_tensor(image).unsqueeze(0)  # Convert to tensor and add batch dimension

# Perform inference
with torch.no_grad():
    predictions = model(image_tensor)
for idx, prediction in enumerate(predictions):
    print(f"Image {idx + 1}:")
    boxes = prediction["boxes"]  # Bounding box coordinates
    labels = prediction["labels"]  # Class labels
    scores = prediction["scores"]  # Confidence scores

plt.imshow(image)
s = np.shape(boxes)
for n in range(0,s[1]):
    b = boxes[n]
    xmin = b[0]
    ymin = b[1]
    xmax = b[2]
    ymax = b[3]
    plt.plot([xmin, xmax, xmax, xmin, xmin],[ymin, ymin, ymax, ymax, ymin],'r-')

plt.show()