import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
# import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
import pickle

# Define the dataset class
class CircleDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, annotations, transforms=None):
        """
        Args:
            image_list: List of file paths to images.
            annotations: List of dictionaries with bounding boxes and labels.
            transforms: torchvision transforms for preprocessing.
        """
        self.image_list = image_list
        self.annotations = annotations
        self.transforms = transforms

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        img = Image.open(img_path).convert("RGB")
        boxes = torch.as_tensor(self.annotations[idx]["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(self.annotations[idx]["labels"], dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        if self.transforms:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.image_list)

# Define the model
def get_model(num_classes):
    # Load pre-trained Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

    # Replace the classifier with a new one for our specific classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# Training script
def train_model(model, dataloader, optimizer, num_epochs=10, device="cuda"):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Main function
if __name__ == "__main__":
    # Simulated data (Replace with your dataset)


    with open("image_list.pkl", 'rb') as file:
        # Deserialize and retrieve the variable from the file
        image_list = pickle.load(file)
    with open("annotations.pkl", 'rb') as file:
        # Deserialize and retrieve the variable from the file
        annotations = pickle.load(file)

    # Dataset and DataLoader
    dataset = CircleDataset(image_list, annotations, transforms=F.to_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Initialize model
    num_classes = 2  # 1 class (circle/ellipse) + background
    model = get_model(num_classes)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Train the model
    train_model(model, dataloader, optimizer, num_epochs=10, device="cpu")

    torch.save(model.state_dict(), "faster_rcnn_contacts.pth")

