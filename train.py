import torch
import torch.nn as nn
from data_loader import SODDataset, split_dataset, make_dataloaders
from sod_model import SODModel

def iou_loss(predictions, targets, smooth=1e-6):
    # I added smooth 1e-6 that is adde to top and bottom to avoid division by zero
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou

def combined_loss(predictions, targets):
    bce = nn.BCELoss()(predictions, targets)
    iou = iou_loss(predictions, targets)
    return bce + 0.5 * iou

EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = SODDataset("data/images", "data/ground_truth_mask")
train_set, val_set, test_set = split_dataset(dataset)
train_loader, val_loader, test_loader = make_dataloaders(train_set, val_set, test_set)

model = SODModel().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for images, masks in train_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        predictions = model(images)
        loss = combined_loss(predictions, masks)
        loss.backward() #back propagation
        optimizer.step() #update weights

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            predictions = model(images)
            loss = combined_loss(predictions, masks)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "checkpoints/best_model.pth")
        print(f"    --> Best model saved (val loss: {best_val_loss:.4f})")