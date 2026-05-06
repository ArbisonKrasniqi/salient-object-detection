import torch
import matplotlib.pyplot as plt
from data_loader import SODDataset, split_dataset, make_dataloaders
from sod_model import SODModel
from unet_sod_model import UNetSODModel
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = SODDataset("data/images", "data/ground_truth_mask")
train_set, val_set, test_set = split_dataset(dataset)
train_loader, val_loader, test_loader = make_dataloaders(train_set, val_set, test_set)

#model = SODModel().to(DEVICE)
model = UNetSODModel().to(DEVICE)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=DEVICE))
model.eval()


total_iou = 0.0
total_precision = 0.0
total_recall = 0.0
total_f1 = 0.0
num_batches = 0

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        
        predictions = model(images)
        
        # Binarize predictions at 0.5 threshold
        pred_binary = (predictions > 0.5).float()
        
        intersection = (pred_binary * masks).sum()
        union = pred_binary.sum() + masks.sum() - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        tp = (pred_binary * masks).sum()
        fp = (pred_binary * (1 - masks)).sum()
        fn = ((1 - pred_binary) * masks).sum()
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        total_iou += iou.item()
        total_precision += precision.item()
        total_recall += recall.item()
        total_f1 += f1.item()
        num_batches += 1

print(f"IoU:       {total_iou / num_batches:.4f}")
print(f"Precision: {total_precision / num_batches:.4f}")
print(f"Recall:    {total_recall / num_batches:.4f}")
print(f"F1:        {total_f1 / num_batches:.4f}")

batches = list(test_loader)
images, masks = random.choice(batches)
images = images.to(DEVICE)
masks = masks.to(DEVICE)

with torch.no_grad():
    predictions = model(images)
    pred_binary = (predictions > 0.5).float()

fig, axes = plt.subplots(4,4,figsize=(12,12))
titles = ["Image", "Ground Truth", "Prediction", "Overlay"]

for i in range(4):
    image = images[i].cpu().permute(1, 2, 0)
    mask = masks[i].cpu().squeeze(0)
    pred = pred_binary[i].cpu().squeeze(0)
    overlay = image.clone()
    overlay[:, :, 0] = torch.clamp(overlay[:, :, 0] + 0.4 * pred, 0, 1)

    axes[i, 0].imshow(image)
    axes[i, 1].imshow(mask, cmap="gray")
    axes[i, 2].imshow(pred, cmap="gray")
    axes[i, 3].imshow(overlay)

    for j in range(4):
        axes[i, j].axis("off")
        if i == 0:
            axes[i, j].set_title(titles[j])


plt.tight_layout()
plt.savefig("evaluation_preview.png")
print("Saved to evaluation_preview.png")