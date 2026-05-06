import matplotlib.pyplot as plt
from data_loader import SODDataset, split_dataset, make_dataloaders

NUM_SAMPLES = 8
dataset = SODDataset("data/images", "data/ground_truth_mask", 128)
train_set, val_set, test_set = split_dataset(dataset)
train_loader, val_loader, test_loader = make_dataloaders(train_set=train_set,val_set=val_set, test_set=test_set)

image_batch, mask_batch = next(iter(train_loader))

fig, axes = plt.subplots(NUM_SAMPLES, 2, figsize=(6, NUM_SAMPLES * 3))

for i in range(NUM_SAMPLES):
    image = image_batch[i].permute(1,2,0)
    mask = mask_batch[i].squeeze(0)

    axes[i, 0].imshow(image)
    axes[i, 0].set_title("Image")
    axes[i, 0].axis("off")
    
    axes[i, 1].imshow(mask, cmap="gray")
    axes[i, 1].set_title("Mask")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.savefig("batch_preview.png")
print("Saved to batch_preview.png")

