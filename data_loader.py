from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader
import torch
from torchvision.transforms import v2 as T

IMAGE_SIZE = 128
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_SEED = 42

BATCH_SIZE = 32
NUM_WORKERS = 4

class SODDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=IMAGE_SIZE):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)

        self.image_paths = sorted(self.images_dir.glob("*.jpg"))

        self.mask_paths = [self.masks_dir / f"{p.stem}.png" for p in self.image_paths]

        missing = [p for p in self.mask_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} mask file(s) missing. First missing: {missing[0]}"
            )
        
        ## Transform pipeline
        self.image_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale = True)
        ])

        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale = True)
        ])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        image_tensor = self.image_transform(image)
        mask_tensor = self.mask_transform(mask)

        return image_tensor, mask_tensor
    
def split_dataset(dataset, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=SPLIT_SEED):
    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size #AVOID ROUNDING ERRORS
    
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size], generator = generator
    )
    return train_set, val_set, test_set

def make_dataloaders(train_set, val_set, test_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size = batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    dataset = SODDataset(
        images_dir="data/images",
        masks_dir="data/ground_truth_mask",
    )
    print(f"Full dataset size: {len(dataset)}")

    train_set, val_set, test_set = split_dataset(dataset)
    print(f"Train/Val/Test sizes: {len(train_set)} / {len(val_set)} / {len(test_set)}")

    train_loader, val_loader, test_loader = make_dataloaders(train_set, val_set, test_set)
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Val batches per epoch:   {len(val_loader)}")
    print(f"Test batches per epoch:  {len(test_loader)}")

    # Pull one batch and confirm shapes
    image_batch, mask_batch = next(iter(train_loader))
    print(f"\nOne training batch:")
    print(f"  Image batch shape: {image_batch.shape}")
    print(f"  Mask batch shape:  {mask_batch.shape}")