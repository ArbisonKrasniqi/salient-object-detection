from PIL import Image

# Pick a single mask pair
imagename = "0402"
image_path = f"data/images/{imagename}.jpg"
mask_path = f"data/ground_truth_mask/{imagename}.png"

# Load the images
image = Image.open(image_path)
mask = Image.open(mask_path)

print(f"Image loaded: {image}")
print(f"Mask loaded: {mask}")

import numpy as np

#Convert PIL images to NumPy arrays so we can inspect the actual pixel data
image_array = np.array(image)
mask_array = np.array(mask)

print() #Blank line for reading
print(f"Image array shape:  {image_array.shape}")
print(f"Mask array shape:  {mask_array.shape}")
print(f"Image min/max values: {image_array.min()} to {image_array.max()}")


print()
print(f"Mask array shape: {mask_array.shape}")
print(f"Mask array dtype: {mask_array.dtype}")
print(f"Mask min/max values: {mask_array.min()} to {mask_array.max()}")
print(f"Mask unique values: {np.unique(mask_array)}")

import matplotlib.pyplot as plt

#Display image and mask side by side in a single window
fig, axes = plt.subplots(1,2, figsize=(10,5))

axes[0].imshow(image_array)
axes[0].set_title("Input Image")
axes[0].axis("off")

axes[1].imshow(mask_array, cmap="gray")
axes[1].set_title("Ground Truth Mask")
axes[1].axis("off")

plt.tight_layout()
plt.show()