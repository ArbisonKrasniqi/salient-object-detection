"""
Rename MSRA10K dataset to match ECSSD naming convention.
 
Source: MSRA10K_Imgs_GT/MSRA10K_Imgs_GT/Imgs/  (images + masks mixed, sparse IDs)
Target: data_msra/images/         0001.jpg, 0002.jpg, ...
        data_msra/ground_truth_mask/  0001.png, 0002.png, ...
 
Sorted numerically by original ID before renumbering, so pairs stay aligned.
"""
 
from pathlib import Path
import shutil
 
SRC = Path("MSRA10K_Imgs_GT/MSRA10K_Imgs_GT/Imgs")
DST_IMG = Path("data_msra/images")
DST_MASK = Path("data_msra/ground_truth_mask")
 
DST_IMG.mkdir(parents=True, exist_ok=True)
DST_MASK.mkdir(parents=True, exist_ok=True)
 
# Get all .jpg files, sort numerically by stem (filename without extension)
jpgs = sorted(SRC.glob("*.jpg"), key=lambda p: int(p.stem))
 
print(f"Found {len(jpgs)} jpg files")
 
renamed = 0
missing = 0
for new_idx, jpg in enumerate(jpgs, start=1):
    png = jpg.with_suffix(".png")  # corresponding mask
    if not png.exists():
        print(f"  ⚠ missing mask for {jpg.name}, skipping")
        missing += 1
        continue
 
    new_name = f"{new_idx:04d}"  # 0001, 0002, ... zero-padded to 4 digits
    shutil.copy2(jpg, DST_IMG / f"{new_name}.jpg")
    shutil.copy2(png, DST_MASK / f"{new_name}.png")
    renamed += 1
 
print(f"\n✓ Renamed {renamed} pairs")
if missing:
    print(f"⚠ Skipped {missing} unpaired images")
 
