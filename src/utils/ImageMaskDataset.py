from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

from torch.utils.data import Dataset
from PIL import Image


class ImageMaskDataset(Dataset):
    """
    Paired segmentation dataset:
      - image: .jpg
      - mask:  .png
    Pairing rule: same filename stem, e.g. filename.jpg <-> filename.png

    Expected structure:
      root/
        train/
          *.jpg
          *.png
        test/
          *.jpg
          *.png
    """

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Optional[Callable] = None,  # applied to IMAGE
        target_transform: Optional[Callable] = None,  # applied to MASK
        joint_transform: Optional[
            Callable
        ] = None,  # applied to (IMAGE, MASK) together (for synced aug)
        image_ext: str = ".jpg",
        mask_ext: str = ".png",
        strict_pairs: bool = True,
    ):
        self.root = Path(root)
        self.split_dir = self.root / ("train" if train else "test")
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.image_ext = image_ext.lower()
        self.mask_ext = mask_ext.lower()

        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Missing split dir: {self.split_dir}")

        # Collect all images, then map to masks by stem
        self.images = sorted(self.split_dir.glob(f"*{self.image_ext}"))
        if not self.images:
            raise FileNotFoundError(f"No {self.image_ext} files found in {self.split_dir}")

        self.masks = []
        missing = []
        for img_path in self.images:
            mask_path = img_path.with_suffix(self.mask_ext)
            if mask_path.exists():
                self.masks.append(mask_path)
            else:
                self.masks.append(None)
                missing.append(img_path.name)

        if strict_pairs and missing:
            raise FileNotFoundError(
                "Missing masks for the following images:\n"
                + "\n".join(missing[:50])
                + ("" if len(missing) <= 50 else f"\n... and {len(missing)-50} more")
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[object, object]:
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for {img_path.name}")

        # PIL like torchvision datasets (returns PIL unless you transform to tensor)
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)  # keep mode as-is (often "L" for binary / indexed)
        if image.size != mask.size:
            raise RuntimeError(
                f"Size mismatch for {img_path.name}: image {image.size}, mask {mask.size}"
            )

        # For segmentation augments that must match (crop/flip/etc.)
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask
