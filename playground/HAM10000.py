# HAM10000 has 7 diagnostic categories
import os
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms as T

from fluke.data import DataContainer

HAM10000_CLASSES = {
    "akiec": 0,  # Actinic keratoses
    "bcc":   1,  # Basal cell carcinoma
    "bkl":   2,  # Benign keratosis-like lesions
    "df":    3,  # Dermatofibroma
    "mel":   4,  # Melanoma
    "nv":    5,  # Melanocytic nevi
    "vasc":  6,  # Vascular lesions
}


class _HAM10000Raw(Dataset):
    """Internal PyTorch Dataset that reads HAM10000 images from disk."""

    def __init__(self, df: pd.DataFrame, img_dirs: list[str], transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dirs = img_dirs
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx]["image_id"]
        label = self.df.iloc[idx]["label"]

        img = None
        for d in self.img_dirs:
            p = os.path.join(d, image_id + ".jpg")
            if os.path.exists(p):
                img = Image.open(p).convert("RGB")
                break

        if img is None:
            raise FileNotFoundError(f"Image {image_id}.jpg not found in any of {self.img_dirs}")

        if self.transform:
            img = self.transform(img)

        return img, label


def HAM10000(
    path: str = r"./data/HAM10000",
    transforms: Optional[callable] = None,
    onthefly_transforms: Optional[callable] = None,
    img_size: int = 64,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DataContainer:
    """
    Load the HAM10000 (Human Against Machine with 10000 training images) dataset.

    Expected folder structure::

        path/
        ├── HAM10000_metadata.csv
        ├── HAM10000_images_part_1/   (*.jpg)
        └── HAM10000_images_part_2/   (*.jpg)

    The dataset contains 10,015 dermoscopic images across 7 skin lesion classes:
    ``akiec``, ``bcc``, ``bkl``, ``df``, ``mel``, ``nv``, ``vasc``.

    Since no official train/test split is provided, the data is split with
    stratification on the label (80/20 by default).

    Args:
        path (str): Root directory where the dataset is stored. Defaults to ``../data/HAM10000``.
        transforms (callable, optional): Torchvision-compatible transformations applied
            when loading images from disk. If ``None``, images are resized to
            ``(3, img_size, img_size)`` and normalised to [0, 1].
        onthefly_transforms (callable, optional): Transformations applied on-the-fly
            via the data loader (e.g. random augmentations). Defaults to ``None``.
        img_size (int): Height/width to resize images to when no custom transforms
            are provided. Defaults to ``64``.
        test_size (float): Fraction of samples reserved for the test split.
            Defaults to ``0.2``.
        random_state (int): Random seed for the stratified split. Defaults to ``42``.

    Returns:
        DataContainer: The HAM10000 dataset.
    """

    # ------------------------------------------------------------------ #
    # 1. Read metadata
    # ------------------------------------------------------------------ #
    metadata_path = os.path.join(path, "HAM10000_metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata file not found at {metadata_path}.\n"
            "Please download HAM10000 "
            f" and place it under '{path}'."
        )

    df = pd.read_csv(metadata_path)
    df["label"] = df["dx"].map(HAM10000_CLASSES)

    # ------------------------------------------------------------------ #
    # 2. Locate image directories
    # ------------------------------------------------------------------ #
    img_dirs = [
        os.path.join(path, "HAM10000_images_part_1"),
        os.path.join(path, "HAM10000_images_part_2"),
    ]
    img_dirs = [d for d in img_dirs if os.path.isdir(d)]
    if not img_dirs:
        raise FileNotFoundError(
            f"No image directories found under '{path}'. "
            "Expected 'HAM10000_images_part_1' and/or 'HAM10000_images_part_2'."
        )

    # ------------------------------------------------------------------ #
    # 3. Stratified train / test split
    # ------------------------------------------------------------------ #
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )

    # ------------------------------------------------------------------ #
    # 4. Build transforms
    # ------------------------------------------------------------------ #
    if transforms is None:
        # Resize → tensor → [0,1] (ToTensor divides by 255 automatically)
        default_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),           # → (3, H, W) in [0, 1]
        ])
        load_transform = default_transform
    else:
        load_transform = transforms

    # ------------------------------------------------------------------ #
    # 5. Load all images into tensors
    # ------------------------------------------------------------------ #
    def _load_split(split_df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        dataset = _HAM10000Raw(split_df, img_dirs, transform=load_transform)
        images, labels = [], []
        for img, label in dataset:
            images.append(img)
            labels.append(label)
        return torch.stack(images), torch.tensor(labels, dtype=torch.long)

    X_train, y_train = _load_split(train_df)
    X_test, y_test = _load_split(test_df)

    return DataContainer(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        num_classes=len(HAM10000_CLASSES),
        transforms=onthefly_transforms,
    )