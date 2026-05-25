"""ROI image dataset from trans10k_roi_gt labels.csv."""
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


def load_class_names(root: Path) -> list[str]:
    path = root / 'meta' / 'classes.txt'
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}')
    names = [ln.strip() for ln in path.read_text(encoding='utf-8').splitlines() if ln.strip()]
    if not names:
        raise ValueError(f'Empty classes file: {path}')
    return names


def read_labels(split_dir: Path) -> list[dict]:
    labels_path = split_dir / 'labels.csv'
    if not labels_path.is_file():
        raise FileNotFoundError(f'Missing {labels_path}')
    rows = []
    with labels_path.open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


class ROIDataset(Dataset):
    def __init__(self, roi_root: Path, split: str, transform=None, class_names: list[str] | None = None):
        self.roi_root = Path(roi_root)
        self.split = split
        self.split_dir = self.roi_root / split
        self.transform = transform
        self.class_names = class_names or load_class_names(self.roi_root)
        self.name_to_idx = {n: i for i, n in enumerate(self.class_names)}
        self.rows = read_labels(self.split_dir)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        img_path = self.split_dir / row['path']
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        class_name = row['class_name']
        if class_name not in self.name_to_idx:
            raise KeyError(f'Unknown class {class_name!r} in {img_path}')
        label = self.name_to_idx[class_name]
        return image, label, str(img_path)
