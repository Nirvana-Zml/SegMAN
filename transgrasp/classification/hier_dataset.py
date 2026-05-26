"""Hierarchical ROI datasets (P2)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .dataset import read_labels


def load_stage1_groups(roi_root: Path) -> list[str]:
    path = roi_root / 'meta' / 'stage1_groups.txt'
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}; run build_hierarchical_roi_labels.py first')
    names = [ln.strip() for ln in path.read_text(encoding='utf-8').splitlines() if ln.strip()]
    if len(names) != 2:
        raise ValueError(f'Expected 2 stage1 groups in {path}, got {names}')
    return names


def load_stage2_structure(roi_root: Path) -> list[str]:
    path = roi_root / 'meta' / 'stage2_structure.txt'
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}')
    return [ln.strip() for ln in path.read_text(encoding='utf-8').splitlines() if ln.strip()]


class HierStage1Dataset(Dataset):
    """Binary router dataset: structure vs object."""

    def __init__(self, roi_root: Path, split: str, transform=None):
        self.roi_root = Path(roi_root)
        self.split = split
        self.split_dir = self.roi_root / split
        self.transform = transform
        self.class_names = load_stage1_groups(self.roi_root)
        self.name_to_idx = {n: i for i, n in enumerate(self.class_names)}
        self.rows = read_labels(self.split_dir)
        if 'stage1_label' not in (self.rows[0] if self.rows else {}):
            raise KeyError(f'{self.split_dir}/labels.csv missing stage1_label column')

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        img_path = self.split_dir / row['path']
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        stage1 = row['stage1_label']
        if stage1 not in self.name_to_idx:
            raise KeyError(f'Unknown stage1_label {stage1!r} in {img_path}')
        return image, self.name_to_idx[stage1], str(img_path)


class HierStage2StructureDataset(Dataset):
    """3-class head on structure ROIs only: door / wall / window."""

    def __init__(self, roi_root: Path, split: str, transform=None):
        self.roi_root = Path(roi_root)
        self.split = split
        self.split_dir = self.roi_root / split
        self.transform = transform
        self.class_names = load_stage2_structure(self.roi_root)
        self.name_to_idx = {n: i for i, n in enumerate(self.class_names)}
        all_rows = read_labels(self.split_dir)
        self.rows = [
            r for r in all_rows
            if r.get('stage1_label') == 'structure' and r.get('stage2_label') in self.name_to_idx
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        img_path = self.split_dir / row['path']
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        name = row['stage2_label']
        return image, self.name_to_idx[name], str(img_path)

