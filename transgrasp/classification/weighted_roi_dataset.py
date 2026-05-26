"""ROI dataset with per-sample weights and optional P3 augment."""
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from .dataset import ROIDataset, load_class_names
from .roi_augment import apply_p3_pil_augment, build_p3_pil_augment


class WeightedROIDataset(ROIDataset):
    def __init__(
        self,
        roi_root: Path,
        split: str,
        transform=None,
        class_names: list[str] | None = None,
        aug: str = 'none',
    ):
        super().__init__(roi_root, split, transform=transform, class_names=class_names)
        self.weights = [1.0] * len(self.rows)
        self.aug = aug
        self._pil_aug = build_p3_pil_augment() if aug == 'p3' and split == 'train' else None

        weights_path = self.split_dir / 'sample_weights.csv'
        if weights_path.is_file():
            wmap = {}
            with weights_path.open(encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    wmap[row['path']] = float(row['weight'])
            self.weights = [wmap.get(r['path'], 1.0) for r in self.rows]

    def __getitem__(self, index: int):
        row = self.rows[index]
        img_path = self.split_dir / row['path']
        image = Image.open(img_path).convert('RGB')
        if self._pil_aug is not None:
            image = apply_p3_pil_augment(image, self._pil_aug)
        if self.transform is not None:
            image = self.transform(image)
        class_name = row['class_name']
        label = self.name_to_idx[class_name]
        return image, label, str(img_path)

    def get_weight(self, index: int) -> float:
        return float(self.weights[index])
