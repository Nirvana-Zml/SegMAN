"""Instance predictors for scheme E (semantic / Mask R-CNN / GT oracle)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from transgrasp.pipelines.roi_extract import extract_instance_rois, padded_bbox
from transgrasp.pipelines.roi_postprocess import (
    CLASSES,
    ExtractConfig,
    InstanceROI,
    postprocess_instances,
)
from transgrasp.pipelines.seg_model import SegMANSegmentor


class BaseInstancePredictor(ABC):
    @abstractmethod
    def predict_instances(
        self, rgb: np.ndarray, extract_cfg: ExtractConfig,
    ) -> list[InstanceROI]:
        ...


class SemanticInstancePredictor(BaseInstancePredictor):
    """Default: SegMAN semantic seg -> CC (scheme B)."""

    def __init__(
        self,
        segmentor: SegMANSegmentor,
        refine_cfg=None,
        use_tta: bool = False,
        tta_scales=(0.75, 1.0, 1.25),
    ):
        self.segmentor = segmentor
        self.refine_cfg = refine_cfg
        self.use_tta = use_tta
        self.tta_scales = tta_scales

    def predict_instances(
        self, rgb: np.ndarray, extract_cfg: ExtractConfig,
    ) -> list[InstanceROI]:
        from transgrasp.pipelines.seg_refine import apply_seg_refine

        if self.use_tta:
            pred_label = self.segmentor.predict_label_map_tta(
                rgb, scales=self.tta_scales)
        else:
            pred_label = self.segmentor.predict_label_map(rgb)
        if self.refine_cfg is not None:
            pred_label = apply_seg_refine(rgb, pred_label, self.refine_cfg)
        return extract_instance_rois(rgb, pred_label, extract_cfg=extract_cfg)


class GTOracleInstancePredictor(BaseInstancePredictor):
    """Upper bound: GT semantic label -> CC (E2-0 pipeline validation)."""

    def __init__(self, gt_label: np.ndarray):
        self.gt_label = gt_label

    def predict_instances(
        self, rgb: np.ndarray, extract_cfg: ExtractConfig,
    ) -> list[InstanceROI]:
        return extract_instance_rois(rgb, self.gt_label, extract_cfg=extract_cfg)


class MaskRCNNInstancePredictor(BaseInstancePredictor):
    """E1-lite: torchvision Mask R-CNN on pseudo COCO instances."""

    SCORE_THRESH = 0.3

    def __init__(self, checkpoint: str | Path, device: str = 'cuda:0'):
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

        self.device = torch.device(
            'cuda' if device.startswith('cuda') and torch.cuda.is_available() else 'cpu')
        num_classes = len(CLASSES)  # including background

        model = maskrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, 256, num_classes)

        ckpt = Path(checkpoint)
        if not ckpt.is_file():
            raise FileNotFoundError(f'Mask R-CNN checkpoint not found: {ckpt}')
        state = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(state['model'] if 'model' in state else state)
        model.to(self.device)
        model.eval()
        self.model = model

    def predict_instances(
        self, rgb: np.ndarray, extract_cfg: ExtractConfig,
    ) -> list[InstanceROI]:
        h, w = rgb.shape[:2]
        img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        img_t = img_t.to(self.device)

        with torch.no_grad():
            out = self.model([img_t])[0]

        instances: list[InstanceROI] = []
        counters = {cid: 0 for cid in range(1, len(CLASSES))}

        scores = out['scores'].cpu().numpy()
        labels = out['labels'].cpu().numpy()
        boxes = out['boxes'].cpu().numpy()
        masks = out['masks'].cpu().numpy()

        for score, label, box, mask in zip(scores, labels, boxes, masks):
            if score < self.SCORE_THRESH:
                continue
            class_id = int(label)
            if class_id <= 0 or class_id >= len(CLASSES):
                continue
            binary = (mask[0] >= 0.5).astype(np.uint8)
            area = int(binary.sum())
            min_a = extract_cfg.min_area_per_class.get(
                CLASSES[class_id], extract_cfg.min_area)
            if area < min_a:
                continue
            x0, y0, x1, y1 = [int(v) for v in box]
            x0 = max(0, min(x0, w - 1))
            y0 = max(0, min(y0, h - 1))
            x1 = max(x0 + 1, min(x1, w))
            y1 = max(y0 + 1, min(y1, h))
            px0, py0, px1, py1 = padded_bbox(
                x0, y0, x1, y1, w, h, extract_cfg.bbox_pad)
            crop = rgb[py0:py1, px0:px1]
            if crop.size == 0:
                continue
            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[binary > 0] = 1
            class_name = CLASSES[class_id]
            inst_id = counters[class_id]
            counters[class_id] += 1
            instances.append(InstanceROI(
                class_id=class_id,
                class_name=class_name,
                instance_id=inst_id,
                bbox=(px0, py0, px1, py1),
                area=area,
                crop_rgb=crop,
                mask=full_mask,
            ))
        return postprocess_instances(instances, extract_cfg, rgb=rgb)


class Mask2FormerInstancePredictor(BaseInstancePredictor):
    """F1: mmdet Mask2Former instance masks."""

    SCORE_THRESH = 0.3

    def __init__(
        self,
        config: str | Path,
        checkpoint: str | Path,
        device: str = 'cuda:0',
        score_thresh: float = 0.3,
    ):
        from mmdet.apis import init_detector

        self.device = device
        self.score_thresh = score_thresh
        cfg_path = Path(config)
        ckpt_path = Path(checkpoint)
        if not cfg_path.is_file():
            raise FileNotFoundError(f'M2F config not found: {cfg_path}')
        if not ckpt_path.is_file():
            raise FileNotFoundError(f'M2F checkpoint not found: {ckpt_path}')
        self.model = init_detector(str(cfg_path), str(ckpt_path), device=device)

    def predict_instances(
        self, rgb: np.ndarray, extract_cfg: ExtractConfig,
    ) -> list[InstanceROI]:
        from mmdet.apis import inference_detector
        from mmdet.structures import DetDataSample

        h, w = rgb.shape[:2]
        with torch.no_grad():
            result = inference_detector(self.model, rgb)

        if isinstance(result, DetDataSample):
            pred = result.pred_instances
            scores = pred.scores.cpu().numpy()
            labels = pred.labels.cpu().numpy()
            masks = pred.masks.cpu().numpy()
        else:
            pred = result.pred_instances
            scores = pred.scores.cpu().numpy()
            labels = pred.labels.cpu().numpy()
            masks = pred.masks.cpu().numpy()

        instances: list[InstanceROI] = []
        counters = {cid: 0 for cid in range(1, len(CLASSES))}

        for score, label, mask in zip(scores, labels, masks):
            if float(score) < self.score_thresh:
                continue
            class_id = int(label) + 1  # mmdet 0-indexed -> CLASSES index
            if class_id <= 0 or class_id >= len(CLASSES):
                continue
            if mask.ndim == 3:
                mask = mask[0]
            binary = (mask > 0.5).astype(np.uint8)
            area = int(binary.sum())
            min_a = extract_cfg.min_area_per_class.get(
                CLASSES[class_id], extract_cfg.min_area)
            if area < min_a:
                continue
            ys, xs = np.where(binary > 0)
            if len(xs) == 0:
                continue
            x0, x1 = int(xs.min()), int(xs.max() + 1)
            y0, y1 = int(ys.min()), int(ys.max() + 1)
            px0, py0, px1, py1 = padded_bbox(
                x0, y0, x1, y1, w, h, extract_cfg.bbox_pad)
            crop = rgb[py0:py1, px0:px1]
            if crop.size == 0:
                continue
            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[binary > 0] = 1
            class_name = CLASSES[class_id]
            inst_id = counters[class_id]
            counters[class_id] += 1
            instances.append(InstanceROI(
                class_id=class_id,
                class_name=class_name,
                instance_id=inst_id,
                bbox=(px0, py0, px1, py1),
                area=area,
                crop_rgb=crop,
                mask=full_mask,
            ))
        return postprocess_instances(instances, extract_cfg, rgb=rgb)


def build_instance_predictor(args, segmentor=None, gt_label=None) -> BaseInstancePredictor:
    source = getattr(args, 'instance_source', 'semantic')
    if source == 'semantic':
        if segmentor is None:
            raise ValueError('segmentor required for semantic instance source')
        from transgrasp.pipelines.seg_refine import build_refine_config
        refine_cfg = build_refine_config(args) if hasattr(args, 'refine_morph_close') else None
        return SemanticInstancePredictor(
            segmentor,
            refine_cfg=refine_cfg,
            use_tta=getattr(args, 'seg_tta', False),
        )
    if source == 'gt_oracle':
        if gt_label is None:
            raise ValueError('gt_label required for gt_oracle')
        return GTOracleInstancePredictor(gt_label)
    if source == 'maskrcnn':
        return MaskRCNNInstancePredictor(
            args.maskrcnn_checkpoint,
            device=getattr(args, 'device', 'cuda:0'),
        )
    if source == 'm2f':
        return Mask2FormerInstancePredictor(
            args.m2f_config,
            args.m2f_checkpoint,
            device=getattr(args, 'device', 'cuda:0'),
            score_thresh=getattr(args, 'm2f_score_thresh', 0.3),
        )
    raise ValueError(f'Unknown instance_source: {source}')
