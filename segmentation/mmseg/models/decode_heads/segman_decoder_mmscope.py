# MMSCopE decode head: SegMANDecoder + BPM + MSBEC + L_bd
import torch
import torch.nn.functional as F

from mmseg.ops import resize
from ..builder import HEADS
from ..modules.mmscope import (
    BoundaryProbabilityModule,
    MultiScaleBoundaryEnhance,
    semantic_seg_to_boundary,
)
from .segman_decoder import SegMANDecoder


def _parse_mmscope_cfg(mmscope_cfg):
    mmscope_cfg = mmscope_cfg or {}
    return dict(
        enable_bpm=mmscope_cfg.get('enable_bpm', True),
        enable_msbec=mmscope_cfg.get('enable_msbec', True),
        boundary_loss_weight=mmscope_cfg.get('boundary_loss_weight', 0.4),
        refine_eta=mmscope_cfg.get('refine_eta', 0.1),
        dilate_kernel=mmscope_cfg.get('dilate_kernel', 5),
        erode_kernel=mmscope_cfg.get('erode_kernel', 5),
    )


@HEADS.register_module()
class SegMANDecoderMMSCopE(SegMANDecoder):
    """SegMAN decoder with boundary probability (BPM) and MSBEC fusion."""

    def __init__(self, mmscope_cfg=None, **kwargs):
        super().__init__(**kwargs)
        cfg = _parse_mmscope_cfg(mmscope_cfg)
        self.enable_bpm = cfg['enable_bpm']
        self.enable_msbec = cfg['enable_msbec']
        self.boundary_loss_weight = cfg['boundary_loss_weight']
        self.refine_eta = cfg['refine_eta']
        self.dilate_kernel = cfg['dilate_kernel']
        self.erode_kernel = cfg['erode_kernel']
        norm_cfg = dict(type='SyncBN', requires_grad=True)
        if self.enable_bpm:
            self.bpm = BoundaryProbabilityModule(
                self.embed_dim, norm_cfg=norm_cfg)
        if self.enable_msbec:
            self.msbec = MultiScaleBoundaryEnhance(
                self.embed_dim, norm_cfg=norm_cfg)
        self.last_p_bd = None
        self.last_bd_logits = None

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        x, c2, c3, c4 = self.forward_mlp_decoder(x)
        self.last_p_bd = None
        self.last_bd_logits = None
        if self.enable_bpm:
            p_bd, w_bd, bd_logits = self.bpm(x)
            self.last_p_bd = p_bd
            self.last_bd_logits = bd_logits
            w = 1.0 + self.refine_eta * w_bd
            x = x * w
            c2 = c2 * w
        x = self.forward_winssm(x, c2, c3, c4)
        if self.enable_msbec and self.last_p_bd is not None:
            x = self.msbec(x, self.last_p_bd)
        return self.cls_seg(x)

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        if (self.boundary_loss_weight > 0 and self.last_bd_logits is not None
                and self.enable_bpm):
            y_bd = semantic_seg_to_boundary(
                gt_semantic_seg,
                dilate_k=self.dilate_kernel,
                erode_k=self.erode_kernel,
                ignore_index=self.ignore_index,
            )
            if y_bd.shape[-2:] != self.last_bd_logits.shape[-2:]:
                y_bd = resize(
                    y_bd,
                    size=self.last_bd_logits.shape[-2:],
                    mode='nearest',
                )
            if self.ignore_index is not None:
                valid = (gt_semantic_seg.squeeze(1) != self.ignore_index).float()
                valid = valid.unsqueeze(1)
                if valid.shape[-2:] != y_bd.shape[-2:]:
                    valid = resize(
                        valid, size=y_bd.shape[-2:], mode='nearest')
                loss_bd = F.binary_cross_entropy_with_logits(
                    self.last_bd_logits, y_bd, reduction='none')
                loss_bd = (loss_bd * valid).sum() / valid.sum().clamp_min(1.0)
            else:
                loss_bd = F.binary_cross_entropy_with_logits(
                    self.last_bd_logits, y_bd, reduction='mean')
            losses['loss_bd'] = loss_bd * self.boundary_loss_weight
        return losses
