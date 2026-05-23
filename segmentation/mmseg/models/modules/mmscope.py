"""MMSCopE modules: BPM, MSBEC, boundary GT."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize


def semantic_seg_to_boundary(seg_map, dilate_k=5, erode_k=5, ignore_index=255):
    """Build binary boundary map Y_bd from semantic labels.

    Y_bd = Dilate(fg) - Erode(fg), fg = valid foreground (class > 0, not ignore).

    Args:
        seg_map: (B, H, W) or (B, 1, H, W), class indices.
        dilate_k, erode_k: morphological kernel sizes (odd recommended).
        ignore_index (int, optional): Padded / invalid labels (default 255).

    Returns:
        (B, 1, H, W) float in [0, 1].
    """
    if seg_map.dim() == 3:
        seg_map = seg_map.unsqueeze(1)
    if ignore_index is None:
        fg = (seg_map > 0).float()
    else:
        valid = seg_map != ignore_index
        fg = ((seg_map > 0) & valid).float()
    if dilate_k > 1:
        pad = dilate_k // 2
        dilated = F.max_pool2d(fg, dilate_k, stride=1, padding=pad)
    else:
        dilated = fg
    if erode_k > 1:
        pad = erode_k // 2
        eroded = 1.0 - F.max_pool2d(
            1.0 - fg, erode_k, stride=1, padding=pad)
    else:
        eroded = fg
    return (dilated - eroded).clamp(0.0, 1.0)


class BoundaryProbabilityModule(nn.Module):
    """Boundary probability map (BPM) and spatial weight W_bd."""

    def __init__(self, in_channels, norm_cfg=None):
        super().__init__()
        norm_cfg = norm_cfg or dict(type='SyncBN', requires_grad=True)
        self.bd_head = nn.Sequential(
            ConvModule(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU'),
            ),
            ConvModule(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU'),
            ),
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )
        self.w_head = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) fused decode feature (_c).

        Returns:
            p_bd: (B, 1, H, W) sigmoid boundary probability.
            w_bd: (B, 1, H, W) spatial refine weight.
            bd_logits: (B, 1, H, W) for BCEWithLogits loss.
        """
        bd_logits = self.bd_head(x)
        p_bd = torch.sigmoid(bd_logits)
        p_pool = F.adaptive_avg_pool2d(p_bd, 1).expand_as(p_bd)
        w_bd = self.w_head(torch.cat([p_bd, p_pool], dim=1))
        return p_bd, w_bd, bd_logits


class MultiScaleBoundaryEnhance(nn.Module):
    """Multi-scale boundary-enhanced convolution (MSBEC)."""

    def __init__(self, channels, norm_cfg=None):
        super().__init__()
        norm_cfg = norm_cfg or dict(type='SyncBN', requires_grad=True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                groups=channels, bias=False),
            ConvModule(
                channels, channels, kernel_size=1, norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU')),
        )
        self.branch1 = nn.Sequential(
            ConvModule(
                channels, channels, kernel_size=3, stride=2, padding=1,
                norm_cfg=norm_cfg, act_cfg=dict(type='ReLU')),
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                groups=channels, bias=False),
            ConvModule(
                channels, channels, kernel_size=1, norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU')),
        )
        self.branch2 = nn.Sequential(
            ConvModule(
                channels, channels, kernel_size=5, stride=4, padding=2,
                norm_cfg=norm_cfg, act_cfg=dict(type='ReLU')),
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                groups=channels, bias=False),
            ConvModule(
                channels, channels, kernel_size=1, norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU')),
        )
        self.ref_conv = ConvModule(
            channels * 3 + 1,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
        )
        self.fuse_conv = ConvModule(
            channels * 2,
            channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
        )

    def forward(self, f_sem, p_bd):
        """Fuse multi-scale boundary features with F_sem (residual)."""
        if p_bd.shape[-2:] != f_sem.shape[-2:]:
            p_bd = resize(
                p_bd, size=f_sem.shape[-2:], mode='bilinear',
                align_corners=False)
        s0 = self.branch0(f_sem)
        s1 = resize(
            self.branch1(f_sem), size=s0.shape[-2:],
            mode='bilinear', align_corners=False)
        s2 = resize(
            self.branch2(f_sem), size=s0.shape[-2:],
            mode='bilinear', align_corners=False)
        f_ref = self.ref_conv(torch.cat([s0, s1, s2, p_bd], dim=1))
        return self.fuse_conv(torch.cat([f_sem, f_ref], dim=1)) + f_sem
