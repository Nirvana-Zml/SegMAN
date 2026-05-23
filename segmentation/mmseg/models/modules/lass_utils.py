"""Utilities for LASS encoder (background mask from semantic labels)."""

import torch
import torch.nn.functional as F


def semantic_seg_to_mbg(seg_map, target_size=None, dilate_kernel=5, ignore_index=255):
    """Build background mask M_bg from semantic segmentation labels.

    Args:
        seg_map (Tensor): (B, H, W) or (B, 1, H, W), class ids (0=background).
        target_size (tuple, optional): (H, W) to resize mask with nearest.
        dilate_kernel (int): Dilate foreground before inverting so boundary
            pixels are treated as foreground.
        ignore_index (int, optional): Label ids to ignore (e.g. pad 255).
            Pixels equal to ignore_index are not treated as foreground.

    Returns:
        Tensor: (B, 1, H, W), 1 on background, 0 on foreground.
    """
    if seg_map.dim() == 3:
        seg_map = seg_map.unsqueeze(1)
    if ignore_index is None:
        fg = (seg_map > 0).float()
    else:
        valid = seg_map != ignore_index
        fg = ((seg_map > 0) & valid).float()
    if dilate_kernel > 1:
        pad = dilate_kernel // 2
        fg = F.max_pool2d(fg, dilate_kernel, stride=1, padding=pad)
    m_bg = 1.0 - fg
    if target_size is not None:
        m_bg = F.interpolate(
            m_bg, size=target_size, mode='nearest', align_corners=None)
    return m_bg
