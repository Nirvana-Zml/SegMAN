#!/usr/bin/env python
import torch
from mmseg.models.modules.lass_utils import semantic_seg_to_mbg
from mmseg.models.modules.mmscope import semantic_seg_to_boundary

y_pad = torch.full((2, 1, 64, 64), 255, dtype=torch.long)
m = semantic_seg_to_mbg(y_pad, dilate_kernel=5, ignore_index=255)
b = semantic_seg_to_boundary(y_pad, ignore_index=255)
assert m.min() == 1.0 and m.max() == 1.0, 'all-pad must be background'
assert b.sum() == 0, 'all-pad must have no boundary'
# old bug: 255 counted as foreground
m_bad = semantic_seg_to_mbg(y_pad, ignore_index=None)
assert m_bad.max() == 0.0, '255 without ignore_index was wrongly foreground'
print('ignore_index fix ok')
