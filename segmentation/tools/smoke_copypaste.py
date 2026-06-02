#!/usr/bin/env python3
from mmseg.datasets.pipelines import Trans10KCopyPaste
import numpy as np

p = Trans10KCopyPaste()
r = {
    'img': np.zeros((512, 512, 3), dtype=np.uint8),
    'gt_semantic_seg': np.zeros((512, 512), dtype=np.uint8),
}
r = p(r)
print('ok max_class', int(r['gt_semantic_seg'].max()))
