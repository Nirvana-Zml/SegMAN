#!/usr/bin/env python
"""Check checkpoint key loading for route B model."""
import torch
from mmcv import Config
from mmseg.models import build_segmentor


def main():
    cfg = Config.fromfile('local_configs/segman_trans/segman_b_trans10k_lass.py')
    model = build_segmentor(cfg.model)
    for name, path in [
        ('baseline', 'outputs/trans10k_segman_b/iter_80000.pth'),
        ('lass', 'outputs/trans10k_lass_mmscope/iter_80000.pth'),
    ]:
        sd = torch.load(path, map_location='cpu')
        sd = sd.get('state_dict', sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'=== {name} ===')
        print('missing', len(missing), 'unexpected', len(unexpected))
        dec_miss = [k for k in missing if k.startswith('decode_head')]
        dec_unexp = [k for k in unexpected if k.startswith('decode_head')]
        print('decode_head missing (first 5):', dec_miss[:5])
        print('decode_head unexpected:', dec_unexp[:5])
        # reload fresh model each time
        model = build_segmentor(cfg.model)


if __name__ == '__main__':
    main()
