#!/usr/bin/env python
"""Phase-1 acceptance checks (steps 1.1-1.4). Run from segmentation/ with segman env."""
import sys


def test_11_12():
    from mmseg.models.modules.ltab import LTAB
    from mmseg.models.modules.rsm import ReflectionSuppression
    import torch

    x = torch.randn(2, 64, 64, 64)
    assert LTAB(64)(x).shape == x.shape
    m = torch.zeros(2, 1, 64, 64)
    m[:, :, :20, :] = 1.0
    assert ReflectionSuppression(64)(x, m).shape == x.shape
    print('1.1-1.2 ok')


def test_14():
    from mmseg.models import build_backbone
    import torch

    cfg = dict(
        type='SegMANEncoderLASS_b',
        pretrained=None,
        style='pytorch',
        lass_cfg=dict(enable_stages=[0, 1, 2]),
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print('warn: VSSM triton scan needs CUDA; skipping 1.4 on CPU')
        return
    model = build_backbone(cfg).to(device)
    model.eval()
    x = torch.randn(1, 3, 512, 512, device=device)
    with torch.no_grad():
        outs = model(x)
    shapes = [tuple(o.shape) for o in outs]
    chans = [o.shape[1] for o in outs]
    assert len(outs) == 4, shapes
    assert chans == [96, 160, 364, 560], chans
    print('1.4 shapes:', shapes)


def main():
    test_11_12()
    test_14()
    print('phase1 verify ok')


if __name__ == '__main__':
    main()
