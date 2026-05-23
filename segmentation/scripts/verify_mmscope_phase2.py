#!/usr/bin/env python
"""Phase-2 acceptance checks (steps 2.1-2.4, 2.3 decoder). Run from segmentation/."""
import sys


def test_21():
    from mmseg.models.modules.mmscope import BoundaryProbabilityModule
    import torch

    m = BoundaryProbabilityModule(180)
    x = torch.randn(2, 180, 64, 64)
    p_bd, w_bd, logits = m(x)
    assert p_bd.shape == (2, 1, 64, 64), p_bd.shape
    assert w_bd.shape == (2, 1, 64, 64), w_bd.shape
    print('2.1 BPM ok', p_bd.shape, w_bd.shape)


def test_22():
    import torch
    from mmseg.models.modules.mmscope import MultiScaleBoundaryEnhance

    if not torch.cuda.is_available():
        print('warn: skip 2.2 MSBEC (need CUDA)')
        return
    m = MultiScaleBoundaryEnhance(180).cuda()
    f_sem = torch.randn(2, 180, 64, 64, requires_grad=True, device='cuda')
    p_bd = torch.rand(2, 1, 64, 64, device='cuda')
    out = m(f_sem, p_bd)
    loss = out.sum()
    loss.backward()
    assert out.shape == f_sem.shape
    print('2.2 MSBEC ok', out.shape)


def test_24():
    import torch
    from mmseg.models.modules.mmscope import semantic_seg_to_boundary

    y = torch.zeros(2, 1, 64, 64, dtype=torch.long)
    y[:, :, 20:40, 20:40] = 3
    bd = semantic_seg_to_boundary(y, dilate_k=5, erode_k=5, ignore_index=255)
    y_pad = torch.full((2, 1, 64, 64), 255, dtype=torch.long)
    bd_pad = semantic_seg_to_boundary(y_pad, ignore_index=255)
    assert bd_pad.sum() == 0, 'padding 255 must not form boundary'
    assert bd.shape == (2, 1, 64, 64)
    assert bd.sum() > 0
    print('2.4 Y_bd ok', bd.sum().item())


def test_23():
    import torch
    from mmseg.models import build_head

    if not torch.cuda.is_available():
        print('warn: skip 2.3 decoder (need CUDA)')
        return
    cfg = dict(
        type='SegMANDecoderMMSCopE',
        in_channels=[96, 160, 364, 560],
        in_index=[0, 1, 2, 3],
        channels=180,
        feat_proj_dim=320,
        dropout_ratio=0.1,
        num_classes=12,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        mmscope_cfg=dict(boundary_loss_weight=0.4, refine_eta=0.1),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    )
    head = build_head(cfg).cuda().eval()
    feats = [
        torch.randn(1, 96, 128, 128, device='cuda'),
        torch.randn(1, 160, 64, 64, device='cuda'),
        torch.randn(1, 364, 32, 32, device='cuda'),
        torch.randn(1, 560, 16, 16, device='cuda'),
    ]
    with torch.no_grad():
        out = head(feats)
    print('2.3 decoder ok', out.shape)


def main():
    test_21()
    test_22()
    test_24()
    test_23()
    print('phase2 verify ok')


if __name__ == '__main__':
    main()
