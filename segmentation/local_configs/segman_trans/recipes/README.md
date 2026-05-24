# Route B training recipes

| File | Purpose |
|------|---------|
| `segman_b_trans10k_lass_fix5k.py` | **Frozen fix5k** (5k iter, lr 3e-5). Do not change for v2 experiments. |
| `../segman_b_trans10k_lass.py` | Shared LASS+MMSCopE architecture (fix5k & phase-3 base). **Do not edit** for balanced-v2. |
| `../segman_b_trans10k_lass_balanced.py` | balanced v1 (balanced10k archive). |
| `../segman_b_trans10k_lass_balanced_v2.py` | **balanced-v2** retrain (bowl-focused). |

**fix5k train (reproduce)**:

```bash
python tools/train.py local_configs/segman_trans/recipes/segman_b_trans10k_lass_fix5k.py \
  --work-dir outputs/trans10k_lass_mmscope_fix5k \
  --load-from outputs/trans10k_segman_b/iter_80000.pth \
  --no-validate
```

**balanced-v2 train**: see `scripts/train_route_b_balanced_v2.sh`
