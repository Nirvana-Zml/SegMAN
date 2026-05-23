# 2-class: background vs any transparent region (optional simpler baseline)
_base_ = ['./segman_b_trans10k.py']

model = dict(
    decode_head=dict(num_classes=2))
