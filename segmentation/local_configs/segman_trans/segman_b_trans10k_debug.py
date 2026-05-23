# Short Trans10K run to check mIoU trend before full 80k training.
# Usage:
#   python tools/train.py local_configs/segman_trans/segman_b_trans10k_debug.py \
#       --work-dir outputs/trans10k_debug

_base_ = ['./segman_b_trans10k.py']

# 2000 iters ~ quick sanity check on full train/val set
runner = dict(type='IterBasedRunner', max_iters=2000)

# 仅结束时 eval 一次，避免 val 1000 张导致 OOM/Killed；ckpt 仍每 500 存
checkpoint_config = dict(by_epoch=False, interval=500)
evaluation = dict(interval=2000, metric='mIoU', save_best='mIoU')

# Docker 默认 /dev/shm 仅 64MB，workers 过大易 Bus error；OOM 可改 0
data = dict(samples_per_gpu=2, workers_per_gpu=2)
