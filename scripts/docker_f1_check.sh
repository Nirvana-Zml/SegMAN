#!/usr/bin/env bash
set -euo pipefail
source /root/anaconda3/etc/profile.d/conda.sh
conda activate segman
python -c "import mmseg; print('mmseg', mmseg.__version__)"
python -c "import mmcv; print('mmcv', mmcv.__version__)"
python -c "import mmdet; print('mmdet', mmdet.__version__)" 2>/dev/null || echo "mmdet: NOT INSTALLED"
python -c "
import json
from pathlib import Path
for split in ('train','val'):
    p = Path('segmentation/data/trans10k/coco_instances') / f'{split}.json'
    d = json.loads(p.read_text())
    print(split, 'images', len(d['images']), 'anns', len(d['annotations']))
"
