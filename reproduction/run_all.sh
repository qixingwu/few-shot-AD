#!/bin/bash
# Run the upstream algorithms for MVTec AD and VisA, 1-shot, round 0.
set -euo pipefail
cd /data/wuqixing/UniVAD
export PATH=/data/wuqixing/conda_envs/univad/bin:$PATH
export TORCH_HOME=$PWD/cache/torch
export HF_HOME=$PWD/cache/huggingface
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg PYTHONUNBUFFERED=1
exec 9>reproduction/run.lock
flock -n 9 || exit 2
python -m pip freeze > reproduction/environment.freeze.txt
for dataset in mvtec visa; do
  # Call the author's segmentation function unchanged. Only the selected
  # reference and all test images are consumed by the 1-shot evaluation.
  python - "$dataset" > "reproduction/logs/segment_${dataset}.log" 2>&1 <<'PY'
import sys,json
from pathlib import Path
import cv2,yaml,torch
from models.component_segmentaion import grounding_segmentation
name=sys.argv[1]
root='./data/mvtec' if name=='mvtec' else './data/VisA_pytorch/1cls'
meta=json.loads(Path(root,'meta.json').read_text())
torch.set_num_threads(4)
for category,items in meta['test'].items():
    digits=3 if name=='mvtec' or category in ['capsules','cashew','chewinggum','fryum','pipe_fryum'] else 4
    ext='.png' if name=='mvtec' else '.JPG'
    ref=f'{root}/{category}/train/good/{0:0{digits}d}{ext}'
    paths=[ref]+[f"{root}/{x['img_path']}" for x in items]
    save='./masks/'+root.split('/data/')[-1]+'/'+category
    pending=[]
    for path in paths:
        mask=Path(save,'/'.join(str(Path(path).with_suffix('')).split('/')[-3:]),'grounding_mask.png')
        if not mask.is_file() or cv2.imread(str(mask),0) is None: pending.append(path)
    print(f'{name}/{category}: {len(pending)} remaining',flush=True)
    if pending:
        config=yaml.safe_load(Path(f'configs/class_histogram/{category}.yaml').read_text())
        grounding_segmentation(pending,save,config['grounding_config'])
    print(f'COMPLETE {name}/{category}',flush=True)
PY
  root=./data/mvtec
  if [ "$dataset" = visa ]; then root=./data/VisA_pytorch/1cls; fi
  python test_univad.py --dataset "$dataset" --data_path "$root" --round 0 --image_size 448 --k_shot 1 2>&1 | tee "reproduction/logs/test_${dataset}.log"
done
echo 'MVTec AD and VisA evaluations completed.'
