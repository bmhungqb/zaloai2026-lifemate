#!/bin/bash
# Kaggle cell:  !git clone https://github.com/bmhungqb/zaloai2026-lifemate.git /kaggle/working/repo && bash /kaggle/working/repo/kaggle/run_kaggle.sh
set -e
cd /kaggle/working/repo
D=$(dirname $(dirname $(find /kaggle/input -path '*train/train.jsonl' | head -1)))
ln -sfn "$D/train" train; ln -sfn "$D/public_test" public_test
ls train | head
python -m lifemate.run asr --split train --n 300 --out outputs
python -m lifemate.run agent --split train --n 300 --mode oracle --shots 0 --backend vllm --out outputs --tag z0
python -m lifemate.run agent --split train --n 300 --mode oracle --shots 4 --backend vllm --out outputs --tag s4
python -m lifemate.run agent --split train --n 300 --mode asr --shots 4 --backend vllm --out outputs --tag asr4
