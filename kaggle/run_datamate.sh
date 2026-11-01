#!/bin/bash
# Kaggle (2xT4): attach private dataset containing datamate train/ (and public_test/) folders, then:
#   !git clone https://github.com/bmhungqb/zaloai2026-lifemate.git /kaggle/working/repo && bash /kaggle/working/repo/kaggle/run_datamate.sh
set -e
cd /kaggle/working/repo
D=$(dirname $(dirname $(find /kaggle/input -path '*train/questions.jsonl' | head -1)))
mkdir -p data/datamate; ln -sfn "$D/train" data/datamate/train; ln -sfn "$D/public_test" data/datamate/public_test
python -m datamate.run solve --split train --n 150 --K 3 --backend vllm --model Qwen/Qwen3-8B --tp 2 --cache /kaggle/working/dm_cache --out outputs
