#!/bin/bash
# After the post-test chain (test700 + MMAU + causal) finishes, search for a BIGGER
# label-free win: compare the dynamics-based variants on the test set to find the
# strongest selector, then run the best on the full 1700-item test set.
set -e
cd /livingrooms/khlu/undergraduate_research/Yu-Han/contra_decode/voxparadox/VoxParadox
SHIM=$PWD/_steer_shim
SCR=/livingrooms/khlu/undergraduate_research/Yu-Han/contra_decode/scripts
export PYTHONPATH="$SHIM:$SCR/steering:$SCR/ablation:$SCR/pipeline:$PWD"
export HF_HOME=/livingrooms/khlu/hf_cache
PY=/home/khlu/miniconda3/envs/omni/bin/python

echo "[bigger] waiting for post-test chain to finish..."
while ! grep -q "\[post\] DONE" methods_exp/out/posttest_chain.log 2>/dev/null; do sleep 60; done
sleep 20
echo "[bigger] variants comparison on test700"

$PY methods_exp/omni_perq_methods.py \
  --data_path data/voxparadox_test700.json --audio_base data \
  --output_dir methods_exp/out/test700_variants \
  --methods content,meanplustvar,meanplus2tvar,content_x_tvar,tvar_real \
  --ks 50 --alphas 16 --noise-seeds 4

echo "[bigger] DONE"
