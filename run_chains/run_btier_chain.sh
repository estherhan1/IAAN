#!/bin/bash
# Wait for the A-tier run to finish, then run all B-tier (global-mask) evals on the
# single shared GPU: build the causal/attribution mask, then evaluate every global
# method (discriminative, probe, probe_attr, omp, causal) with the official protocol.
set -e
cd /livingrooms/khlu/undergraduate_research/Yu-Han/contra_decode/voxparadox/VoxParadox
SHIM=$PWD/_steer_shim
SCR=/livingrooms/khlu/undergraduate_research/Yu-Han/contra_decode/scripts
export PYTHONPATH="$SHIM:$SCR/steering:$SCR/ablation:$SCR/pipeline:$PWD"
export HF_HOME=/livingrooms/khlu/hf_cache
PY=/home/khlu/miniconda3/envs/omni/bin/python

echo "[chain] waiting for A-tier to finish..."
while [ ! -f methods_exp/out/atier/methods_summary.json ]; do sleep 60; done
# give the A-tier process a moment to release the GPU
sleep 30
echo "[chain] A-tier done; building causal mask"

$PY methods_exp/build_causal_mask.py \
  --data_path data/voxparadox_dev300.json --audio_base data \
  --out methods_exp/out/causal_mask.pt --limit 120

echo "[chain] B-tier eval: global_masks (discriminative/probe/probe_attr/omp)"
$PY methods_exp/omni_global_mask_steer.py \
  --data_path data/voxparadox_dev300.json --audio_base data \
  --masks methods_exp/out/global_masks.pt \
  --output_dir methods_exp/out/btier_global \
  --methods discriminative,probe,probe_attr,omp \
  --ks 50,200 --alphas 12

echo "[chain] B-tier eval: causal mask"
$PY methods_exp/omni_global_mask_steer.py \
  --data_path data/voxparadox_dev300.json --audio_base data \
  --masks methods_exp/out/causal_mask.pt \
  --output_dir methods_exp/out/btier_causal \
  --methods causal,causal_abs \
  --ks 50,200 --alphas 12

echo "[chain] DONE"
