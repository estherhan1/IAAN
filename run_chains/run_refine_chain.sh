#!/bin/bash
# Wait for the B-tier chain to finish, then run the refinement A-tier sweep:
# dynamics-based + magnitude-preserving hybrids at finer K/alpha, to push past the
# current +3.67 improvement of tvar/meanplustvar over content.
set -e
cd /livingrooms/khlu/undergraduate_research/Yu-Han/contra_decode/voxparadox/VoxParadox
SHIM=$PWD/_steer_shim
SCR=/livingrooms/khlu/undergraduate_research/Yu-Han/contra_decode/scripts
export PYTHONPATH="$SHIM:$SCR/steering:$SCR/ablation:$SCR/pipeline:$PWD"
export HF_HOME=/livingrooms/khlu/hf_cache
PY=/home/khlu/miniconda3/envs/omni/bin/python

echo "[refine] waiting for B-tier chain to finish..."
while ! grep -q "\[chain\] DONE" methods_exp/out/btier_chain.log 2>/dev/null; do sleep 60; done
sleep 30
echo "[refine] B-tier done; launching refinement sweep"

$PY methods_exp/omni_perq_methods.py \
  --data_path data/voxparadox_dev300.json --audio_base data \
  --output_dir methods_exp/out/refine \
  --methods content,tvar,meanplustvar,meanplus2tvar,tvar_real,content_x_tvar \
  --ks 30,50 --alphas 16,24 --noise-seeds 4

echo "[refine] DONE"
