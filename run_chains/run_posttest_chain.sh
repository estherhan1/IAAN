#!/bin/bash
# After the VoxParadox test700 run finishes, run the remaining GPU work serially:
#   1) MMAU diagnosis: IAAN (content) vs tvar/meanplustvar on 300 balanced MMAU items
#   2) fixed causal/attribution mask build + its eval (label-aware upper bound)
set -e
cd /livingrooms/khlu/undergraduate_research/Yu-Han/contra_decode/voxparadox/VoxParadox
SHIM=$PWD/_steer_shim
SCR=/livingrooms/khlu/undergraduate_research/Yu-Han/contra_decode/scripts
export PYTHONPATH="$SHIM:$SCR/steering:$SCR/ablation:$SCR/pipeline:$PWD"
export HF_HOME=/livingrooms/khlu/hf_cache
PY=/home/khlu/miniconda3/envs/omni/bin/python
MMAU=/livingrooms/khlu/undergraduate_research/Yu-Han/audio_caption/MMAU

echo "[post] waiting for test700 to finish..."
while [ ! -f methods_exp/out/test700/methods_summary.json ]; do sleep 60; done
sleep 20
echo "[post] test700 done -> MMAU diagnosis"

$PY methods_exp/omni_mmau_methods.py \
  --data_path $MMAU/mmau_sub300.json --audio_base $MMAU \
  --output_dir methods_exp/out/mmau300 \
  --methods content,tvar,meanplustvar --ks 50 --alphas 8,16

echo "[post] MMAU done -> causal mask build (fixed)"
$PY methods_exp/build_causal_mask.py \
  --data_path data/voxparadox_dev300.json --audio_base data \
  --out methods_exp/out/causal_mask.pt --limit 120

echo "[post] causal eval on VoxParadox dev300"
$PY methods_exp/omni_global_mask_steer.py \
  --data_path data/voxparadox_dev300.json --audio_base data \
  --masks methods_exp/out/causal_mask.pt \
  --output_dir methods_exp/out/btier_causal \
  --methods causal,causal_abs --ks 50,200 --alphas 16

echo "[post] DONE"
