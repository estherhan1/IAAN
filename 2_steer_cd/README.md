# 2_steer_cd — the Steer-CD breakthrough, on three models

Steer-CD keeps the IAAN encoder steering but replaces the readout with a contrast
between the steered and the unsteered model, on the **same real audio**:

```
answer = argmax_L  [ logit_steered(L)  -  λ · logit_unsteered(L) ]
```

The steered model is the expert (audio evidence amplified); the same model unsteered is
the amateur (≈ pure language prior). Subtracting the amateur removes the residual prior
that steering alone cannot suppress. λ is a single scalar chosen on dev300; it sits on a
broad plateau. Both components are necessary: CD without steering fails (the unsteered
audio signal is too weak), steering without CD saturates at ~34 (folder `1_`).

Two readout implementations, same idea:
- **letter-logit readout** — one prefill per variant, read the four answer-letter
  logits at the last position, combine offline. This is how all three models are
  evaluated (`collect_letter_logits*.py` + `eval_cd_letters.py`), because it is cheap
  and lets any λ / expert be swept without re-running the model.
- **per-token generation** — `omni_perq_tokencd.py`, dual KV-cache branches, contrast at
  every generated token with an adaptive-plausibility mask. This proves the win holds
  under the real generate+parse protocol, not just the letter readout.

Held-out **test700**, dev-chosen operating point, each model steered on its own audio
encoder:

| Model | folder | baseline | steer-only | **Steer-CD** | ALA base → CD | op |
|---|---|---|---|---|---|---|
| Qwen2.5-Omni-7B | `qwen/` | 9.86 | 34.43 | **49.43** | 77 → 13 | meanplustvar `K100 a16`, λ0.75 |
| Audio-Flamingo-3-7B | `af3/` | 15.00 | 32.00 | **48.00** | 67 → 10 | meanplustvar `K50 a8`, λ1.0 |
| Kimi-Audio-7B | `kimi/` | 20.29 | 30.43 | **32.71** | 66 → 27 | content `K200 a12`, λ0.5 |

All numbers reproduced by running `eval_cd_letters.py` on the `cll_*_test700.jsonl`
files in each sub-folder (offline, no GPU).

---

All Qwen scripts (`collect_letter_logits.py`, `eval_cd_letters.py`, `eval_letter_cd.py`,
`omni_perq_tokencd.py`) run in the **`omni`** conda env with the shim on `PYTHONPATH`
(see the top-level README). `eval_cd_letters.py` is pure-offline and needs no GPU.

## qwen/

| Script | Purpose |
|---|---|
| `collect_letter_logits.py` | Collect the four answer-letter prefill logits for `real` / `blank` / `noise` and any steered config on Qwen2.5-Omni. Writes `cll_*.jsonl`. |
| `eval_cd_letters.py` | Generic offline Steer-CD evaluator: for each expert variant and each λ, `argmax_L[expert−λ·amateur]`, scored gt/ALA with optional per-task table. Works on any model's `cll_*.jsonl`. |
| `eval_letter_cd.py` | Earlier CD-readout evaluator over a `confpick` dir (ensemble / contrastive variants). |
| `omni_perq_tokencd.py` | Per-token contrastive generation (dual KV-cache, adaptive plausibility). The generate+parse-protocol version. |

Results (`results/`):
- `cll_test700.jsonl` — real/blank/noise + meanplustvar scale K50a16 / K100a16. The main test-set letter logits.
- `cll_dev300_audioablate.jsonl` — dev300, includes the audio-ablation variants that show CD-without-steering fails.
- `cll_mmsu_dev600.jsonl` — MMSU_para dev-transfer probe (external λ tuning; the no-conflict set picks λ=0, so λ cannot be tuned there).
- `tokencd_dev300/`, `tokencd_dev300_plaus/`, `tokencd_test700/` — per-token CD. dev300 λ0.75 `p0.1` = 52.67; **test700 λ0.75 `p0.1` = 48.71** (within 0.7 pp of the readout).

Verify the headline: `$PY eval_cd_letters.py --letters methods_exp/out/cll_test700.jsonl --dataset data/voxparadox_test700.json --amateur real --experts meanplustvar_scale_K100_a16,meanplustvar_scale_K50_a16 --lams 0,0.5,0.75,1`

Per-token re-run: `$PY omni_perq_tokencd.py --data_path data/voxparadox_test700.json --audio_base data --output_dir methods_exp/out/tokencd_test700 --expert meanplustvar:scale:50:16 --lams 0.75 --plaus 0.1`

## af3/

`af3_steer_cd_collect.py` (run in the **`af3`** conda env, from the repo root). AF3
answers as `(A) ...`, so it forces a `(` token and reads the letter logits at the next
position. Selection on `audio_tower` fc2, per-clip real-vs-noise, z←α on top-K. Results:
`cll_af3_dev300.jsonl`, `cll_af3_test700.jsonl`. On test700 the meanplustvar `K50 a8`
expert with λ1.0 gives 48.00 (steer-only 32.00, baseline 15.00).

Re-run: `python af3_steer_cd_collect.py --data-path data/voxparadox_test700.json --audio-base data --out methods_exp/out/cll_af3_test700.jsonl --variants real,content:50:8,meanplustvar:50:8`

## kimi/

`kimi_steer_cd_collect.py` (run in the **`kimi`** conda env, `HF_HUB_OFFLINE=1`). Kimi
answers with a bare letter, so the first text token is the answer; one `alm.forward`
gives the letter distribution at the last position. Selection on the Whisper
`speech_encoder` fc2. Note the TikToken signature `tok.encode(s, bos=False, eos=False)`
(not `add_special_tokens=`). Results: `cll_kimi_dev300.jsonl`, `cll_kimi_test700.jsonl`.
On test700 the content `K200 a12` expert with λ0.5 gives 32.71 (steer-only 30.43,
baseline 20.29) — Kimi is the least pathological of the three, so CD mostly trades ALA
for spread.

Re-run: `HF_HOME=/livingrooms/khlu/hf_cache HF_HUB_OFFLINE=1 /home/khlu/miniconda3/envs/kimi/bin/python kimi_steer_cd_collect.py --data_path data/voxparadox_test700.json --audio_base data --out methods_exp/out/cll_kimi_test700.jsonl`
