# IAAN SLT 2026 — paper-aligned reproduction work

**Status: partial implementation, NOT an independently validated reproduction.**
This directory separates the published IAAN method from the later Steer-CD work
in the repository root. The original files are unchanged.

Paper: *Enhancing How Large Audio-Language Models Hear: Identifying and
Amplifying Acoustic Neurons in the Audio Encoder* (IEEE SLT 2026).

## Exact algorithm (paper Sec. III)

For each audio clip independently:

1. Run the audio encoder on real waveform `x` and a fixed-seed, length- and
   moment-matched Gaussian noise reference `r`. Capture each encoder block's
   **input to its second FFN projection** (`fc2`), after the activation function.
2. Compute signed scores `mean_t(a_real) - mean_t(a_reference)` per neuron.
3. Rank scores **globally across encoder layers**, select top `K`.
4. On a new forward pass of the **real audio**, multiply only those FFN
   neuron activations by `g > 1`. Do not modify LLM backbone activations.
5. Use the model's *original greedy generation and answer parsing*.

`iaan_core.py` implements reference generation, layer pre-hook activation
collection, global signed top-K selection, and temporary amplification hooks.
The module is adapter-independent and **is not** a ready-to-run inference CLI.

### Quick CPU unit tests

```bash
pip install torch
cd paper_reproduction
python -m unittest discover -s tests -v
```

## Original model code already present in this repo

| Model | Available implementation | What is still missing |
|---|---|---|
| Qwen2.5-Omni | `../1_iaan_baselines/selectors/omni_perq_methods.py` (`content` selector) | External `contra_decode` modules, `run_perquestion_steering`, `omni_eval_voxparadox`, official dataset/eval, environment |
| Audio-Flamingo-3 | `../2_steer_cd/af3/af3_steer_cd_collect.py` (`content` variant) | Original `af3_perq_official.py` greedy-generate-and-parse pipeline; file here instead collects Steer-CD letter logits |
| Kimi-Audio | `../2_steer_cd/kimi/kimi_steer_cd_collect.py` (`content` variant) | Original `kimi_perq_official.py` greedy-generate-and-parse pipeline, portable `kimia_infer` setup; currently collects letter logits |
| AF3-PD | No complete original IAAN inference script found | Adapter and original model-specific evaluation |

**Do not** compare letter-logit Steer-CD results in `2_steer_cd/` to the
paper's greedy-generation IAAN Table I. Results stored in this repository
also use dev300/test700 subsets from later experiments.

## Evaluation contract from SLT 2026

- **Selection set:** LISTEN, 190 sarcasm speech clips reconstructed into MCQs.
- **Tune:** neuron budget `K` and gain `g` to maximize `Acc - ALA`
  on LISTEN, *not* on VoxParadox.
- **Test:** all ten VoxParadox attributes, 200 MCQs per attribute (2,000
  total); evaluate accuracy, adversarial-label agreement (ALA), and per-task
  scores using model-specific official greedy-generation protocols.
- **Published main averages (baseline → IAAN):** AF3 13.1 → 38.8;
  Qwen 7.6 → 29.0; Kimi 21.2 → 30.9; AF3-PD 70.5 → 75.0.
- **Important:** the published paper does not imply that any preexisting
  `dev300`/`test700` result here is equivalent to the full paper evaluation.

## Before claiming a reproduction

1. Recover the original `*_perq_official.py` model pipelines and the
   missing `contra_decode` modules or reimplement and audit those adapters.
2. Pin actual tested model revisions, package versions, prompt templates,
   random seeds, waveform preprocessing and sample-ordering conventions.
3. Obtain permitted VoxParadox and LISTEN data and record their checksums;
   no benchmark audio or model weights are bundled here.
4. Implement/test the official answer parser, greedy generation, the ten
   task aggregation, ALA, and the LISTEN hyperparameter sweep.
5. Run all models and compare every Table I task to the paper. Document
   deviations, not just an average.

## Integration example (pseudocode for a model adapter)

```python
# layers = [block.fc2 for block in model.audio_encoder.layers]
# real = collect_means(lambda: encode_audio(real_wave), layers)
# noise = gaussian_reference(real_wave, seed=clip_seed)
# ref  = collect_means(lambda: encode_audio(noise), layers)
# selected = select_topk(real, ref, k=50)
# with amplify_fc2_inputs(layers, selected, gain=16):
#     answer = original_model_greedy_generate(real_wave, original_prompt)
```

These lines must be adapted to model-specific audio preprocessors and
encoder access. For padded features, mask padded time steps when pooling.
