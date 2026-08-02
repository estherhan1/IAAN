# run_chains — serial-GPU orchestration scripts

These four shell scripts chained the experiments on the single shared RTX 3090: each one
waits for the previous stage's summary file / log marker, then launches the next run.
They span folders `1_` and `3_`, which is why they live here rather than inside one
category. Each has its driver log alongside it.

| Script | Waits for | Then produces | Lands in |
|---|---|---|---|
| `run_btier_chain.sh` | A-tier `atier/methods_summary.json` | causal mask + all global/causal mask evals | `1_iaan_baselines/global_causal_masks/` |
| `run_refine_chain.sh` | B-tier `[chain] DONE` | finer K/α selector refinement sweep | `1_iaan_baselines/selectors/results/refine` |
| `run_posttest_chain.sh` | `test700/methods_summary.json` | MMAU diagnosis + causal mask + causal eval | `3_mmau_control/`, `1_.../global_causal_masks/` |
| `run_biggerwin_chain.sh` | post-test `[post] DONE` | hybrid-selector comparison on test700 | `1_iaan_baselines/selectors/results/test700_variants` |

Logs: `btier_chain.log`, `refine_chain.log`, `posttest_chain.log`, `biggerwin_chain.log`.

All assume the repo root as cwd and the `omni` env with the shim on `PYTHONPATH` (the
header of each script sets `SHIM`, `SCR`, `PYTHONPATH`, `HF_HOME`, `PY`). The Steer-CD
collection runs (folder `2_`) were launched directly, not through these chains.
