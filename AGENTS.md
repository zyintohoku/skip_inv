# Agent Quick Start

This repository is an experimental Stable Diffusion inversion workspace. Most experiment implementations live in `experiments/`; root-level Python files with the same names are compatibility wrappers for old shell/Slurm entrypoints. Shell launchers live in `scripts/`, generated artifacts live in `outputs/`, and analysis artifacts live in `results/`.

Read these first when starting a new session:

- `docs/CODE_RETRIEVAL_INDEX.md`: task-oriented map for finding the right code path quickly.
- `docs/SCRIPT_STYLE_GUIDE.md`: shell/Slurm path and node-slot conventions.
- `PY_FILES_OVERVIEW.md`: one-line purpose for every `.py` file currently in the repo.
- `scripts/README.md`: older scripts overview; prefer `docs/CODE_RETRIEVAL_INDEX.md` when the two differ.

## Main Code Paths

| Need | Start Here | Core Logic |
| --- | --- | --- |
| SD v1.4 generation + FPI/AFPI/AIDI inversion | `experiments/run.py`, `experiments/run_batch.py`, `experiments/run_single_sample.py` | `utils/inv_methods.py` |
| PIE real-image VAE encode -> invert -> reconstruct | `scripts/run_pie_image_inv_rec.sh`, `experiments/run_pie_image_inv_rec.py` | `utils/inv_methods.py`, `experiments/prompt_pressure_pipeline.py` |
| Prompt pressure traces and diagnostics | `experiments/prompt_pressure_pipeline.py`, `experiments/prompt_pressure_saved_latents.py` | `analysis/plot_prompt_pressure_*.py`, `analysis/compute_prompt_pressure_distribution_metrics.py` |
| Saved-latent FPI and seed studies | `experiments/run_fpi_from_saved_latents.py`, `experiments/run_with_generation_trace.py` | `analysis/compute_fpi_gs7_seed_psnr.py` |
| SLERP experiments | `experiments/run_slerp_seed_pair_fpi.py` | `analysis/make_slerp_*.py`, `experiments/generate_uncond_slerp_initial_latents.py` |
| SDXL Euler experiments | `experiments/run_sdxl_euler_prompt_grid_gen_inv_rec.py` | `utils/sdxl_euler_inv_methods.py`, `analysis/compute_sdxl_euler_prompt_grid_psnr.py` |
| Skip inversion with adaptive CFG schedule | `experiments/skip_inv.py` | `utils/skip_pipe.py` |
| Prompt-to-Prompt editing | `p2p/p2p.py` | `p2p/ptp_utils.py`, `p2p/seq_aligner.py` |

## Search Recipes

Use `rg` first:

```bash
rg -n "class Inversion|def invert|def __call__" utils experiments
rg -n "run_pie_image_inv_rec|pie_image_inv_rec|encode_image_to_latent" .
rg -n "prompt_pressure|guidance_delta|bending_B_t|relative_pressure" .
rg -n "slerp|alpha_label|parse_alphas" .
rg -n "compute_psnr|input_rec_psnr|gen_rec" analysis run*.py
rg -n "sbatch|--wrap|python " scripts
```

## Repository Hygiene

- Avoid moving code until imports and Slurm wrappers are updated together.
- Root-level `run*.py`, `generate*.py`, `prompt_pressure*.py`, `skip_inv.py`, and `test_*.py` are wrappers; edit the real implementation in `experiments/`.
- Treat `outputs/`, `results/`, and `log/` as generated/experiment data unless the user asks to reorganize them.
- Many scripts assume they are run from repo root.
- Slurm scripts should default to `PROJECT_DIR=$PWD` and use `--chdir="$PROJECT_DIR"` for submitted jobs.
