# Code Retrieval Index

This document is for fast retrieval in a fresh agent session. It is organized by task rather than by filename.

For a complete one-line list of every Python file, see `../PY_FILES_OVERVIEW.md`.

## Directory Roles

| Path | Role |
| --- | --- |
| `experiments/` | Main experiment entrypoints and active research scripts moved from repo root. |
| repo root `*.py` | Compatibility wrappers for old shell/Slurm commands; do not edit core logic here. |
| `scripts/` | Shell/Slurm launchers and a few small Python helper scripts. These usually call root-level Python files. |
| `utils/` | Reusable diffusion pipeline and inversion implementations. |
| `analysis/` | Metrics, aggregation, plotting, exports, and post-hoc analysis. |
| `p2p/` | Prompt-to-Prompt editing implementation. |
| `outputs/` | Generated images, latent `.pt` files, traces, and run outputs. |
| `results/` | Aggregated metrics, plots, tables, reports, and exported analysis data. |
| `PIE_bench/` | Dataset mapping and source images. |
| `log/` | Slurm stdout/stderr logs. |

## Core Implementations

### Stable Diffusion v1.4 inversion

Start with `utils/inv_methods.py`.

Important symbols:

- `MyStableDiffusionPipeline`: custom SD pipeline returning both PIL image and final latent.
- `Inversion`: DDIM/FPI/AFPI/AIDI inversion logic.
- `encode_image_to_latent`: VAE encode real images into SD latent space.
- `preprocess_image_for_vae`: image resize/normalization helper.

Common callers:

- `experiments/run.py`, `experiments/run_batch.py`: generate -> invert -> reconstruct over PIE prompts.
- `experiments/run_single_sample.py`: focused single-sample debug.
- `experiments/run_fpi_from_saved_latents.py`: rerun FPI from saved generated latents.
- `experiments/run_prompt_ablation_fpi_gen_inv_rec.py`: prompt-list/grid FPI experiments.
- `experiments/run_pie_image_inv_rec.py`: real PIE image VAE latent inversion/reconstruction.
- `experiments/run_slerp_seed_pair_fpi.py`: seed-pair SLERP + FPI.

Useful search:

```bash
rg -n "class Inversion|def invert|def prev_step|def next_step|class MyStableDiffusionPipeline|encode_image_to_latent" utils/inv_methods.py
rg -n "from utils.inv_methods" -g "*.py" .
```

### Prompt pressure diagnostics

Start with `experiments/prompt_pressure_pipeline.py`.

Important symbols:

- `run_prompt_pressure_trace`: records guided/unconditional trajectories, noise predictions, pressure metrics.
- `ddim_model_output_coeff`: coefficient used to convert guidance delta to latent movement scale.
- `make_init_latent`, `parse_seeds`, `resolve_prompt`, `make_scheduler`: shared helpers used by many experiments.
- Output metrics include `guidance_delta_l2`, `prompt_pressure_P_t`, `relative_pressure_R_t`, and `bending_B_t`.

Common callers:

- `experiments/prompt_pressure_saved_latents.py`
- `experiments/generate_aidi_gs7_seed_pressure.py`
- `experiments/cfg_directional_features_saved_latents.py`
- `experiments/latent_perturbation_saved_latents.py`
- `experiments/run_with_generation_trace.py`

Analysis files:

- `analysis/compute_prompt_pressure_distribution_metrics.py`
- `analysis/plot_prompt_pressure_by_seed.py`
- `analysis/plot_prompt_pressure_by_psnr.py`
- `analysis/plot_prompt_pressure_normalized.py`
- `analysis/plot_guidance_delta_l2.py`
- `analysis/plot_latent_step_lengths.py`

Useful search:

```bash
rg -n "prompt_pressure|run_prompt_pressure_trace|guidance_delta_l2|relative_pressure_R_t|bending_B_t" .
```

### PIE real-image inversion/reconstruction

Launcher: `scripts/run_pie_image_inv_rec.sh`

Python entrypoint: `experiments/run_pie_image_inv_rec.py`

Compatibility entrypoint: `run_pie_image_inv_rec.py`

Flow:

1. Select records by `--image_dir`, `--keys_csv`, `--mapping_keys`, or `--sample_ids`.
2. Resolve prompt from `PIE_bench/mapping_file.json`.
3. Encode source image with `encode_image_to_latent`.
4. Decode VAE baseline image.
5. Invert image latent with `Inversion.invert`.
6. Reconstruct from inverted latent with `MyStableDiffusionPipeline`.
7. Save images, `image_latents.pt`, `inv_latents.pt`, `rec_latents.pt`, per-sample CSV, JSONL traces, and summary JSON.

Important functions:

- `select_records`
- `load_image_dir_records`
- `load_key_csv_records`
- `image_mse_psnr`
- `main`

Typical run:

```bash
bash scripts/run_pie_image_inv_rec.sh yagi35
MAPPING_KEYS=311000000008 OUTPUT=outputs/pie_key311_image_inv_rec bash scripts/run_pie_image_inv_rec.sh yagi35
KEYS_CSV=results/fpi_gs7_seed_psnr/prompt_psnr_best30.csv bash scripts/run_pie_image_inv_rec.sh yagi35
```

Useful search:

```bash
rg -n "pie_image_inv_rec|select_records|image_latents|input_rec_psnr|encode_image_to_latent" .
```

### SLERP experiments

Start with `experiments/run_slerp_seed_pair_fpi.py`.

Important symbols:

- `slerp_latent`
- `parse_alphas`
- `alpha_label`
- `cast_for_save`
- `load_tensor_list`

Related scripts:

- `scripts/run_slerp_seed_pair_fpi.sh`
- `scripts/run_slerp_seed_pair_fpi_second_sensitive.sh`
- `scripts/run_uncond_slerp_initial_latents.sh`
- `experiments/generate_uncond_slerp_initial_latents.py`
- `experiments/generate_selected_inv_latent_uncond.py`
- `experiments/run_sample692_seed1_seed4_inv_slerp.py`

Analysis files:

- `analysis/make_slerp_metric_tables.py`
- `analysis/make_slerp_divergence_tables.py`
- `analysis/make_slerp_gen_rec_error_montage.py`
- `analysis/plot_slerp_inversion_loss_curves.py`

Useful search:

```bash
rg -n "slerp|parse_alphas|alpha_label|seed_pair|interpolation" .
```

### Prompt grid and prompt ablation

Core runner: `experiments/run_prompt_ablation_fpi_gen_inv_rec.py`

Aggregation:

- `analysis/compute_prompt_grid_fpi_psnr.py`
- `analysis/compute_id185_308_ablation_fpi_psnr.py`
- `analysis/compute_sdxl_euler_prompt_grid_psnr.py` for SDXL runs.

Plotting:

- `analysis/plot_modifier_prompt_grid_heatmap.py`
- `analysis/plot_puppy_dog_field_heatmap.py`
- `analysis/plot_psnr_threshold_percent_heatmaps.py`
- `analysis/analyze_prompt_structure_psnr.py`
- `analysis/analyze_paraphrase_prompt_grid.py`

Launchers:

- `scripts/run_id185_308_ablation_fpi_gen_inv_rec.sh`
- `scripts/run_modifier_prompt_grid_fpi_gen_inv_rec.sh`
- `scripts/run_modifier_extra_context_fpi_gen_inv_rec.sh`
- `scripts/run_paraphrase_prompt_grid_fpi_gen_inv_rec.sh`
- `scripts/run_puppy_dog_field_fpi_gen_inv_rec.sh`

Useful search:

```bash
rg -n "prompt_csv|prompt_grid|modifier|paraphrase|puppy|dog|field" .
```

### SDXL Euler experiments

Start with:

- `experiments/run_sdxl_euler_prompt_grid_gen_inv_rec.py`
- `utils/sdxl_euler_inv_methods.py`

Launcher:

- `scripts/run_sdxl_euler_prompt_grid_gen_inv_rec.sh`

Aggregation:

- `analysis/compute_sdxl_euler_prompt_grid_psnr.py`

Temporary tests:

- `tmp_sdxl_base_test/sdxl_euler_inversion_reconstruct.py`
- `tmp_sdxl_base_test/test_sdxl_base_only.py`

Useful search:

```bash
rg -n "SDXL|Euler|StableDiffusionXLPipeline|sdxl_euler" .
```

### Skip inversion and adaptive CFG schedule

Start with:

- `experiments/skip_inv.py`
- `utils/skip_pipe.py`

Launch/test:

- `scripts/run_skip_inv_ablation.sh`
- `test_skip_inv.py`

Analysis:

- `analysis/analyze_skip_inv_results.py`
- `analysis/plot_skip_inv_distribution.py`

Useful search:

```bash
rg -n "cfg_schedule|reset_gs|force_converge_before_step|skip_inv" .
```

### Prompt-to-Prompt editing

Start with `p2p/p2p.py`.

Support files:

- `p2p/ptp_utils.py`
- `p2p/seq_aligner.py`

Launchers:

- `scripts/run_p2p_aidi_all.sh`
- `scripts/run_p2p_afpi_ldt_all.sh`
- `scripts/run_p2p_skip_inv_all.sh`
- `scripts/run_p2p_upper_bound.sh`

Evaluation:

- `analysis/evaluate_p2p.py`
- `analysis/plot_p2p_results.py`
- `analysis/plot_p2p_clip_psnr.py`

Useful search:

```bash
rg -n "AttentionControl|AttentionReplace|AttentionRefine|LocalBlend|p2p|Prompt-to-Prompt" p2p analysis scripts
```

## Script Categories

| Category | Scripts |
| --- | --- |
| Baseline AIDI/AFPI/FPI runs | `scripts/run.sh`, `scripts/run_afpi_ldt.sh`, `scripts/run_batch.sh`, `scripts/run_aidi_135.sh`, `scripts/run_gs7_seed_1_10.sh`, `scripts/test_ablation.sh` |
| Saved-latent and seed studies | `scripts/run_fpi_gs7_seed_from_saved_latents.sh`, `scripts/run_fpi_gs7_seed_generation_trace.sh`, `scripts/run_aidi_gs7_seed_generation_pressure.sh`, `scripts/run_cfg7_rec_latent_refpi.sh` |
| Prompt pressure | `scripts/run_prompt_pressure_top10.sh`, `scripts/run_prompt_pressure_seed_sensitive_top10.sh`, `scripts/run_prompt_pressure_best_worst_top10_saved_latents.sh`, `scripts/analyze_prompt_pressure_seed_sensitive_top10.sh`, `scripts/analyze_prompt_pressure_best_worst_top10_saved_latents.sh`, `scripts/plot_pressure_vs_psnr_saved_latent_top10_all.sh` |
| Prompt grid/ablation | `scripts/run_id185_308_ablation_fpi_gen_inv_rec.sh`, `scripts/run_modifier_prompt_grid_fpi_gen_inv_rec.sh`, `scripts/run_modifier_extra_context_fpi_gen_inv_rec.sh`, `scripts/run_paraphrase_prompt_grid_fpi_gen_inv_rec.sh`, `scripts/run_puppy_dog_field_fpi_gen_inv_rec.sh` |
| PIE image inversion | `scripts/run_pie_image_inv_rec.sh` |
| SLERP | `scripts/run_slerp_seed_pair_fpi.sh`, `scripts/run_slerp_seed_pair_fpi_second_sensitive.sh`, `scripts/run_uncond_slerp_initial_latents.sh`, `scripts/run_selected_inv_latent_uncond_generation.sh`, `scripts/run_selected_inv_latent_best_prompt_generation.sh` |
| SDXL | `scripts/run_sdxl_euler_prompt_grid_gen_inv_rec.sh` |
| P2P editing | `scripts/run_p2p_aidi_all.sh`, `scripts/run_p2p_afpi_ldt_all.sh`, `scripts/run_p2p_skip_inv_all.sh`, `scripts/run_p2p_upper_bound.sh` |
| Node/environment helpers | `scripts/check_yagi_nodes.sh`, `scripts/collect_node_info*.sh`, `scripts/node_info_quick_ref.sh`, `scripts/collect_test_aidi_results.sh` |

To see exact Python calls in shell launchers:

```bash
rg -n "python |python3 |sbatch|--wrap" scripts -g "*.sh" -g "*.sbatch"
```

## Output Conventions

Common latent names:

- `init_latents.pt`: initial random latent.
- `gen_latents.pt`: latent after forward generation.
- `inv_latents.pt`: inverted latent.
- `rec_latents.pt`: latent after reconstruction.
- `image_latents.pt`: VAE-encoded real image latent, mainly from PIE image runs.
- `convergence_losses*.json`: inversion loss curves.
- `trace_tensors.pt`, `*_traces.jsonl`: prompt pressure or inversion traces.

Common metric names:

- `init_inv`: initial latent vs inverted latent.
- `gen_rec`: generated latent/image vs reconstructed latent/image.
- `input_vae_psnr`: real input image vs VAE decode.
- `input_rec_psnr`: real input image vs inversion reconstruction.
- `prompt_pressure_P_t`, `relative_pressure_R_t`, `bending_B_t`: prompt pressure trace metrics.

## What Not To Assume

- Do not assume `results/` is clean; it is intentionally a mixed analysis workspace.
- Do not assume shell launchers are portable; many contain cluster-specific `sbatch` and `conda activate afpi`.
- Current script convention is `PROJECT_DIR=${PROJECT_DIR:-$PWD}` plus Slurm `--chdir="$PROJECT_DIR"`; see `docs/SCRIPT_STYLE_GUIDE.md`.
- Root Python files are compatibility wrappers after the reorg. Edit implementations under `experiments/`, and keep wrappers unless all shell/Slurm callers are updated.
- Before deleting or archiving anything in `outputs/` or `results/`, ask the user.
