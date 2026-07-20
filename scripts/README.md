# Scripts 目录导航

本目录主要放 shell/Slurm 启动脚本。大多数脚本只是选择节点、设置环境变量、提交 `sbatch`，真正的实验逻辑通常在 `experiments/`、`utils/` 或 `analysis/` 中。仓库根目录同名 Python 文件多为兼容 wrapper，用来保证旧 Slurm 命令仍可运行。

新 session 快速检索代码时，优先看：

- `../AGENTS.md`
- `../docs/CODE_RETRIEVAL_INDEX.md`
- `../docs/SCRIPT_STYLE_GUIDE.md`
- `../PY_FILES_OVERVIEW.md`

## 常用入口

| 任务 | 脚本 | 调用的主要 Python |
| --- | --- | --- |
| AIDI guidance scale 实验 | `run.sh` | `run.py` |
| AFPI 不同 LDT 实验 | `run_afpi_ldt.sh` | `run.py` |
| 分批运行并合并 | `run_batch.sh`, `run_aidi_135.sh` | `run_batch.py`, `scripts/merge_batch_results.py` |
| GS7 多 seed | `run_gs7_seed_1_10.sh` | `run.py` |
| PIE 原图 VAE encode/invert/reconstruct | `run_pie_image_inv_rec.sh` | `run_pie_image_inv_rec.py` |
| 从保存 latent 运行 FPI | `run_fpi_gs7_seed_from_saved_latents.sh` | `run_fpi_from_saved_latents.py` |
| 生成 trace | `run_fpi_gs7_seed_generation_trace.sh` | `run_with_generation_trace.py` |
| Prompt pressure | `run_prompt_pressure_top10.sh`, `run_prompt_pressure_seed_sensitive_top10.sh`, `run_prompt_pressure_best_worst_top10_saved_latents.sh` | `prompt_pressure_pipeline.py`, `prompt_pressure_saved_latents.py` |
| Prompt pressure 后处理 | `analyze_prompt_pressure_seed_sensitive_top10.sh`, `analyze_prompt_pressure_best_worst_top10_saved_latents.sh`, `plot_pressure_vs_psnr_saved_latent_top10_all.sh` | `analysis/compute_prompt_pressure_distribution_metrics.py`, `analysis/plot_prompt_pressure_*.py` |
| Prompt grid / ablation | `run_id185_308_ablation_fpi_gen_inv_rec.sh`, `run_modifier_prompt_grid_fpi_gen_inv_rec.sh`, `run_modifier_extra_context_fpi_gen_inv_rec.sh`, `run_paraphrase_prompt_grid_fpi_gen_inv_rec.sh`, `run_puppy_dog_field_fpi_gen_inv_rec.sh` | `run_prompt_ablation_fpi_gen_inv_rec.py`, `analysis/compute_prompt_grid_fpi_psnr.py` |
| SLERP | `run_slerp_seed_pair_fpi.sh`, `run_slerp_seed_pair_fpi_second_sensitive.sh`, `run_uncond_slerp_initial_latents.sh` | `run_slerp_seed_pair_fpi.py`, `generate_uncond_slerp_initial_latents.py` |
| SDXL Euler prompt grid | `run_sdxl_euler_prompt_grid_gen_inv_rec.sh` | `run_sdxl_euler_prompt_grid_gen_inv_rec.py`, `analysis/compute_sdxl_euler_prompt_grid_psnr.py` |
| Skip inversion 消融 | `run_skip_inv_ablation.sh` | `skip_inv.py` |
| P2P 编辑 | `run_p2p_aidi_all.sh`, `run_p2p_afpi_ldt_all.sh`, `run_p2p_skip_inv_all.sh`, `run_p2p_upper_bound.sh` | `p2p/p2p.py` |
| 节点/环境信息 | `check_yagi_nodes.sh`, `collect_node_info*.sh`, `node_info_quick_ref.sh` | shell-only 或系统命令 |

## 检索脚本调用关系

```bash
rg -n "python |python3 |sbatch|--wrap" scripts -g "*.sh" -g "*.sbatch"
```

## Node 指定方式

新脚本统一支持以下写法：

```bash
bash scripts/run_xxx.sh yagi35 yagi38
bash scripts/run_xxx.sh yagi35:2 yagi38:1
NODE_TASKS=yagi35:2,yagi38:1 bash scripts/run_xxx.sh
NODE_SLOTS=yagi35,yagi35,yagi38 bash scripts/run_xxx.sh
```

`yagi35:2` 表示两个 node slot；多数脚本按 slot 轮询分配任务。

旧脚本如果还没有接入 `scripts/lib/slurm_common.sh`，后续改动时应按 `../docs/SCRIPT_STYLE_GUIDE.md` 迁移。

## 使用提醒

- 脚本默认 `PROJECT_DIR=$PWD`，也就是提交时所在目录；服务器上通常是 `~/po1/yan/skip_inv`，本地挂载通常是 `~/remote/skip_inv`。
- 多数脚本默认 `conda activate afpi`。
- 提交类脚本应使用 `sbatch --chdir="$PROJECT_DIR"`，避免在 `--wrap` 中 `cd "$PROJECT_DIR"`。
- `outputs/`、`results/`、`log/` 通常是生成数据，不要把它们当作源代码目录整理。
