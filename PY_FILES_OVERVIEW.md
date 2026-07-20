# Python 文件作用概览

本文件用于整理当前仓库内 `.py` 文件的用途，便于后续决定哪些保留在根目录、哪些移动到 `analysis/`、`utils/`、`scripts/` 或归档目录。

新 session 或 agent 建议先读 `AGENTS.md` 和 `docs/CODE_RETRIEVAL_INDEX.md`；本文件作为完整 Python 文件清单使用。

说明基于文件名、文件头注释、参数入口和主要导入关系的快速梳理；个别实验脚本可能带有历史上下文，后续整理时可再细化。

## 根目录兼容入口

根目录下的 `run*.py`、`generate*.py`、`prompt_pressure*.py`、`skip_inv.py`、`test_*.py` 等现在是兼容 wrapper。它们会转发到 `experiments/` 中的真实实现，目的是让已有 shell/Slurm 脚本和旧命令继续可用。修改实验逻辑时应编辑 `experiments/` 里的文件。

## `experiments/` 实验与核心入口

| 文件 | 简要作用 |
| --- | --- |
| `experiments/__init__.py` | 将 `experiments/` 标记为实验入口包。 |
| `experiments/sitecustomize.py` | 让直接执行 `python experiments/<script>.py` 时自动加入 repo root 到 import path。 |
| `experiments/cfg_directional_features_saved_latents.py` | 基于已保存 latent 计算 CFG 方向相关特征，并输出与 PSNR 等指标联动的汇总数据。 |
| `experiments/gen.py` | 从已有 `inv_latents.pt` 重新按不同 guidance scale 生成重建图像和 `rec_latents.pt`。 |
| `experiments/generate_aidi_gs7_seed_pressure.py` | 为 AIDI GS7 多 seed 生成 prompt pressure 诊断数据。 |
| `experiments/generate_selected_inv_latent_uncond.py` | 对选定 inversion latent 运行 unconditional 生成，并制作相关图像/记录。 |
| `experiments/generate_uncond_slerp_initial_latents.py` | 在 seed pair 的初始 latent 间做 SLERP，并生成 unconditional 轨迹/图像。 |
| `experiments/latent_perturbation_saved_latents.py` | 对已保存 latent 做扰动敏感性实验，评估扰动对生成/重建的影响。 |
| `experiments/prompt_pressure_pipeline.py` | Prompt pressure 诊断的公共流水线：seed latent、DDIM 系数、trace 采集、保存单样本诊断。 |
| `experiments/prompt_pressure_saved_latents.py` | 基于已保存 latent 批量计算 prompt pressure trace 和汇总。 |
| `experiments/run.py` | 早期主实验入口：遍历 PIE 样本，生成、反演、重建并保存 init/inv/gen/rec latents。 |
| `experiments/run_batch.py` | `run.py` 的分批版本，支持 sample range 和 batch 文件后缀，便于多任务并行。 |
| `experiments/run_cfg7_rec_latent_refpi.py` | 使用 CFG=7 重建 latent，并结合 RefPI/FPI 相关配置输出对比记录。 |
| `experiments/run_fpi_from_saved_latents.py` | 从已有生成 latent 或保存结果继续运行 FPI inversion 与重建。 |
| `experiments/run_pie_image_inv_rec.py` | 对 PIE 原始图像做 VAE 编码、反演、重建，并计算图像/latent 指标。 |
| `experiments/run_prompt_ablation_fpi_gen_inv_rec.py` | 对一组 prompt 做生成、FPI 反演、重建和 PSNR 统计，服务 prompt ablation。 |
| `experiments/run_sample692_seed1_seed4_inv_slerp.py` | 针对 sample 692、seed 1/4 的 inversion latent SLERP 专项实验。 |
| `experiments/run_sdxl_euler_prompt_grid_gen_inv_rec.py` | 用 SDXL base + Euler 对 prompt grid 运行生成、反演、重建。 |
| `experiments/run_single_sample.py` | 单样本调试入口，生成/反演/重建并保存单个样本结果。 |
| `experiments/run_slerp_seed_pair_fpi.py` | 对 seed pair latent 做 SLERP，并对各 alpha 运行 FPI 与指标记录。 |
| `experiments/run_with_generation_trace.py` | 运行生成/反演/重建，同时保存 generation trace 便于分析中间过程。 |
| `experiments/run_worst_prompt_cfg_mismatch_fpi.py` | 针对 worst prompt 或 CFG mismatch 场景运行 FPI 对比实验。 |
| `experiments/skip_inv.py` | Skip inversion 实验入口，支持 adaptive CFG schedule、checkpoint/resume 和 fixed CFG 对照。 |
| `experiments/test_aidi.py` | AIDI 早期测试脚本，结构接近 `run.py`，主要用于快速验证 AIDI 生成/反演逻辑。 |
| `experiments/test_api.py` | OpenAI API 图像调用的小型测试脚本。 |
| `experiments/test_fpi.py` | FPI/AFPI 细节测试脚本，包含收敛、latent 轨迹、重建检查和可视化辅助。 |
| `experiments/test_skip_inv.py` | 单样本测试 `skip_inv` 的 CFG schedule 行为。 |

## `analysis/` 分析、统计与绘图

| 文件 | 简要作用 |
| --- | --- |
| `analysis/analyze_aidi.py` | 汇总不同 AIDI guidance scale 的 latent 指标并打印表格。 |
| `analysis/analyze_aidi_gs7_new_results.py` | 分析新一批 AIDI-GS7 结果，计算每样本 latent 指标。 |
| `analysis/analyze_aidi_gs7_prompt_psnr.py` | 分析 prompt/sample id 对 AIDI-GS7 多 seed PSNR 的影响。 |
| `analysis/analyze_aidi_gs7_results.py` | 分析 AIDI-GS7 旧结果，计算 init/inv、gen/rec 等指标。 |
| `analysis/analyze_epsilon_guidance_cos_sensitive.py` | 对 seed-sensitive 样本分析 epsilon/guidance 方向余弦与 PSNR 的关系。 |
| `analysis/analyze_fpi_sensitive_top10_seed_effect.py` | 对 FPI 敏感 top10 样本分析 seed 影响，并输出矩阵/图。 |
| `analysis/analyze_latent_perturbation_sensitivity.py` | 汇总 latent perturbation 实验，比较不同 seed 下扰动敏感度。 |
| `analysis/analyze_paraphrase_prompt_grid.py` | 分析 paraphrase prompt grid 的 FPI PSNR 行为，兼容未完成输出。 |
| `analysis/analyze_prompt_structure_psnr.py` | 用启发式 prompt 结构特征分析其与 PSNR 的关系。 |
| `analysis/analyze_skip_inv_results.py` | 分析 skip_inv 各配置输出，找出 worst samples 和指标分布。 |
| `analysis/compute_aidi_gs7_image_metrics.py` | 对 AIDI-GS7 图像结果计算 PSNR、SSIM、LPIPS 等图像指标。 |
| `analysis/compute_aidi_gs7_seed_psnr.py` | 计算 `outputs/aidi_gs7_seed*` 中 gen/rec 图像的 PSNR，并按 seed/prompt 汇总。 |
| `analysis/compute_all_prompt_inv_latent_gaussian_and_plots.py` | 计算 all-prompt inversion latent 的高斯统计，并绘制其与 CLIP/PSNR 的关系。 |
| `analysis/compute_all_reconstruction_clip_scores.py` | 为所有 FPI-GS7 seed 的 gen/rec 图像对计算 CLIP image score。 |
| `analysis/compute_fpi_gs7_seed_psnr.py` | 计算 FPI-GS7 多 seed 的重建 PSNR，包含 prompt 特征和均值/方差图。 |
| `analysis/compute_id185_308_ablation_fpi_psnr.py` | 聚合 id185/id308 prompt ablation FPI 结果的 PSNR 指标。 |
| `analysis/compute_prompt_grid_fpi_psnr.py` | 聚合任意 prompt grid 的 per-seed FPI gen/inv/rec 指标。 |
| `analysis/compute_prompt_pressure_distribution_metrics.py` | 从 prompt pressure traces 计算曲线分布指标，如 entropy、gini、stage sums。 |
| `analysis/compute_sdxl_euler_prompt_grid_psnr.py` | 聚合 SDXL Euler prompt grid 的 per-seed gen/inv/rec 指标。 |
| `analysis/compute_top10_reconstruction_clip_pressure.py` | 为 best/worst/sensitive top10 prompt 计算重建 CLIP 分数并关联 pressure。 |
| `analysis/copy_worst_sample_images.py` | 复制指定 worst sample 的 gen/rec 图像到集中目录。 |
| `analysis/drilldown_seed_sensitive_sample.py` | 展开单个 seed-sensitive 样本在多 seed 下的 pressure、PSNR 和图像路径。 |
| `analysis/evaluate_p2p.py` | 评估 P2P 编辑结果，计算 CLIP、LPIPS、PSNR、SSIM 等指标。 |
| `analysis/export_aidi_gs7_all_merge_with_images.py` | 导出/合并 AIDI-GS7 样本表并附带图像路径。 |
| `analysis/export_aidi_gs7_best_samples.py` | 复制 AIDI-GS7 最佳样本图像并生成 HTML 预览。 |
| `analysis/export_aidi_gs7_worst_samples.py` | 复制 AIDI-GS7 最差样本图像并生成 HTML 预览。 |
| `analysis/generate_report.py` | 为 ablation 结果生成 Markdown/JSON 报告。 |
| `analysis/make_gen_rec_seed_pair_montage.py` | 为指定 sample 的多 seed gen/rec 图像制作对比 montage。 |
| `analysis/make_slerp_divergence_tables.py` | 从 SLERP FPI traces 生成 divergence timestep 对比表。 |
| `analysis/make_slerp_gen_rec_error_montage.py` | 为 SLERP FPI group 制作 gen/rec/error/loss montage。 |
| `analysis/make_slerp_metric_tables.py` | 从 SLERP per-alpha CSV 生成紧凑指标表和 LaTeX/Markdown 表。 |
| `analysis/merge_cfg_directional_feature_summaries.py` | 合并 CFG directional feature 的 summary CSV。 |
| `analysis/merge_modifier_context_fpi_results.py` | 合并 original/extra modifier-context FPI 结果，并绘制 context heatmap。 |
| `analysis/plot_ablation.py` | 绘制 ablation 结果散点图，并输出方法对比表。 |
| `analysis/plot_aidi.py` | 汇总并绘制 AIDI 多 guidance scale 的指标表现。 |
| `analysis/plot_aidi_diagonal_distribution.py` | 绘制 AIDI 结果的对角线/分布相关图。 |
| `analysis/plot_aidi_distribution.py` | 绘制 AIDI 每样本 init-inv 与 gen-rec 分布散点。 |
| `analysis/plot_aidi_gen_rec_distribution.py` | 专门绘制 AIDI gen-rec 指标分布。 |
| `analysis/plot_aidi_gs7_generation_pressure_vs_psnr.py` | 绘制 AIDI-GS7 generation pressure 与 PSNR 的关系。 |
| `analysis/plot_aidi_gs7_prompt_mean_std.py` | 绘制 AIDI-GS7 prompt 平均 PSNR 与 seed-sensitive 标准差。 |
| `analysis/plot_alignment_cos_time_examples.py` | 挑选样本绘制 alignment cosine 随 timestep 变化的示例图。 |
| `analysis/plot_all_prompt_clip_psnr_summary.py` | 绘制 all-prompt CLIP 与 PSNR 的总体关系、直方图和选点标注。 |
| `analysis/plot_cfg_directional_features_vs_psnr.py` | 绘制 CFG directional features 与 PSNR 的关系图。 |
| `analysis/plot_convergence.py` | 可视化 FPI/AFPI/AIDI 的 convergence loss 曲线，可附带 gen/rec 图像。 |
| `analysis/plot_fpi_top10_pressure_vs_psnr.py` | 绘制 FPI top10 prompt 的 pressure 与 PSNR 关系。 |
| `analysis/plot_guidance_delta_l2.py` | 绘制 guidance delta L2 曲线及其分组均值。 |
| `analysis/plot_latent_step_length_time_examples.py` | 绘制 latent step length 随 timestep 的示例图。 |
| `analysis/plot_latent_step_lengths.py` | 从 trace tensor 计算并绘制 latent step length 曲线和汇总。 |
| `analysis/plot_modifier_prompt_grid_heatmap.py` | 绘制 modifier prompt-grid 的 class-by-context PSNR heatmap。 |
| `analysis/plot_p2p_clip_psnr.py` | 绘制 P2P 编辑结果的 CLIP/PSNR 综合对比图。 |
| `analysis/plot_p2p_results.py` | 绘制 P2P 结果的 CLIP threshold 曲线和汇总图。 |
| `analysis/plot_pressure_stage_sums_by_label.py` | 按 label 汇总 prompt pressure 不同阶段的和并绘图。 |
| `analysis/plot_prompt_pressure_by_psnr.py` | 按 PSNR 给 prompt pressure 曲线着色，比较高低重建质量。 |
| `analysis/plot_prompt_pressure_by_seed.py` | 按 seed 绘制 prompt pressure 曲线和样本/分组均值。 |
| `analysis/plot_prompt_pressure_normalized.py` | 绘制归一化后的 prompt pressure 曲线。 |
| `analysis/plot_psnr_threshold_percent_heatmaps.py` | 绘制 class x context 在不同 PSNR 阈值上的通过率 heatmap。 |
| `analysis/plot_puppy_dog_field_heatmap.py` | 绘制 puppy/dog/field prompt-grid 的 PSNR heatmap。 |
| `analysis/plot_seed_sensitive_pressure_vs_psnr.py` | 绘制 seed-sensitive 样本的 pressure 指标与 PSNR 关系。 |
| `analysis/plot_sensitive_cumulative_pressure_by_psnr.py` | 绘制 sensitive 样本累计 pressure 曲线，并按 PSNR 着色。 |
| `analysis/plot_sensitive_final_step_pressure_vs_psnr.py` | 分析 final-step pressure 与 PSNR 的散点关系。 |
| `analysis/plot_skip_inv_distribution.py` | 绘制 skip_inv 各配置的每样本指标分布。 |
| `analysis/plot_slerp_inversion_loss_curves.py` | 绘制 SLERP FPI inversion loss 曲线，并按重建 PSNR 着色。 |
| `analysis/plot_threshold_comparison.py` | 比较不同阈值设置下 latent 和图像指标的分布/通过率。 |
| `analysis/sample_prompt_seed_variance.py` | 随机采样 prompts，比较 prompt-wise seed variance 与 seed-wise prompt variance。 |
| `analysis/summarize_saved_latent_prompt_pressure_table.py` | 把 saved-latent prompt pressure 指标整理成 prompt/label 汇总表。 |
| `analysis/two_way_anova_prompt_seed.py` | 对 prompt 和 seed 效应做 two-way ANOVA。 |

## `utils/` 公共实现

| 文件 | 简要作用 |
| --- | --- |
| `utils/inv_methods.py` | Stable Diffusion v1.4 的自定义生成 pipeline、VAE image-to-latent 编码，以及 FPI/AFPI/AIDI inversion 实现。 |
| `utils/reproducibility.py` | 设置随机种子、PyTorch deterministic/benchmark 等可重复性选项。 |
| `utils/sdxl_euler_inv_methods.py` | SDXL base + Euler 反演/生成的公共类和工具函数。 |
| `utils/skip_pipe.py` | 支持 skip/adaptive CFG schedule 的自定义 Stable Diffusion pipeline 与 inversion 实现。 |

## `p2p/` Prompt-to-Prompt 编辑相关

| 文件 | 简要作用 |
| --- | --- |
| `p2p/p2p.py` | Prompt-to-Prompt 注意力控制、替换/细化/reweight 等编辑控制逻辑。 |
| `p2p/ptp_utils.py` | P2P 辅助工具：词索引、attention alpha、控制器注册、图像生成等。 |
| `p2p/seq_aligner.py` | P2P prompt token 对齐工具，用于 replacement/refinement mapper。 |

## `scripts/` 中的 Python 辅助脚本

| 文件 | 简要作用 |
| --- | --- |
| `scripts/compare_methods.py` | 比较 AFPI/AIDI 等方法的 init-inv、gen-rec 与运行时间，并生成可视化。 |
| `scripts/compare_servers.py` | 比较两个服务器/节点输出 latent 的差异，输出统计检验和 JSON 报告。 |
| `scripts/merge_batch_results.py` | 合并 `run_batch.py` 产生的多 batch latent 和 convergence loss。 |
| `scripts/quick_compare_yagi.py` | 针对 yagi36/yagi37 的一次硬编码快速结果对比说明。 |

## `docs/` 示例代码

| 文件 | 简要作用 |
| --- | --- |
| `docs/REPRODUCIBILITY_EXAMPLES.py` | 展示如何在实验脚本中使用 `utils/reproducibility.py` 设置可重复性。 |

## `tmp_sdxl_base_test/` 临时 SDXL 测试

| 文件 | 简要作用 |
| --- | --- |
| `tmp_sdxl_base_test/sdxl_euler_inversion_reconstruct.py` | 临时测试 SDXL Euler inversion 与 reconstruction 流程。 |
| `tmp_sdxl_base_test/test_sdxl_base_only.py` | 只加载/运行 SDXL base 的最小测试脚本。 |

## 后续整理建议

- 根目录建议只保留当前最常用的实验入口和核心 pipeline；历史专项实验可移动到 `experiments/` 或 `scripts/experiments/`。
- `analysis/` 可以进一步按 `compute/`、`plot/`、`export/`、`archive/` 分层，尤其是 prompt pressure、SLERP、P2P、AIDI-GS7 这几类脚本。
- `scripts/` 中 `.sh` 可单独建立索引，按“提交实验”“分析后处理”“节点/环境工具”分类。
- `tmp_sdxl_base_test/` 若仍有价值，可改名为 `experiments/sdxl_euler/`；若只是临时验证，可归档或删除。
