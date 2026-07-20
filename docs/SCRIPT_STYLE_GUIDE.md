# Shell/Slurm Script Style Guide

本仓库的 shell 脚本分两类：提交到 Slurm 服务器的脚本占多数，本地直接执行的后处理脚本占少数。以后新增或改脚本时按这里的约定写，方便本地挂载路径和服务器真实路径共存。

## 路径约定

服务器路径通常是：

```bash
~/po1/yan/skip_inv
```

本地挂载路径通常是：

```bash
~/remote/skip_inv
```

脚本不要硬编码这两个路径。默认使用执行脚本时的当前目录：

```bash
PROJECT_DIR=${PROJECT_DIR:-$PWD}
```

提交 Slurm 时，使用 `--chdir="$PROJECT_DIR"` 设置作业工作目录，不要在 `--wrap` 里再 `cd "$PROJECT_DIR"`：

```bash
sbatch --chdir="$PROJECT_DIR" \
       --wrap="bash -lc 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && python run.py'"
```

本地执行脚本也不要自动 `cd`。如果用户想指定别的目录，可以显式传：

```bash
PROJECT_DIR=~/po1/yan/skip_inv bash scripts/some_script.sh
```

## 共享 helper

提交类脚本优先 source：

```bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/slurm_common.sh"
```

它会提供：

- `PROJECT_DIR=${PROJECT_DIR:-$PWD}`
- `TARGET_NODES` 默认 yagi29,yagi33-yagi41
- `slurm_load_node_slots NODE_SLOTS_ARRAY "$@"`
- `slurm_node_for_index "$idx" "${NODE_SLOTS_ARRAY[@]}"`
- `slurm_partition "$node"`
- `slurm_prepare_log_dir`

目前已接入这套 helper 的主要脚本包括：

- `scripts/run_pie_image_inv_rec.sh`
- `scripts/run_all_reconstruction_clip_scores.sh`
- `scripts/run_cfg7_rec_latent_refpi.sh`
- `scripts/run_worst_prompt_cfg_mismatch_fpi.sh`
- `scripts/run_prompt_pressure_top10.sh`
- `scripts/run_prompt_pressure_seed_sensitive_top10.sh`
- `scripts/run_prompt_pressure_best_worst_top10_saved_latents.sh`
- `scripts/run_slerp_seed_pair_fpi.sh`
- `scripts/run_uncond_slerp_initial_latents.sh`
- `scripts/run_sdxl_euler_prompt_grid_gen_inv_rec.sh`
- `scripts/run_selected_inv_latent_uncond_generation.sh`
- `scripts/run_selected_inv_latent_best_prompt_generation.sh`

## Node 指定规范

统一支持三种方式。

### 1. 命令行直接指定 node

```bash
bash scripts/run_xxx.sh yagi35 yagi38
```

任务按 node 轮询提交。

### 2. 命令行指定每个 node 的任务槽数

```bash
bash scripts/run_xxx.sh yagi35:2 yagi38:1
```

等价于 node slot 列表：

```text
yagi35, yagi35, yagi38
```

多数脚本会按这个 slot 列表轮询提交任务。若任务数等于 slot 数，这就是精确的“每个 node 提交几个任务”；若任务数更多，它表示分配权重/轮询槽位。

### 3. 环境变量指定

```bash
NODE_TASKS=yagi35:2,yagi38:1 bash scripts/run_xxx.sh
NODE_SLOTS=yagi35,yagi35,yagi38 bash scripts/run_xxx.sh
TASKS_PER_NODE=2 bash scripts/run_xxx.sh
TARGET_NODES=yagi35,yagi38 bash scripts/run_xxx.sh
```

优先级：

1. 命令行 node 参数
2. `NODE_TASKS`
3. `NODE_SLOTS`
4. 自动从 `TARGET_NODES` 里找 idle/mix node，并按 `TASKS_PER_NODE` 展开

## 推荐提交模板

```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/slurm_common.sh"

CONDA_ENV=${CONDA_ENV:-afpi}
MEM=${MEM:-48G}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
slurm_prepare_log_dir
slurm_load_node_slots NODE_SLOTS_ARRAY "$@"

TASKS=(1 2 3)
for idx in "${!TASKS[@]}"; do
    task=${TASKS[$idx]}
    node=$(slurm_node_for_index "$idx" "${NODE_SLOTS_ARRAY[@]}")
    partition=$(slurm_partition "$node")
    job_name="example_${task}"

    cmd=(python run.py --seed "$task" --output "outputs/example_${task}")
    printf -v quoted_cmd '%q ' "${cmd[@]}"

    sbatch --job-name="$job_name" \
           --chdir="$PROJECT_DIR" \
           --nodelist="$node" \
           --partition="$partition" \
           --gres=gpu:1 \
           --mem="$MEM" \
           --cpus-per-task="$CPUS_PER_TASK" \
           --output="$LOG_DIR/${job_name}.out" \
           --error="$LOG_DIR/${job_name}.err" \
           --wrap="bash -lc 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && $quoted_cmd'"
done
```

## 本地执行模板

```bash
#!/bin/bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$PWD}
INPUT_ROOT=${INPUT_ROOT:-outputs/some_run}
OUTPUT_ROOT=${OUTPUT_ROOT:-results/some_analysis}

python analysis/some_analysis.py \
    --input_root "$INPUT_ROOT" \
    --output_dir "$OUTPUT_ROOT"
```

## 避免事项

- 不要默认 `PROJECT_DIR=/po1/yan/skip_inv`。
- 不要在脚本开头 `cd "$PROJECT_DIR"`。
- 不要在 `--wrap` 中用 `cd $PROJECT_DIR && ...`；使用 `--chdir="$PROJECT_DIR"`。
- 不要在不同脚本里各写一套 node 正则。使用 `TARGET_NODES`、`NODE_TASKS` 和 helper。
- 不要把本地挂载路径写进提交脚本。
