#!/usr/bin/env bash

# Shared conventions for repository shell launchers.
#
# Path rule:
#   PROJECT_DIR defaults to the directory where the user invoked the script.
#   Slurm jobs should use --chdir="$PROJECT_DIR"; avoid cd in --wrap.
#
# Node rule:
#   1. Positional nodes win: bash script.sh yagi35 yagi38:2
#   2. NODE_TASKS supports per-node slots: NODE_TASKS=yagi35:2,yagi38:1
#   3. NODE_SLOTS supports explicit repeated slots: NODE_SLOTS=yagi35,yagi35,yagi38
#   4. Otherwise use available nodes from TARGET_NODES, repeated TASKS_PER_NODE times.

PROJECT_DIR=${PROJECT_DIR:-$PWD}
TARGET_NODES=${TARGET_NODES:-yagi29,yagi33,yagi34,yagi35,yagi36,yagi37,yagi38,yagi39,yagi40,yagi41}
TASKS_PER_NODE=${TASKS_PER_NODE:-1}
DEFAULT_PARTITION=${DEFAULT_PARTITION:-48-4}
LOG_DIR=${LOG_DIR:-log}

slurm_require_command() {
    local cmd=$1
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: $cmd not found. Run this on a Slurm login/server node." >&2
        return 1
    fi
}

slurm_prepare_log_dir() {
    mkdir -p "$PROJECT_DIR/$LOG_DIR"
}

slurm_partition() {
    local node=$1
    local partition
    partition=$(sinfo -N -h -o "%P" -n "$node" 2>/dev/null | head -1 | sed 's/*$//')
    if [ -z "$partition" ]; then
        partition=$DEFAULT_PARTITION
    fi
    printf '%s\n' "$partition"
}

slurm_available_nodes() {
    sinfo -N -h -o "%N %t" | awk -v nodes="$TARGET_NODES" '
        BEGIN {
            split(nodes, arr, ",");
            for (i in arr) allow[arr[i]] = 1;
        }
        $2 ~ /idle|mix/ && ($1 in allow) { print $1 }
    ' | sort -V
}

slurm_show_target_status() {
    sinfo -N -h -o "%N %P %t" | awk -v nodes="$TARGET_NODES" '
        BEGIN {
            split(nodes, arr, ",");
            for (i in arr) allow[arr[i]] = 1;
        }
        ($1 in allow) { print }
    ' | sort -V
}

slurm_emit_node_slots_from_specs() {
    local specs=("$@")
    local spec node count i
    for spec in "${specs[@]}"; do
        spec=${spec//,/ }
        for token in $spec; do
            node=${token%%:*}
            count=${token#*:}
            if [ "$node" = "$count" ]; then
                count=1
            fi
            if ! [[ "$count" =~ ^[0-9]+$ ]] || [ "$count" -lt 1 ]; then
                echo "Error: invalid node task spec '$token'. Use yagi35 or yagi35:2." >&2
                return 1
            fi
            for ((i = 0; i < count; i++)); do
                printf '%s\n' "$node"
            done
        done
    done
}

slurm_resolve_node_slots() {
    local arg_count=$#
    local available_nodes node i

    if [ "$arg_count" -gt 0 ]; then
        slurm_emit_node_slots_from_specs "$@"
        return
    fi

    if [ -n "${NODE_TASKS:-}" ]; then
        slurm_emit_node_slots_from_specs "$NODE_TASKS"
        return
    fi

    if [ -n "${NODE_SLOTS:-}" ]; then
        slurm_emit_node_slots_from_specs "$NODE_SLOTS"
        return
    fi

    available_nodes=$(slurm_available_nodes)
    if [ -z "$available_nodes" ]; then
        echo "Error: no available nodes found in TARGET_NODES=$TARGET_NODES" >&2
        slurm_show_target_status >&2 || true
        return 1
    fi

    while IFS= read -r node; do
        [ -z "$node" ] && continue
        for ((i = 0; i < TASKS_PER_NODE; i++)); do
            printf '%s\n' "$node"
        done
    done <<< "$available_nodes"
}

slurm_load_node_slots() {
    local -n _out_array=$1
    shift
    mapfile -t _out_array < <(slurm_resolve_node_slots "$@")
    if [ "${#_out_array[@]}" -eq 0 ]; then
        echo "Error: resolved zero Slurm node slots." >&2
        return 1
    fi
}

slurm_node_for_index() {
    local idx=$1
    shift
    local slots=("$@")
    printf '%s\n' "${slots[$((idx % ${#slots[@]}))]}"
}

slurm_conda_prefix() {
    local conda_env=${1:-${CONDA_ENV:-afpi}}
    printf 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate %q' "$conda_env"
}
