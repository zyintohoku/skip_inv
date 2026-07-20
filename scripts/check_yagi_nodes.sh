#!/bin/bash

# Show Slurm node/GPU status for yagi37-42 and yagi45.
#
# Usage:
#   bash scripts/check_yagi_nodes.sh
#   NODES=yagi37,yagi38 bash scripts/check_yagi_nodes.sh

set -euo pipefail

NODES=${NODES:-yagi37,yagi38,yagi39,yagi40,yagi41,yagi42,yagi45}
DEBUG=${DEBUG:-0}

if ! command -v sinfo >/dev/null 2>&1; then
    echo "sinfo not found. Run this on the Slurm login/server node."
    exit 1
fi

sum_tres_gpu() {
    awk -F, '
        {
            generic_sum = 0
            typed_sum = 0
            for (i = 1; i <= NF; i++) {
                if ($i ~ /^gres\/gpu=/) {
                    split($i, parts, "=")
                    generic_sum += parts[2]
                } else if ($i ~ /^gres\/gpu:[^=]+=/) {
                    split($i, parts, "=")
                    typed_sum += parts[2]
                }
            }
            print (typed_sum > 0 ? typed_sum : generic_sum)
        }
    '
}

sum_gres_gpu() {
    awk -F, '
        {
            sum = 0
            for (i = 1; i <= NF; i++) {
                entry = $i
                sub(/\(.*/, "", entry)
                n = split(entry, parts, ":")
                if (parts[1] == "gpu") {
                    value = parts[n]
                    if (value ~ /^[0-9]+$/) {
                        sum += value
                    }
                }
            }
            print sum
        }
    '
}

gpu_counts() {
    local node=$1
    local node_info gres gres_used cfg alloc total used free
    node_info=$(scontrol show node "$node" 2>/dev/null || true)
    if [ -z "$node_info" ]; then
        echo "- - -"
        return
    fi

    gres=$(printf '%s\n' "$node_info" | tr ' ' '\n' | awk -F= '$1=="Gres"{print $2; exit}')
    gres_used=$(printf '%s\n' "$node_info" | tr ' ' '\n' | awk -F= '$1=="GresUsed"{print $2; exit}')
    cfg=$(printf '%s\n' "$node_info" | tr ' ' '\n' | awk -F= '$1=="CfgTRES"{print $2; exit}')
    alloc=$(printf '%s\n' "$node_info" | tr ' ' '\n' | awk -F= '$1=="AllocTRES"{print $2; exit}')

    total=$(printf '%s\n' "$cfg" | sum_tres_gpu)
    if [ "${total:-0}" -eq 0 ]; then
        total=$(printf '%s\n' "$gres" | sum_gres_gpu)
    fi
    used=$(printf '%s\n' "$gres_used" | sum_gres_gpu)
    if [ "${used:-0}" -eq 0 ]; then
        used=$(printf '%s\n' "$alloc" | sum_tres_gpu)
    fi

    total=${total:-0}
    used=${used:-0}
    free=$((total - used))
    if [ "$free" -lt 0 ]; then
        free=0
    fi
    echo "$total $used $free"
}

node_field() {
    local node_info=$1
    local field=$2
    printf '%s\n' "$node_info" | tr ' ' '\n' | awk -F= -v field="$field" '$1==field {print $2; exit}'
}

mem_to_gb() {
    local mb=${1:-0}
    awk -v mb="$mb" 'BEGIN { printf "%.1fG", mb / 1024 }'
}

cpu_mem_free() {
    local node=$1
    local node_info cpu_alloc cpu_tot cpu_free real_mem alloc_mem mem_free
    node_info=$(scontrol show node "$node" 2>/dev/null || true)
    if [ -z "$node_info" ]; then
        echo "- -"
        return
    fi

    cpu_alloc=$(node_field "$node_info" "CPUAlloc")
    cpu_tot=$(node_field "$node_info" "CPUTot")
    real_mem=$(node_field "$node_info" "RealMemory")
    alloc_mem=$(node_field "$node_info" "AllocMem")

    cpu_alloc=${cpu_alloc:-0}
    cpu_tot=${cpu_tot:-0}
    real_mem=${real_mem:-0}
    alloc_mem=${alloc_mem:-0}

    cpu_free=$((cpu_tot - cpu_alloc))
    if [ "$cpu_free" -lt 0 ]; then
        cpu_free=0
    fi
    mem_free=$((real_mem - alloc_mem))
    if [ "$mem_free" -lt 0 ]; then
        mem_free=0
    fi

    echo "$cpu_free $(mem_to_gb "$mem_free")"
}

echo "Node status: $NODES"
echo ""
printf '%-8s %-12s %-10s %5s %5s %5s %9s %9s\n' "NODE" "PARTITION" "STATE" "GPU" "USED" "FREE" "CPU_FREE" "MEM_FREE"
printf '%-8s %-12s %-10s %5s %5s %5s %9s %9s\n' "--------" "------------" "----------" "-----" "-----" "-----" "---------" "---------"

IFS=',' read -ra NODE_ARRAY <<< "$NODES"
for node in "${NODE_ARRAY[@]}"; do
    node=${node//[[:space:]]/}
    if [ -z "$node" ]; then
        continue
    fi

    info=$(sinfo -N -h -o "%N %P %t" -n "$node" 2>/dev/null | head -1 || true)
    if [ -z "$info" ]; then
        printf '%-8s %-12s %-10s %5s %5s %5s %9s %9s\n' "$node" "-" "unknown" "-" "-" "-" "-" "-"
        continue
    fi

    read -r _node partition state <<< "$info"
    read -r total used free <<< "$(gpu_counts "$node")"
    read -r cpu_free mem_free <<< "$(cpu_mem_free "$node")"
    printf '%-8s %-12s %-10s %5s %5s %5s %9s %9s\n' "$node" "$partition" "$state" "$total" "$used" "$free" "$cpu_free" "$mem_free"

    if [ "$DEBUG" = "1" ]; then
        node_info=$(scontrol show node "$node" 2>/dev/null || true)
        echo "  raw Gres=$(node_field "$node_info" "Gres")"
        echo "  raw GresUsed=$(node_field "$node_info" "GresUsed")"
        echo "  raw CfgTRES=$(node_field "$node_info" "CfgTRES")"
        echo "  raw AllocTRES=$(node_field "$node_info" "AllocTRES")"
        echo "  raw CPUAlloc=$(node_field "$node_info" "CPUAlloc") CPUTot=$(node_field "$node_info" "CPUTot") RealMemory=$(node_field "$node_info" "RealMemory") AllocMem=$(node_field "$node_info" "AllocMem")"
    fi
done

echo ""
echo "Jobs on these nodes:"
if command -v squeue >/dev/null 2>&1; then
    squeue -w "$NODES" -o "%.18i %.9P %.24j %.8u %.2t %.10M %.6D %R" || true
else
    echo "squeue not found."
fi
