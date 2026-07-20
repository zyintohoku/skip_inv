# Node Information Collection Scripts

这些脚本用于收集SLURM集群中yagi节点的详细信息。

## 📋 可用脚本

### 1. 标准版本（推荐）
```bash
./scripts/collect_node_info.sh
```
- ✅ 详细输出，带时间戳
- ✅ 错误处理
- ✅ 使用建议

**输出**: `node_info_YYYYMMDD_HHMMSS.txt`

### 2. 简洁版本
```bash
./scripts/collect_node_info_simple.sh [output_file]
```
- ✅ 快速执行
- ✅ 实时显示进度
- ✅ 可自定义输出文件名

**示例**:
```bash
./scripts/collect_node_info_simple.sh my_nodes.txt
```

### 3. 高级版本（推荐用于对比）
```bash
./scripts/collect_node_info_advanced.sh [output_file]
```
- ✅ 详细输出 + CSV表格
- ✅ 自动提取关键信息
- ✅ 对比表格展示

**输出**: 
- `node_info_YYYYMMDD_HHMMSS.txt` - 详细信息
- `node_info_YYYYMMDD_HHMMSS.csv` - CSV表格

---

## 🚀 快速开始

### 最简单的方式
```bash
cd /home/yzeng/remote/skip_inv
./scripts/collect_node_info.sh
```

### 查看结果
```bash
# 查看生成的文件
ls -lh node_info_*.txt

# 查看最新的文件
cat node_info_*.txt | less

# 或者
tail -f node_info_*.txt
```

---

## 📊 目标节点

脚本会收集以下节点的信息：
- yagi29, yagi33, yagi34, yagi35, yagi36
- yagi37, yagi38, yagi39, yagi40, yagi41
- yagi43, yagi45

总共：**12个节点**

---

## 📄 输出示例

### 标准输出格式
```
################################################################################
# Node: yagi36
################################################################################
NodeName=yagi36 Arch=x86_64 CoresPerSocket=4 
   CPUAlloc=4 CPUEfctv=16 CPUTot=16 CPULoad=5.03
   Gres=gpu:rtxA6000:4
   RealMemory=257500 AllocMem=122880 
   State=MIXED Sockets=2 Boards=1
   ...
```

### CSV表格格式（高级版）
```
Node     State  CPUs  Memory(MB)  GPUs  GPU_Type   Sockets  CoresPerSocket
yagi29   IDLE   128   257561      4     rtxA6000   1        64
yagi33   MIXED  64    515123      4     rtxA6000   2        32
yagi36   MIXED  16    257500      4     rtxA6000   2        4
yagi37   MIXED  128   257561      4     rtxA6000   1        64
...
```

---

## 🔍 常用查询

收集完信息后，可以使用以下命令进行分析：

### 查找特定信息
```bash
# 查看所有GPU信息
grep 'Gres' node_info_*.txt

# 查看所有CPU配置
grep 'CPUTot' node_info_*.txt

# 查看节点状态
grep 'State=' node_info_*.txt

# 查看内存配置
grep 'RealMemory' node_info_*.txt
```

### 统计分析
```bash
# 统计各状态的节点数量
grep -oP 'State=\K[^ ]+' node_info_*.txt | sort | uniq -c

# 统计GPU类型
grep -oP 'gpu:\K[^:]+' node_info_*.txt | sort | uniq -c

# 统计CPU总数
grep -oP 'CPUTot=\K[0-9]+' node_info_*.txt | awk '{s+=$1} END {print "Total CPUs:", s}'

# 统计总内存
grep -oP 'RealMemory=\K[0-9]+' node_info_*.txt | awk '{s+=$1} END {printf "Total Memory: %.2f GB\n", s/1024}'
```

### 对比分析
```bash
# 对比yagi36和yagi37
diff <(grep -A 20 "Node: yagi36" node_info_*.txt) \
     <(grep -A 20 "Node: yagi37" node_info_*.txt)

# 查看所有单socket节点
grep -B 1 'Sockets=1' node_info_*.txt

# 查看所有双socket节点
grep -B 1 'Sockets=2' node_info_*.txt
```

---

## 🛠️ 自定义节点列表

如果需要查询不同的节点，编辑脚本中的 `NODES` 数组：

```bash
# 编辑任一脚本
vim scripts/collect_node_info.sh

# 修改这一行：
NODES=(29 33 34 35 36 37 38 39 40 41 43 45)

# 改为你需要的节点：
NODES=(36 37)  # 只查询yagi36和yagi37
```

---

## 📋 输出字段说明

| 字段 | 说明 | 示例 |
|------|------|------|
| NodeName | 节点名称 | yagi36 |
| State | 节点状态 | IDLE/MIXED/ALLOCATED/DOWN |
| CPUTot | 总CPU核心数 | 16 |
| CPUAlloc | 已分配CPU数 | 4 |
| CoresPerSocket | 每个socket的核心数 | 4 |
| Sockets | Socket数量 | 2 |
| RealMemory | 总内存(MB) | 257500 |
| Gres | 通用资源(GPU等) | gpu:rtxA6000:4 |
| OS | 操作系统 | Linux 6.8.0-90-generic |

---

## 💡 使用场景

### 场景1: 选择合适的节点运行实验
```bash
# 收集信息
./scripts/collect_node_info_advanced.sh

# 查看空闲节点
grep 'State=IDLE' node_info_*.txt | grep -oP 'Node: \K[^ ]+'

# 查看带RTX A6000的节点
grep 'rtxA6000' node_info_*.txt | grep -oP 'Node: \K[^ ]+'
```

### 场景2: 调试不同服务器上的结果差异
```bash
# 收集yagi36和yagi37的详细信息
./scripts/collect_node_info.sh

# 对比两个节点的配置
diff <(grep -A 30 "Node: yagi36" node_info_*.txt) \
     <(grep -A 30 "Node: yagi37" node_info_*.txt) | less
```

### 场景3: 为论文收集硬件信息
```bash
# 使用高级版本生成表格
./scripts/collect_node_info_advanced.sh hardware_specs.txt

# CSV可以直接导入到Excel或LaTeX表格
cat hardware_specs.csv
```

---

## ⚠️ 注意事项

1. **权限要求**: 需要能够执行 `scontrol` 命令
2. **节点可用性**: 某些节点可能不存在或不可访问
3. **输出大小**: 每个节点约1-2KB，12个节点总共约15-25KB

---

## 🔧 故障排除

### 问题1: Permission denied
```bash
# 添加执行权限
chmod +x scripts/collect_node_info*.sh
```

### 问题2: scontrol: command not found
```bash
# 确保在SLURM环境中运行
# 或者加载SLURM模块
module load slurm
```

### 问题3: 某些节点信息无法获取
```bash
# 检查节点是否存在
sinfo -N | grep yagi

# 手动测试单个节点
scontrol show node yagi36
```

---

## 📚 相关文档

- [SLURM scontrol文档](https://slurm.schedmd.com/scontrol.html)
- [服务器差异分析](docs/SERVER_DIFFERENCE_ANALYSIS.md)
- [节点选择最佳实践](docs/SERVER_DIFFERENCE_README.md)

---

## 📝 更新日志

- **2026-04-05**: 创建脚本
  - 添加标准版、简洁版、高级版三个版本
  - 支持12个yagi节点
  - 自动生成CSV对比表格

---

**维护者**: AI Assistant  
**最后更新**: 2026-04-05
