# AIDI 测试脚本

在 yagi35, yagi38, yagi39, yagi41 上测试 AIDI 方法

## 使用方法

### 基本用法（默认测试 4 个节点）
```bash
bash scripts/run_test_aidi.sh
```

### 指定单个节点
```bash
bash scripts/run_test_aidi.sh yagi35
```

### 指定多个节点
```bash
bash scripts/run_test_aidi.sh yagi35 yagi38 yagi39 yagi41
```

## 查看结果

```bash
# 查看任务状态
squeue -u $(whoami)

# 查看输出
tail -f log/test_aidi_yagi35.out
tail -f log/test_aidi_yagi38.out
tail -f log/test_aidi_yagi39.out
tail -f log/test_aidi_yagi41.out

# 查看所有输出
tail -f log/test_aidi_yagi*.out
```

## 输出文件

- `log/test_aidi_yagi35.out` - yagi35 输出
- `log/test_aidi_yagi38.out` - yagi38 输出
- `log/test_aidi_yagi39.out` - yagi39 输出
- `log/test_aidi_yagi41.out` - yagi41 输出

## 结果格式

每个节点的输出类似：
```
3.737744691534317e-06
total_time: 85.81229710578918
avg_time: 0.12258899586541312
```

## 注意

- yagi41 有 6 个 GPU，配置不同，已排除
- 脚本会自动检测节点所在分区
- 使用 conda 环境 `afpi`
