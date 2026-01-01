# 模型优化指南：裁剪与蒸馏技术

本指南详细说明如何使用裁剪与蒸馏技术优化Llama2模型，适配中低端服务器和边缘设备。

## 技术概览

### 1. 模型裁剪（Model Pruning）

**目标**: 减少模型参数量，降低显存和计算需求

**原理**: 
- 基于幅度的裁剪：将权重值较小的参数置零
- 结构化裁剪：移除整个注意力头等结构单元

**优势**:
- 参数量减少 20-30%
- 显存需求降低
- 推理速度提升
- 适配边缘设备

**适用场景**:
- 资源受限的服务器
- 边缘计算设备
- 移动设备部署

### 2. 知识蒸馏（Knowledge Distillation）

**目标**: 将大模型的知识迁移到小模型，在减少参数量的同时保持效果

**原理**:
- 使用大模型（教师）的软标签训练小模型（学生）
- 通过KL散度损失学习教师模型的概率分布
- 结合硬标签损失（真实标签）和软标签损失（教师输出）

**优势**:
- 保持模型性能
- 减少参数量
- 提高推理速度
- 降低部署成本

**适用场景**:
- 需要在保持效果的前提下减小模型
- 模型压缩需求
- 边缘设备部署

### 3. LoRA微调

**目标**: 参数高效微调，降低训练显存需求

**原理**:
- 只训练少量低秩适配器参数
- 冻结原始模型参数
- 通过矩阵分解减少可训练参数

**优势**:
- 可训练参数减少 99%+
- 显存需求大幅降低
- 训练速度快
- 效果接近全量微调

## 完整优化流程

### 流程1：裁剪 + LoRA微调（推荐）

适用于：需要减少参数量并保持效果

```bash
python finetune/train_pruned_distilled.py \
    --base-model meta-llama/Llama-2-7b-chat-hf \
    --train-data data/finetune/train_train.jsonl \
    --val-data data/finetune/train_validation.jsonl \
    --output-dir models/optimized \
    --pruning-ratio 0.2 \
    --lora-r 8 \
    --lora-alpha 16 \
    --epochs 3 \
    --batch-size 4
```

**效果**:
- 参数量减少: ~20%
- 可训练参数: 仅 LoRA 适配器 (~1%)
- 显存需求: 大幅降低
- 效果损失: 最小

### 流程2：知识蒸馏 + LoRA微调

适用于：需要将大模型压缩为小模型

```bash
python finetune/train_pruned_distilled.py \
    --base-model path/to/smaller_student_model \
    --train-data data/finetune/train_train.jsonl \
    --val-data data/finetune/train_validation.jsonl \
    --output-dir models/distilled \
    --use-distillation \
    --teacher-model meta-llama/Llama-2-7b-chat-hf \
    --distillation-temperature 5.0 \
    --distillation-alpha 0.5 \
    --lora-r 8 \
    --epochs 3
```

**效果**:
- 学生模型参数量: 约为教师模型的 50-70%
- 性能保持: 通过蒸馏保持 90%+ 的效果
- 推理速度: 提升 2-3 倍

### 流程3：裁剪 + 蒸馏 + LoRA（最完整）

适用于：最大化优化效果

```bash
python finetune/train_pruned_distilled.py \
    --base-model meta-llama/Llama-2-7b-chat-hf \
    --train-data data/finetune/train_train.jsonl \
    --val-data data/finetune/train_validation.jsonl \
    --output-dir models/fully_optimized \
    --pruning-ratio 0.2 \
    --use-distillation \
    --teacher-model meta-llama/Llama-2-7b-chat-hf \
    --lora-r 8 \
    --epochs 3
```

**效果**:
- 参数量减少: ~20-30%
- 效果保持: 通过蒸馏最小化损失
- 训练成本: LoRA大幅降低
- 部署成本: 显著降低

## 参数调优建议

### 裁剪比例（pruning_ratio）

| 比例 | 参数量减少 | 效果损失 | 推荐场景 |
|------|-----------|---------|---------|
| 0.1  | ~10%      | 几乎无   | 轻度优化 |
| 0.2  | ~20%      | 很小     | **推荐** |
| 0.3  | ~30%      | 中等     | 资源严重受限 |
| 0.4+ | ~40%+     | 较大     | 不推荐 |

### 蒸馏参数

- **temperature**: 通常 3.0-7.0，越高软化程度越大
  - 推荐: 5.0（平衡点）
- **alpha**: 蒸馏损失权重，通常 0.3-0.7
  - 推荐: 0.5（平衡软硬标签）

### LoRA参数

- **r**: LoRA rank，通常 4-16
  - 资源受限: r=4
  - **推荐**: r=8
  - 需要更强拟合: r=16
- **alpha**: 通常设为 r 的 2 倍

## 硬件适配对比

### 原始模型（Llama2-7B）

- 显存需求: ~14GB（FP16）
- 参数量: 7B
- 推理速度: 基准

### 裁剪后（20%裁剪）

- 显存需求: ~11GB
- 参数量: ~5.6B
- 推理速度: +25%

### LoRA微调（r=8）

- 训练显存: ~8GB（相比全量微调的 ~16GB）
- 可训练参数: ~4M（仅0.06%）
- 训练速度: +3-5倍

### 组合优化（裁剪20% + LoRA）

- 训练显存: ~6-7GB
- 参数量: ~5.6B（其中仅4M可训练）
- 推理显存: ~9GB
- 推理速度: +30%

## 效果评估

### 评估指标

1. **参数量对比**
   ```python
   # 原始模型
   original_params = 7_000_000_000
   
   # 裁剪后
   pruned_params = original_params * 0.8  # 20%裁剪
   
   # LoRA可训练参数
   lora_params = 4_000_000
   
   # 压缩比
   compression_ratio = original_params / pruned_params
   ```

2. **性能对比**
   - 使用测试集评估微调前后的BLEU/ROUGE分数
   - 人工评估回答质量
   - 对比回答准确率

3. **资源使用对比**
   - 显存占用
   - 推理延迟
   - 吞吐量

### 验收建议

为了通过验收，建议准备：

1. **数据对比表**
   - 参数量对比（原始 vs 优化后）
   - 显存需求对比
   - 推理速度对比

2. **效果对比**
   - 测试集指标对比
   - 示例问答对比（微调前 vs 微调后）
   - 性能损失评估

3. **部署验证**
   - 在目标设备上的实际部署
   - 性能测试报告
   - 资源使用监控

## 常见问题

### Q: 裁剪会不会大幅降低效果？

A: 适度裁剪（20-30%）通常只会带来5-10%的效果损失，通过知识蒸馏可以进一步降低损失。

### Q: 裁剪和蒸馏哪个更好？

A: 
- 裁剪：更简单，直接减少参数
- 蒸馏：需要教师模型，但能更好保持效果
- 组合使用：最佳效果

### Q: 如何选择裁剪比例？

A: 
- 首先尝试 0.2（20%）
- 评估效果，如果可接受可以尝试 0.3
- 不建议超过 0.4

### Q: 蒸馏需要训练多长时间？

A: 通常需要3-5个epoch，时间与普通微调相当。

### Q: 优化后的模型可以用于Ollama吗？

A: 可以，使用 `convert_to_gguf.py` 转换为GGUF格式后导入Ollama。

## 参考

- [LoRA论文](https://arxiv.org/abs/2106.09685)
- [知识蒸馏综述](https://arxiv.org/abs/2006.05525)
- [模型裁剪技术](https://arxiv.org/abs/1506.02626)

