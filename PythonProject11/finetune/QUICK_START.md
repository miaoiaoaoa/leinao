# 微调快速开始指南

本指南将帮助您快速了解和使用模型微调脚本。

## 脚本概览

| 脚本 | 功能 | 用途 |
|------|------|------|
| `create_sample_data.py` | 创建示例数据 | 快速生成测试数据 |
| `prepare_data.py` | 数据预处理 | 将原始数据转换为训练格式 |
| `train_lora.py` | LoRA微调训练 | 执行模型微调 |
| `evaluate_model.py` | 模型评估 | 评估微调效果 |
| `convert_to_gguf.py` | 格式转换 | 转换为Ollama可用的GGUF格式 |

## 完整流程示例

### 步骤1: 创建示例数据（用于测试）

```bash
python finetune/create_sample_data.py --output-dir data/sample
```

这将创建10条示例中文问答数据，用于快速测试微调流程。

### 步骤2: 准备训练数据

```bash
python finetune/prepare_data.py \
    --input data/sample/sample_data.json \
    --input-format json \
    --format-type alpaca \
    --output-dir data/finetune \
    --output-name train
```

这会：
- 读取JSON格式的数据
- 转换为Llama2的训练格式
- 自动划分训练集/验证集/测试集
- 保存为JSONL格式到 `data/finetune/` 目录

### 步骤3: 执行微调训练

```bash
python finetune/train_lora.py \
    --base-model meta-llama/Llama-2-7b-chat-hf \
    --train-data data/finetune/train_train.jsonl \
    --val-data data/finetune/train_validation.jsonl \
    --output-dir models/finetuned \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-4 \
    --lora-r 8 \
    --lora-alpha 16
```

如果显存不足，添加 `--use-quantization` 参数。

### 步骤4: 评估模型

**交互式评估**（推荐先测试）：
```bash
python finetune/evaluate_model.py \
    --base-model meta-llama/Llama-2-7b-chat-hf \
    --lora-model models/finetuned/lora_adapter \
    --interactive
```

**批量评估**：
```bash
python finetune/evaluate_model.py \
    --base-model meta-llama/Llama-2-7b-chat-hf \
    --lora-model models/finetuned/lora_adapter \
    --test-data data/finetune/train_test.jsonl \
    --output evaluation_results.csv
```

### 步骤5: 转换为GGUF格式（供Ollama使用）

**注意**: 此步骤需要先合并LoRA适配器，然后使用llama.cpp工具转换。

```bash
# 1. 合并LoRA适配器（在evaluate_model.py中实现merge功能）
# 2. 使用llama.cpp转换（需要安装llama.cpp）
python finetune/convert_to_gguf.py \
    --model-path models/finetuned/merged_model \
    --quantization Q4_K_M \
    --create-modelfile \
    --model-name llama2-finetuned
```

## 验收检查清单

使用以下清单确保微调流程完整：

- [ ] **数据准备**
  - [ ] 准备了至少100条高质量的训练数据
  - [ ] 数据格式正确（instruction/input/output）
  - [ ] 数据已划分为训练/验证/测试集

- [ ] **模型训练**
  - [ ] 训练配置合理（epochs, learning_rate等）
  - [ ] 训练loss正常下降
  - [ ] 验证loss正常下降
  - [ ] 模型已保存到指定目录

- [ ] **模型评估**
  - [ ] 在测试集上进行了评估
  - [ ] 记录了评估指标（loss, 回答质量等）
  - [ ] 进行了人工评估和对比

- [ ] **模型部署**
  - [ ] 模型可以正常加载
  - [ ] 推理速度满足要求
  - [ ] 已回答质量有明显提升（如有数据支持）

- [ ] **文档记录**
  - [ ] 填写了训练报告（参考 `training_report.md`）
  - [ ] 记录了关键参数和结果
  - [ ] 保存了训练日志

## 常见问题

### Q: 训练数据需要多少条？
A: 最少建议100条，理想情况下1000-10000条。数据质量比数量更重要。

### Q: 训练需要多长时间？
A: 取决于数据量和硬件：
- 100条数据 + GPU: 约30分钟-1小时
- 1000条数据 + GPU: 约2-5小时
- CPU训练会慢很多，不推荐

### Q: 如何判断训练是否成功？
A: 观察验证loss：
- loss持续下降：训练正常
- loss不再下降：可能已经收敛，可以停止
- loss上升：可能过拟合，需要早停

### Q: 微调后模型太大怎么办？
A: 使用量化转换：
- Q4_K_M: 压缩到约4GB，质量损失较小
- Q5_K_M: 压缩到约5GB，质量更好
- Q8_0: 压缩到约8GB，质量接近原模型

## 验收建议

为了顺利通过验收，建议：

1. **准备充足的数据**：至少准备100-500条高质量的中文问答数据
2. **完整记录过程**：使用 `training_report.md` 模板记录训练过程
3. **对比评估**：展示微调前后的效果对比
4. **演示功能**：准备一些示例问题和回答进行演示
5. **说明改进**：明确说明微调带来的改进点

## 获取帮助

如果遇到问题：
1. 查看 `finetune/README.md` 获取详细文档
2. 检查训练日志中的错误信息
3. 确认依赖已正确安装：`pip install -r requirements.txt`

