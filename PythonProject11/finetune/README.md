# 模型微调指南

本目录包含用于对Llama2模型进行微调的完整脚本和工具。

## 目录结构

```
finetune/
├── prepare_data.py          # 数据准备脚本
├── train_lora.py            # LoRA微调训练脚本
├── evaluate_model.py        # 模型评估脚本
├── convert_to_gguf.py       # 转换为GGUF格式（供Ollama使用）
├── model_pruning.py         # 模型裁剪脚本
├── knowledge_distillation.py # 知识蒸馏脚本
├── train_pruned_distilled.py # 整合训练（裁剪+蒸馏+LoRA）
├── create_sample_data.py    # 创建示例数据
├── config.yaml              # 配置文件
├── training_report.md       # 训练报告模板
├── QUICK_START.md          # 快速开始指南
├── OPTIMIZATION_GUIDE.md   # 优化技术指南
└── README.md               # 本文件
```

## 核心特性

本项目实现了PPTX中提到的技术创新点：

1. **模型裁剪（Pruning）**: 基于幅度的裁剪技术，减少模型参数量
2. **知识蒸馏（Distillation）**: 将大模型知识迁移到小模型，保持效果
3. **LoRA微调**: 参数高效微调，降低显存需求
4. **整合优化**: 支持裁剪+蒸馏+LoRA的组合使用

这些技术可以适配中低端服务器和边缘设备，显著降低硬件门槛。

## 快速开始

### 1. 安装依赖

```bash
pip install transformers torch peft datasets accelerate bitsandbytes
```

### 2. 准备数据

数据格式示例（JSON格式）：
```json
[
  {
    "instruction": "请解释什么是机器学习",
    "input": "",
    "output": "机器学习是人工智能的一个分支..."
  },
  {
    "instruction": "如何制作咖啡？",
    "input": "使用咖啡机",
    "output": "首先准备咖啡豆和水..."
  }
]
```

转换为训练格式：
```bash
python finetune/prepare_data.py \
    --input data/raw_data.json \
    --input-format json \
    --format-type alpaca \
    --output-dir data/finetune \
    --output-name train
```

### 3. 开始微调

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

如果显存不足，可以使用量化：
```bash
python finetune/train_lora.py \
    --base-model meta-llama/Llama-2-7b-chat-hf \
    --train-data data/finetune/train_train.jsonl \
    --output-dir models/finetuned \
    --use-quantization
```

### 4. 评估模型

交互式评估：
```bash
python finetune/evaluate_model.py \
    --base-model meta-llama/Llama-2-7b-chat-hf \
    --lora-model models/finetuned/lora_adapter \
    --interactive
```

在测试集上评估：
```bash
python finetune/evaluate_model.py \
    --base-model meta-llama/Llama-2-7b-chat-hf \
    --lora-model models/finetuned/lora_adapter \
    --test-data data/finetune/train_test.jsonl \
    --output evaluation_results.csv
```

### 5. 转换为GGUF格式（供Ollama使用）

**注意**：实际转换需要安装llama.cpp工具。

```bash
# 首先合并LoRA适配器到基础模型（使用evaluate_model.py中的merge功能）
# 然后使用llama.cpp转换
python finetune/convert_to_gguf.py \
    --model-path models/finetuned/merged_model \
    --quantization Q4_K_M \
    --create-modelfile \
    --model-name llama2-finetuned
```

导入到Ollama：
```bash
ollama create llama2-finetuned -f Modelfile
```

## 详细说明

### 数据准备

`prepare_data.py` 支持多种输入格式：
- JSON格式：`{"instruction": "...", "input": "...", "output": "..."}`
- JSONL格式：每行一个JSON对象
- CSV格式：需要指定列名

支持的格式类型：
- `alpaca`: Alpaca格式（instruction, input, output）
- `qa`: 问答格式（question, answer）

### LoRA微调

LoRA（Low-Rank Adaptation）是一种参数高效的微调方法：
- **r (rank)**: LoRA的秩，通常设置为8或16
- **alpha**: LoRA的缩放参数，通常设置为r的2倍
- **dropout**: 防止过拟合的dropout率

训练参数说明：
- `batch_size`: 批次大小，根据显存调整
- `gradient_accumulation_steps`: 梯度累积步数，相当于增大批次大小
- `learning_rate`: 学习率，LoRA通常使用2e-4
- `num_epochs`: 训练轮数，通常3-5轮

### 模型评估

评估脚本支持两种模式：
1. **交互式模式**：手动输入问题测试模型
2. **批量评估**：在测试集上批量评估并保存结果

### 转换为GGUF格式

GGUF是Ollama使用的模型格式。转换流程：
1. 合并LoRA适配器到基础模型
2. 使用llama.cpp的转换工具转换为GGUF
3. 创建Modelfile定义模型配置
4. 导入到Ollama

量化选项：
- `Q4_0`: 4-bit量化，速度快，质量中等
- `Q4_K_M`: 4-bit量化，平衡版本
- `Q5_K_M`: 5-bit量化，质量更好
- `Q8_0`: 8-bit量化，质量最好但文件较大

## 系统要求

### 最低配置（使用量化）
- GPU: 8GB VRAM
- 内存: 16GB RAM
- 磁盘: 50GB 可用空间

### 推荐配置
- GPU: 16GB+ VRAM (不使用量化)
- 内存: 32GB+ RAM
- 磁盘: 100GB+ 可用空间

## 常见问题

### Q: 训练时显存不足？
A: 使用 `--use-quantization` 启用4-bit量化，或减小batch_size。

### Q: 如何选择LoRA参数？
A: 
- r=8, alpha=16 适合大多数场景
- 需要更强拟合能力时使用 r=16, alpha=32
- 显存受限时使用 r=4, alpha=8

### Q: 训练需要多长时间？
A: 取决于数据量和硬件。通常：
- 1000条数据：约1-2小时（GPU）
- 10000条数据：约5-10小时（GPU）

### Q: 如何知道模型训练好了？
A: 观察验证集loss，当loss不再下降或开始上升时停止训练。

## 验收检查清单

- [ ] 数据准备完成（训练/验证/测试集已划分）
- [ ] 微调训练完成（loss收敛）
- [ ] 模型评估完成（测试集结果）
- [ ] 模型转换为GGUF格式（如需要）
- [ ] 模型已导入Ollama并测试
- [ ] 性能对比报告（微调前后）

## 参考资料

- [LoRA论文](https://arxiv.org/abs/2106.09685)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Ollama文档](https://github.com/ollama/ollama)

