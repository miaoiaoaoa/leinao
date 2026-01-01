# 本地自然语言处理应答系统

基于Docker与Llama2-7B的智能问答系统实现

## 项目简介

这是一个本地化部署的自然语言处理问答系统，基于Llama2-7B模型实现。系统支持：
- 本地化部署，无需依赖外部API
- 基于Llama2-7B的智能问答
- 图形化Web界面（Gradio）
- RESTful API接口（FastAPI）
- Docker容器化部署
- 4-bit量化支持，降低显存需求

## 技术栈

- **模型**: Llama2-7B-Chat
- **框架**: Transformers, PyTorch
- **界面**: Gradio
- **API**: FastAPI
- **部署**: Docker, Docker Compose
- **量化**: BitsAndBytes (4-bit量化)

## 项目结构

```
.
├── app.py                 # Gradio图形界面主程序
├── api_server.py          # FastAPI REST API服务器
├── model_loader.py        # 模型加载模块
├── qa_system.py          # 问答系统核心模块
├── requirements.txt      # Python依赖
├── Dockerfile            # Docker镜像构建文件
├── docker-compose.yml    # Docker Compose配置
└── README.md            # 项目文档
```

## 安装与使用

### 方式一：本地运行

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **运行Gradio界面**
```bash
python app.py
```
访问 http://localhost:7860

3. **运行API服务器**
```bash
python api_server.py
```
API文档: http://localhost:8000/docs

### 方式二：Docker部署

1. **构建镜像**
```bash
docker build -t nlp-qa-system .
```

2. **运行容器**
```bash
docker-compose up -d
```

3. **访问服务**
- Gradio界面: http://localhost:7860
- API接口: http://localhost:8000

## API使用说明

### 初始化系统
```bash
POST /api/init
{
    "model_name": "meta-llama/Llama-2-7b-chat-hf",
    "use_quantization": true
}
```

### 发送问题
```bash
POST /api/chat
{
    "question": "你好，请介绍一下你自己",
    "max_length": 512,
    "temperature": 0.7,
    "top_p": 0.9
}
```

### 清空对话历史
```bash
POST /api/clear_history
```

### 获取对话历史
```bash
GET /api/history
```

## 代码使用示例

### 基本使用

```python
from model_loader import ModelLoader
from qa_system import QASystem

# 加载模型
model_loader = ModelLoader(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    use_quantization=True
)
model_loader.load_model()

# 创建问答系统
qa = QASystem(
    model=model_loader.get_model(),
    tokenizer=model_loader.get_tokenizer()
)

# 提问
answer = qa.generate_answer("你好，请介绍一下你自己")
print(answer)
```

更多示例请参考 `example_usage.py` 文件。

## 配置说明

### 模型配置

- **model_name**: 模型名称或本地路径
  - 默认: `meta-llama/Llama-2-7b-chat-hf`
  - 可以使用HuggingFace模型ID或本地路径

- **use_quantization**: 是否使用4-bit量化
  - `true`: 使用量化，显存需求约4-6GB
  - `false`: 不使用量化，显存需求约14-16GB

### 生成参数

- **max_length**: 最大生成长度（默认512）
- **temperature**: 温度参数，控制随机性（0.1-1.0，默认0.7）
- **top_p**: nucleus sampling参数（0.0-1.0，默认0.9）

## 系统要求

### 最低配置
- CPU: 8核
- 内存: 16GB RAM
- 显存: 4GB VRAM (使用量化) / 16GB VRAM (不使用量化)
- 磁盘: 20GB 可用空间

### 推荐配置
- CPU: 16核+
- 内存: 32GB+ RAM
- 显存: 8GB+ VRAM (使用量化)
- GPU: NVIDIA GPU (CUDA支持)

## 注意事项

1. **模型下载**: 首次运行需要下载Llama2-7B模型（约13GB），请确保网络连接正常
2. **显存优化**: 如果显存不足，建议使用4-bit量化模式
3. **模型许可**: Llama2模型需要HuggingFace账号和访问许可
4. **GPU加速**: 推荐使用NVIDIA GPU以获得更好的性能

## 代码使用示例

### 基本使用

```python
from model_loader import ModelLoader
from qa_system import QASystem

# 加载模型
model_loader = ModelLoader(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    use_quantization=True
)
model_loader.load_model()

# 创建问答系统
qa = QASystem(
    model=model_loader.get_model(),
    tokenizer=model_loader.get_tokenizer()
)

# 提问
answer = qa.generate_answer("你好，请介绍一下你自己")
print(answer)
```

更多示例请参考 `example_usage.py` 文件。

## 模型微调

本项目提供了完整的模型微调工具链，支持使用LoRA方法对Llama2模型进行微调。

### 快速开始微调

1. **准备数据**
```bash
# 创建示例数据（可选）
python finetune/create_sample_data.py

# 准备训练数据
python finetune/prepare_data.py \
    --input data/raw_data.json \
    --input-format json \
    --format-type alpaca
```

2. **开始微调**
```bash
python finetune/train_lora.py \
    --base-model meta-llama/Llama-2-7b-chat-hf \
    --train-data data/finetune/train_train.jsonl \
    --val-data data/finetune/train_validation.jsonl \
    --output-dir models/finetuned \
    --epochs 3
```

3. **评估模型**
```bash
python finetune/evaluate_model.py \
    --base-model meta-llama/Llama-2-7b-chat-hf \
    --lora-model models/finetuned/lora_adapter \
    --interactive
```

详细的微调指南请参考 [finetune/README.md](finetune/README.md)

## 常见问题

### Q: 模型加载失败？
A: 检查网络连接，确保可以访问HuggingFace。如果使用本地模型，检查路径是否正确。

### Q: 显存不足？
A: 启用4-bit量化模式（`use_quantization=true`），或使用更小的模型。

### Q: 生成速度慢？
A: 使用GPU可以显著提升速度。如果没有GPU，可以考虑使用更小的模型或增加硬件资源。

### Q: 如何进行模型微调？
A: 参考 `finetune/README.md` 中的详细说明。主要步骤：准备数据 -> 训练 -> 评估 -> 部署。

## 许可证

本项目仅供学习和研究使用。Llama2模型的使用需遵守Meta的许可证协议。

## 作者

苗雨健
2025/12/14

