# -*- coding: utf-8 -*-
import json
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lora import LoRATrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


sample_data = [
    {"text": "<s>[INST] <<SYS>>\n你是一个有用的AI助手，请用中文回答问题。\n<</SYS>>\n\n请介绍一下你自己 [/INST] 我是一个基于Llama2模型开发的AI助手，专门用于中文问答。我可以帮助您解答各种问题，包括技术咨询、知识问答、文本生成等。我经过了专门的微调训练，能够更好地理解中文语境并提供准确的回答。 </s>"},
    {"text": "<s>[INST] <<SYS>>\n你是一个有用的AI助手，请用中文回答问题。\n<</SYS>>\n\n什么是机器学习？ [/INST] 机器学习是人工智能的一个分支，它使计算机能够从数据中学习，而无需显式编程。通过分析大量数据，机器学习算法可以识别模式、做出预测并改进性能。常见的机器学习类型包括监督学习、无监督学习和强化学习。 </s>"},
    {"text": "<s>[INST] <<SYS>>\n你是一个有用的AI助手，请用中文回答问题。\n<</SYS>>\n\n如何学习Python编程？ [/INST] 学习Python编程可以按照以下步骤：1) 先了解Python基础语法和概念；2) 通过实践编写小程序；3) 学习常用库如NumPy、Pandas；4) 参与开源项目；5) 持续练习和解决问题。推荐从官方文档和在线教程开始。 </s>"},
    {"text": "<s>[INST] <<SYS>>\n你是一个有用的AI助手，请用中文回答问题。\n<</SYS>>\n\n解释一下Docker的作用 [/INST] Docker是一个容器化平台，它允许开发者将应用程序及其依赖项打包成轻量级、可移植的容器。Docker的主要作用包括：1) 环境一致性，确保开发、测试、生产环境相同；2) 隔离性，不同容器互不干扰；3) 资源效率，比虚拟机更轻量；4) 快速部署和扩展。 </s>"},
    {"text": "<s>[INST] <<SYS>>\n你是一个有用的AI助手，请用中文回答问题。\n<</SYS>>\n\n深度学习与机器学习的区别是什么？ [/INST] 深度学习是机器学习的一个子集。主要区别：1) 深度学习使用多层神经网络，而传统机器学习可能使用简单算法；2) 深度学习需要大量数据和计算资源；3) 深度学习可以自动学习特征表示，而传统机器学习通常需要手动特征工程；4) 深度学习在图像、语音等领域表现更好。 </s>"},
    {"text": "<s>[INST] <<SYS>>\n你是一个有用的AI助手，请用中文回答问题。\n<</SYS>>\n\n什么是API？ [/INST] API（应用程序编程接口）是一组定义和协议，用于构建和集成应用软件。它定义了应用程序如何与其他应用程序或服务交互。API允许不同的软件系统相互通信，共享数据和功能。常见的API类型包括REST API、GraphQL API等。 </s>"},
    {"text": "<s>[INST] <<SYS>>\n你是一个有用的AI助手，请用中文回答问题。\n<</SYS>>\n\n如何优化数据库查询性能？ [/INST] 优化数据库查询性能的方法：1) 创建适当的索引；2) 避免SELECT *，只选择需要的列；3) 使用JOIN代替子查询；4) 避免在WHERE子句中使用函数；5) 使用EXPLAIN分析查询计划；6) 考虑查询缓存；7) 定期分析表以更新统计信息。 </s>"},
    {"text": "<s>[INST] <<SYS>>\n你是一个有用的AI助手，请用中文回答问题。\n<</SYS>>\n\n什么是微服务架构？ [/INST] 微服务架构是一种将应用程序构建为小型、独立服务集合的架构风格。每个服务运行在自己的进程中，通过轻量级机制（如HTTP API）通信。优点包括：1) 独立开发和部署；2) 技术栈多样化；3) 可扩展性好；4) 故障隔离。但也带来复杂性增加等挑战。 </s>"},
]


def create_training_data(output_file: str = "data/sample/train.jsonl"):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"创建了训练数据文件: {output_path}")
    logger.info(f"数据量: {len(sample_data)} 条")
    return str(output_path)


def run_finetune_example(use_quantization=True, base_model=None, epochs=1):
    logger.info("=" * 60)
    logger.info("LoRA微调示例")
    logger.info("=" * 60)
    
    data_file = create_training_data()
    
    logger.info("\n开始LoRA微调训练...")
    logger.info("注意：这是一个小数据集示例，实际使用中建议使用更多数据")
    
    if base_model is None:
        base_model = "meta-llama/Llama-2-7b-chat-hf"
    
    trainer = LoRATrainer(
        base_model_name=base_model,
        output_dir="models/finetuned_example",
        lora_r=8,
        lora_alpha=16
    )
    
    trainer.train(
        train_data_path=data_file,
        val_data_path=None,
        num_epochs=epochs,
        batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_steps=10,
        save_steps=50,
        logging_steps=10,
        use_quantization=use_quantization
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("微调完成！")
    logger.info(f"模型保存在: models/finetuned_example")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default=None,
                       help="基础模型名称，默认为meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--epochs", type=int, default=1,
                       help="训练轮数，默认1轮")
    parser.add_argument("--no-quantization", action="store_true",
                       help="不使用量化（需要更多显存）")
    
    args = parser.parse_args()
    
    run_finetune_example(
        use_quantization=not args.no_quantization,
        base_model=args.base_model,
        epochs=args.epochs
    )

