# -*- coding: utf-8 -*-
import json
from pathlib import Path

sample_data = [
    {
        "instruction": "请介绍一下你自己",
        "input": "",
        "output": "我是一个基于Llama2模型开发的AI助手，专门用于中文问答。我可以帮助您解答各种问题，包括技术咨询、知识问答、文本生成等。我经过了专门的微调训练，能够更好地理解中文语境并提供准确的回答。"
    },
    {
        "instruction": "什么是机器学习？",
        "input": "",
        "output": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习，而无需显式编程。通过分析大量数据，机器学习算法可以识别模式、做出预测并改进性能。常见的机器学习类型包括监督学习、无监督学习和强化学习。"
    },
    {
        "instruction": "如何学习Python编程？",
        "input": "",
        "output": "学习Python编程可以按照以下步骤：1) 先了解Python基础语法和概念；2) 通过实践编写小程序；3) 学习常用库如NumPy、Pandas；4) 参与开源项目；5) 持续练习和解决问题。推荐从官方文档和在线教程开始。"
    },
    {
        "instruction": "解释一下Docker的作用",
        "input": "",
        "output": "Docker是一个容器化平台，它允许开发者将应用程序及其依赖项打包成轻量级、可移植的容器。Docker的主要作用包括：1) 环境一致性，确保开发、测试、生产环境相同；2) 隔离性，不同容器互不干扰；3) 资源效率，比虚拟机更轻量；4) 快速部署和扩展。"
    },
    {
        "instruction": "深度学习与机器学习的区别是什么？",
        "input": "",
        "output": "深度学习是机器学习的一个子集。主要区别：1) 深度学习使用多层神经网络，而传统机器学习可能使用简单算法；2) 深度学习需要大量数据和计算资源；3) 深度学习可以自动学习特征表示，而传统机器学习通常需要手动特征工程；4) 深度学习在图像、语音等领域表现更好。"
    },
    {
        "instruction": "什么是API？",
        "input": "",
        "output": "API（应用程序编程接口）是一组定义和协议，用于构建和集成应用软件。它定义了应用程序如何与其他应用程序或服务交互。API允许不同的软件系统相互通信，共享数据和功能。常见的API类型包括REST API、GraphQL API等。"
    },
    {
        "instruction": "如何优化数据库查询性能？",
        "input": "",
        "output": "优化数据库查询性能的方法：1) 创建适当的索引；2) 避免SELECT *，只选择需要的列；3) 使用JOIN代替子查询；4) 避免在WHERE子句中使用函数；5) 使用EXPLAIN分析查询计划；6) 考虑查询缓存；7) 定期分析表以更新统计信息。"
    },
    {
        "instruction": "什么是微服务架构？",
        "input": "",
        "output": "微服务架构是一种将应用程序构建为小型、独立服务集合的架构风格。每个服务运行在自己的进程中，通过轻量级机制（如HTTP API）通信。优点包括：1) 独立开发和部署；2) 技术栈多样化；3) 可扩展性好；4) 故障隔离。但也带来复杂性增加等挑战。"
    },
    {
        "instruction": "Git和GitHub的区别",
        "input": "",
        "output": "Git是一个分布式版本控制系统，用于跟踪代码更改。GitHub是一个基于Git的代码托管平台，提供Git仓库托管服务。主要区别：Git是工具，GitHub是服务。Git可以本地使用，GitHub需要网络访问。GitHub还提供协作功能如Pull Request、Issue跟踪等。"
    },
    {
        "instruction": "如何保证代码质量？",
        "input": "",
        "output": "保证代码质量的方法：1) 编写清晰的代码注释；2) 遵循编码规范和最佳实践；3) 进行代码审查；4) 编写单元测试和集成测试；5) 使用静态代码分析工具；6) 重构技术债务；7) 持续集成和持续部署；8) 性能监控和日志记录。"
    },
]


def create_sample_data(output_dir: str = "data/sample"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    json_file = output_path / "sample_data.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    print(f"已创建示例数据文件: {json_file}")
    print(f"数据量: {len(sample_data)} 条")
    
    jsonl_file = output_path / "sample_data.jsonl"
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"已创建JSONL格式文件: {jsonl_file}")
    
    return json_file, jsonl_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/sample")
    
    args = parser.parse_args()
    
    create_sample_data(args.output_dir)
    print("\n提示：可以使用以下命令准备训练数据：")
    print(f"python finetune/prepare_data.py --input {args.output_dir}/sample_data.json --input-format json")

