# -*- coding: utf-8 -*-
from model_loader import ModelLoader
from qa_system import QASystem
from config import DEFAULT_MODEL_NAME, USE_QUANTIZATION
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_usage():
    print("=" * 60)
    print("基本使用示例")
    print("=" * 60)
    
    print("\n1. 加载模型...")
    model_loader = ModelLoader(
        model_name=DEFAULT_MODEL_NAME,
        use_quantization=USE_QUANTIZATION
    )
    model_loader.load_model()
    
    print("\n2. 初始化问答系统...")
    qa = QASystem(
        model=model_loader.get_model(),
        tokenizer=model_loader.get_tokenizer()
    )
    
    print("\n3. 开始对话...")
    questions = [
        "你好，请介绍一下你自己",
        "Python是什么？",
        "如何使用Docker部署应用？"
    ]
    
    for question in questions:
        print(f"\n问题: {question}")
        answer = qa.generate_answer(question)
        print(f"回答: {answer}")
        print("-" * 60)
    
    print("\n4. 对话历史:")
    history = qa.get_history()
    for i, item in enumerate(history, 1):
        print(f"\n[{i}]")
        print(f"  问题: {item['question']}")
        print(f"  回答: {item['answer'][:100]}...")


def example_api_usage():
    import requests
    
    print("=" * 60)
    print("API使用示例")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    print("\n1. 初始化系统...")
    init_response = requests.post(
        f"{base_url}/api/init",
        json={
            "model_name": DEFAULT_MODEL_NAME,
            "use_quantization": USE_QUANTIZATION
        }
    )
    print(f"初始化结果: {init_response.json()}")
    
    print("\n2. 发送问题...")
    chat_response = requests.post(
        f"{base_url}/api/chat",
        json={
            "question": "你好，请用一句话介绍你自己",
            "max_length": 256,
            "temperature": 0.7
        }
    )
    result = chat_response.json()
    print(f"问题: 你好，请用一句话介绍你自己")
    print(f"回答: {result['answer']}")
    
    print("\n3. 获取对话历史...")
    history_response = requests.get(f"{base_url}/api/history")
    print(f"历史记录: {history_response.json()}")


if __name__ == "__main__":
    print("\n请选择运行模式：")
    print("1. 基本使用示例（直接加载模型）")
    print("2. API使用示例（需要先启动api_server.py）")
    
    choice = input("\n请输入选项 (1 或 2): ").strip()
    
    if choice == "1":
        example_basic_usage()
    elif choice == "2":
        example_api_usage()
    else:
        print("无效选项，运行基本使用示例...")
        example_basic_usage()

