# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from model_loader import ModelLoader
from qa_system import QASystem
from config import (
    DEFAULT_MODEL_NAME, USE_QUANTIZATION,
    API_HOST, API_PORT
)
import logging
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="本地NLP问答系统API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qa_system: Optional[QASystem] = None
model_loader: Optional[ModelLoader] = None


class QuestionRequest(BaseModel):
    question: str
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9


class AnswerResponse(BaseModel):
    answer: str
    status: str
    message: str


class InitRequest(BaseModel):
    model_name: str = DEFAULT_MODEL_NAME
    use_quantization: bool = USE_QUANTIZATION


@app.get("/")
async def root():
    return {
        "message": "本地NLP问答系统API",
        "status": "running",
        "model_loaded": qa_system is not None
    }


@app.post("/api/init", response_model=AnswerResponse)
async def init_system(request: InitRequest):
    global qa_system, model_loader
    
    try:
        logger.info(f"初始化系统，模型: {request.model_name}")
        
        model_loader = ModelLoader(
            model_name=request.model_name,
            use_quantization=request.use_quantization
        )
        model_loader.load_model()
        
        qa_system = QASystem(
            model=model_loader.get_model(),
            tokenizer=model_loader.get_tokenizer()
        )
        
        return AnswerResponse(
            answer="",
            status="success",
            message="系统初始化成功"
        )
        
    except Exception as e:
        logger.error(f"初始化失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"初始化失败: {str(e)}")


@app.post("/api/chat", response_model=AnswerResponse)
async def chat(request: QuestionRequest):
    global qa_system
    
    if qa_system is None:
        raise HTTPException(
            status_code=400,
            detail="系统尚未初始化，请先调用 /api/init 接口"
        )
    
    try:
        answer = qa_system.generate_answer(
            question=request.question,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return AnswerResponse(
            answer=answer,
            status="success",
            message="回答生成成功"
        )
        
    except Exception as e:
        logger.error(f"生成答案失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成答案失败: {str(e)}")


@app.post("/api/clear_history")
async def clear_history():
    global qa_system
    
    if qa_system is None:
        raise HTTPException(
            status_code=400,
            detail="系统尚未初始化"
        )
    
    qa_system.clear_history()
    return {"status": "success", "message": "对话历史已清空"}


@app.get("/api/history")
async def get_history():
    global qa_system
    
    if qa_system is None:
        raise HTTPException(
            status_code=400,
            detail="系统尚未初始化"
        )
    
    return {
        "status": "success",
        "history": qa_system.get_history()
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": qa_system is not None
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info"
    )

