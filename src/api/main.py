from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Optional, List
from ..agents.agent_manager import AgentManager
import asyncio
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

class Question(BaseModel):
    text: str
    doc_id: Optional[str] = None

class Response(BaseModel):
    answer: str
    confidence: float
    sources: List[str]

app = FastAPI()
agent_manager = AgentManager(model_name="mistralai/Mistral-7B-v0.1")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        text_content = contents.decode("utf-8")
        
        result = await agent_manager.process_document(text_content)
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["error"])
            
        return {"message": "Document processed successfully", "details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=Response)
async def ask_question(question: Question):
    try:
        result = await agent_manager.process_query(question.text, question.doc_id)
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["error"])
            
        return Response(
            answer=result["response"],
            confidence=result["validation"]["confidence"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))