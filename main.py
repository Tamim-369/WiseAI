from fastapi import FastAPI, File, UploadFile
from controllers.api_controller import APIController
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
app = FastAPI(title="RAG Application with Gemini")
controller = APIController()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
class QueryRequest(BaseModel):
    question: str
    chat_history: Optional[List[dict]] = None
@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    return await controller.upload_document(file)

@app.post("/query/")  
async def query_documents(request: QueryRequest):
    question = request.question
    chat_history = request.chat_history or []  
    return await controller.query_documents(question, chat_history)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)