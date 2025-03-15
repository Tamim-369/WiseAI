from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import io
from gtts import gTTS  
from controllers.api_controller import APIController

app = FastAPI(title="RAG Application with Gemini")
controller = APIController()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://wise-ai-frontend.vercel.app", "http://localhost:5173"],  
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class QueryRequest(BaseModel):
    question: str
    chat_history: Optional[List[dict]] = None

def strip_markdown(text: str) -> str:
    """Remove Markdown syntax and return plain text."""
    return (
        text
        .replace("#", "")
        .replace("*", "")
        .replace("_", "")
        .replace("`", "")
        .replace("[", "").replace("]", "")
        .replace("(", "").replace(")", "")
        .strip()
        .replace("\n", " ")
        .replace("  ", " ")
    )

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document to the controller."""
    return await controller.upload_document(file)

@app.post("/query/")
async def query_documents(request: QueryRequest):
    """Query documents and stream audio response."""
    question = request.question
    chat_history = request.chat_history or []
    
    data = await controller.query_documents(question, chat_history)
    response_text = data.get("answer", "") if isinstance(data, dict) else str(data)
    clean_text = strip_markdown(response_text)
    
    return await controller.generate_speech_stream(clean_text)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)