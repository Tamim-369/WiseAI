from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from controllers.api_controller import APIController
from fastapi import File, UploadFile
import traceback

app = FastAPI(title="RAG Application with Gemini")
controller = APIController()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://wise-ai-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    chat_history: Optional[List[dict]] = None

def strip_markdown(text: str) -> str:
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
async def upload_document(file:UploadFile = File(...)):
    return await controller.upload_document(file)

@app.post("/query/")
async def query_documents(request: QueryRequest):
    question = request.question
    chat_history = request.chat_history or []
    try:
        data = await controller.query_documents(question, chat_history)
        response_text = data.get("answer", "") if isinstance(data, dict) else str(data)
        clean_text = strip_markdown(response_text)
        return JSONResponse(content={"answer": clean_text})
    except Exception as e:
        print("Query error:", str(e))
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)