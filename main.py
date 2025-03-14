from fastapi import FastAPI, File, UploadFile
from controllers.api_controller import APIController
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RAG Application with Gemini")
controller = APIController()
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    return await controller.upload_document(file)

@app.get("/query/")
async def query_documents(question: str):
    return await controller.query_documents(question)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)