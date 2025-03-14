from models.vector_store import VectorStoreModel
from models.rag_chain import RAGChainModel
from views.api_views import APIView
from typing import AsyncGenerator
import edge_tts
import os

class APIController:
    def __init__(self):
        self.vector_store = VectorStoreModel()
        self.rag_chain = RAGChainModel(self.vector_store.get_retriever())
        self.view = APIView()

    async def upload_document(self, file):
        """Handle document upload and indexing."""
        try:
            # Save file temporarily
            file_path = f"temp_{file.filename}"
            with open(file_path, "wb") as f:
                f.write(await file.read())

            # Index the document
            self.vector_store.load_and_index_documents(file_path)
            response = self.view.success_response(
                data=None,
                message=f"Document {file.filename} indexed successfully"
            )
        except Exception as e:
            response = self.view.error_response(str(e))
        finally:
            if os.path.exists(file_path):
                # Cleanup temporary file
                os.remove(file_path)  

        return response

    async def query_documents(self, question, chat_history):
        """Handle query and return RAG response."""
        try:
            answer = self.rag_chain.query(question, chat_history)
            response = answer
        except Exception as e:
            response = self.view.error_response(str(e))
        return response
    async def generate_speech_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Generate speech using edge_tts and stream it."""
        try:
            tts = edge_tts.Communicate(text, voice="en-US-AndrewMultilingualNeural", rate="-10%", volume="+10%")
            async for chunk in tts.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
                elif chunk["type"] == "WordBoundary":
                    continue
        except Exception as e:
            print(f"An error occurred during speech generation: {e}")
            raise