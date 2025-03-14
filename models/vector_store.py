from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv() 

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))

class VectorStoreModel:
    def __init__(self):
        self.persist_dir = "./chroma_db"
        self.vector_store = None

    def load_and_index_documents(self, file_path):
        """Load a document, split it, and index it into the vector store."""
        # Load the document
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Create or update vector store
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.persist_dir
        )
        print(f"Indexed {len(chunks)} chunks from {file_path}")

    def get_retriever(self):
        """Get a retriever for querying the vector store."""
        if not self.vector_store:
            self.vector_store = Chroma(persist_directory=self.persist_dir, embedding_function=embeddings)
        return self.vector_store.as_retriever(search_kwargs={"k": 3})  