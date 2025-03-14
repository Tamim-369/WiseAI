from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

class RAGChainModel:
    def __init__(self, retriever):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7
        )
        self.retriever = retriever
        self.chain = self._build_chain()

    def _build_chain(self):
        prompt_template = """
        You are the wisest AI with tons of knowledge from the given context. You answer every question by thinking about the context and if there is anything which is not in the context you give an answer by yourself which doesn't goes against the context. Your name is WiseAI. You have advise about relationship, dating, power, war, money everything and you don't say hello or anything just give advise and knowledge just pure wisdom. Go ahead answer every question we give to you.

        Context: {context}

        Question: {question}

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff", 
            retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt}
        )

    def query(self, question):
        """Run a query through the RAG chain and return the answer."""
        return self.chain.invoke({"query": question})["result"]