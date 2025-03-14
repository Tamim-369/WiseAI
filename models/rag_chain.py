from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
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
        You are the wisest AI with tons of knowledge from the given context. You answer every question by thinking about the context and if there is anything which is not in the context you give an answer by yourself which doesn't go against the context. Your name is WiseAI. You have advice about relationships, dating, power, war, money—everything—and you don't say hello or anything, just give advice and knowledge, pure wisdom. Go ahead and answer every question we give to you.

        Context: {context}
        History: {chat_history}
        Question: {question}

        Answer:
        """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question", "chat_history"]
        )

        # Define the chain to process question and chat_history
        chain = (
            RunnablePassthrough.assign(
                context=lambda x: self.retriever.invoke(x["question"]),  
                question=lambda x: x["question"],                      
                chat_history=lambda x: x["chat_history"]                
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def query(self, question, chat_history):
        """Run a query through the RAG chain and return the answer."""
        if isinstance(chat_history, list):
            chat_history_str = "\n".join(
                f"{msg['role']}: {msg['content']}" for msg in chat_history
            )
        else:
            chat_history_str = chat_history or ""

        return self.chain.invoke({
            "question": question,
            "chat_history": chat_history_str
        })

