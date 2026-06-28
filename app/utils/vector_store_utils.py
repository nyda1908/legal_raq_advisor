from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document


class MyVectorStore:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()

    def embed_text(self, documents):
        Chroma.from_documents(
            documents, self.embeddings, persist_directory="../data/chroma_db"
        )

    def get_retriever(self):
        vector_store = Chroma(
            persist_directory="../data/chroma_db", embedding_function=self.embeddings
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 20})
        return retriever
