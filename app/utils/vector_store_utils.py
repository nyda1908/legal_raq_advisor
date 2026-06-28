from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever


class MyVectorStore:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.documents = None

    def embed_text(self, documents):
        self.documents = documents
        Chroma.from_documents(
            documents, self.embeddings, persist_directory="../data/chroma_db"
        )

    def get_retriever(self):
        # dense retriever
        vector_store = Chroma(
            persist_directory="../data/chroma_db", embedding_function=self.embeddings
        )
        dense_retriever = vector_store.as_retriever(search_kwargs={"k": 20})

        # sparse retriever
        bm25_retriever = BM25Retriever.from_documents(self.documents)
        bm25_retriever.k = 20

        # combine
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[0.5, 0.5]
        )
        return ensemble_retriever
