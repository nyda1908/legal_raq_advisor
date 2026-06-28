from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker


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

        # hybrid retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[0.5, 0.5]
        )

        # reranker
        cross_encoder = HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=7)
        reranking_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
        )

        return reranking_retriever
