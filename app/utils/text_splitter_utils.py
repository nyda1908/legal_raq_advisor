import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class MyTextSplitter:
    def __init__(self, documents):
        self.documents = documents

    def get_text_chunks(self):
        encoder = tiktoken.encoding_for_model("text-embedding-ada-002")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=lambda text: len(encoder.encode(text)),
        )
        
        chunks = []
        for doc in self.documents:
            split_texts = text_splitter.split_text(doc.page_content)
            for chunk in split_texts:
                chunks.append(
                    Document(
                        page_content=chunk,
                        metadata=doc.metadata
                    )
                )
        return chunks
