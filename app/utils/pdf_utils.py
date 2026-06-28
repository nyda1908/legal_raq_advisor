from PyPDF2 import PdfReader
from langchain.schema import Document


class MyPDF:
    def __init__(self, pdf):
        self.pdf = pdf

    def get_pdf_text(self):
        documents = []
        for pdf in self.pdf:
            pdf_reader = PdfReader(pdf)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": pdf.name,
                                "page": page_num + 1
                            }
                        )
                    )
        return documents
