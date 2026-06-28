from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document

CLAUSE_TYPES = [
    "termination",
    "indemnity",
    "payment",
    "confidentiality",
    "liability",
    "intellectual property",
    "dispute resolution",
    "governing law",
    "other"
]

class ClauseClassifier:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        self.prompt = ChatPromptTemplate.from_template(
            """You are a legal contract expert. Classify the following contract clause into exactly one of these categories:
{clause_types}

Respond with only the category name, nothing else.

Clause:
{clause}
"""
        )
        self.chain = self.prompt | self.llm

    def classify(self, documents):
        classified = []
        for doc in documents:
            response = self.chain.invoke({
                "clause_types": "\n".join(f"- {c}" for c in CLAUSE_TYPES),
                "clause": doc.page_content
            })
            clause_type = response.content.strip().lower()
            if clause_type not in CLAUSE_TYPES:
                clause_type = "other"

            classified.append(
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "clause_type": clause_type}
                )
            )
        return classified
