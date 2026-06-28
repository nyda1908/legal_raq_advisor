from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
)
from utils.clause_classifier_utils import ClauseClassifier


class MyLangChain:
    def __init__(self):
        self.classifier = ClauseClassifier()

    def generate_answer_chain(self, base_retriever):
        template = """Act as a legal contract answering expert. You will be presented with legal contract clauses as context and a question related to that contract. Each clause has been labelled with its type. Your task is to provide a succinct answer to the question based on the content of the contract. Make sure you reply with "I don't know" if the answer cannot be found in the context.
        ### CONTEXT
        {context}

        ### Question
        Question: {user_prompt}
        """

        prompt = ChatPromptTemplate.from_template(template)
        primary_qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        def retrieve_and_classify(inputs):
            question = inputs["user_prompt"]
            docs = base_retriever.invoke(question)
            classified_docs = self.classifier.classify(docs)
            context = "\n\n".join(
                f"[{doc.metadata.get('clause_type', 'other').upper()} CLAUSE "
                f"- {doc.metadata.get('source', 'unknown')} p.{doc.metadata.get('page', '?')}]\n"
                f"{doc.page_content}"
                for doc in classified_docs
            )
            return {"context": context, "user_prompt": question}

        retrieval_augmented_qa_chain = (
            RunnableLambda(retrieve_and_classify)
            | {
                "response": prompt | primary_qa_llm,
                "context": itemgetter("context"),
            }
        )
        return retrieval_augmented_qa_chain
