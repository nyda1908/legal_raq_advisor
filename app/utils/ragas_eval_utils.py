from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class RAGASEvaluator:
    def __init__(self):
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

    def evaluate(self, questions, answers, contexts, ground_truths):
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
            embeddings=OpenAIEmbeddings(),
        )
        return result

    def format_results(self, result):
        df = result.to_pandas()
        summary = {
            "faithfulness": round(df["faithfulness"].mean(), 3),
            "answer_relevancy": round(df["answer_relevancy"].mean(), 3),
            "context_precision": round(df["context_precision"].mean(), 3),
            "context_recall": round(df["context_recall"].mean(), 3),
        }
        return summary, df
