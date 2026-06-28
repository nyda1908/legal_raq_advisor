from dotenv import load_dotenv
from utils.pdf_utils import MyPDF
from utils.text_splitter_utils import MyTextSplitter
from utils.vector_store_utils import MyVectorStore
from utils.langchain_utils import MyLangChain
from utils.ragas_eval_utils import RAGASEvaluator

load_dotenv()

# --- test cases ---
# ground truths are the ideal answers you write manually
TEST_CASES = [
    {
        "question": "What is the notice period for termination?",
        "ground_truth": "The contract may be terminated with 30 days written notice."
    },
    {
        "question": "Who is responsible for indemnification?",
        "ground_truth": "The vendor is responsible for indemnifying the client against third party claims."
    },
    {
        "question": "What are the payment terms?",
        "ground_truth": "Payment is due within 30 days of invoice receipt."
    },
]


def run_evaluation(pdf_paths):
    # build pipeline
    pdf = MyPDF(pdf=pdf_paths)
    documents = pdf.get_pdf_text()

    splitter = MyTextSplitter(documents)
    chunks = splitter.get_text_chunks()

    vector_store = MyVectorStore()
    vector_store.embed_text(chunks)
    retriever = vector_store.get_retriever()

    lang_chain = MyLangChain()
    chain = lang_chain.generate_answer_chain(retriever)

    # collect questions, answers, contexts, ground truths
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for case in TEST_CASES:
        question = case["question"]
        result = chain.invoke({"user_prompt": question})

        questions.append(question)
        answers.append(result["response"].content)
        contexts.append([doc for doc in result["context"]])
        ground_truths.append(case["ground_truth"])

    # evaluate
    evaluator = RAGASEvaluator()
    result = evaluator.evaluate(questions, answers, contexts, ground_truths)
    summary, df = evaluator.format_results(result)

    print("\n=== RAGAS Evaluation Results ===")
    for metric, score in summary.items():
        print(f"{metric}: {score}")

    df.to_csv("ragas_results.csv", index=False)
    print("\nDetailed results saved to ragas_results.csv")


if __name__ == "__main__":
    run_evaluation(pdf_paths=["path/to/your/contract.pdf"])
