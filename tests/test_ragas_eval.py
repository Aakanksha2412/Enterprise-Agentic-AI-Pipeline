import pytest
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision
from app.agent_graph import ai_brain

def test_pipeline_ragas_benchmarks():
    # Test dataset
    test_questions = [
        "What is my most recent work experience?",
        "What software technology stack do I specialize in?"
    ]
    
    answers = []
    contexts = []

    # Run query through LangGraph Agent
    for q in test_questions:
        res = ai_brain.invoke({"question": q})
        answers.append(res.get("final_answer", ""))
        contexts.append(res.get("retrieved_docs", ["No context retrieved"]))

    # Construct HuggingFace Dataset for Ragas
    data = {
        "question": test_questions,
        "answer": answers,
        "contexts": contexts
    }
    dataset = Dataset.from_dict(data)

    # Compute Ragas Metrics
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevance, context_precision]
    )

    score_faithfulness = results["faithfulness"]
    score_relevance = results["answer_relevance"]

    print(f"\n--- RAGAS EVALUATION METRICS ---")
    print(f"Faithfulness Score: {score_faithfulness:.2f}")
    print(f"Answer Relevance: {score_relevance:.2f}")

    # Assertion Quality Gate: Enforce threshold > 0.70
    assert score_faithfulness >= 0.70, f"Faithfulness score {score_faithfulness} below threshold!"
    assert score_relevance >= 0.70, f"Relevance score {score_relevance} below threshold!"