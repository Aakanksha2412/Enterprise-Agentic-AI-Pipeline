import os
import sys
import types
import numpy as np

# 1. Resolve root package path for 'app' imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 2. Inject dummy module to bypass legacy vertexai import check in ragas
dummy_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_vertex.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_vertex

import pytest
from datasets import Dataset
from ragas import evaluate

# 3. Dynamic metric import fallback
try:
    from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision
except ModuleNotFoundError:
    from ragas.metrics import faithfulness, answer_relevancy, context_precision

from app.agent_graph import ai_brain


def test_pipeline_ragas_benchmarks():
    test_questions = [
        "What is my most recent work experience?",
        "What software technology stack do I specialize in?"
    ]

    # Ground truth answers required by context_precision
    ground_truths = [
        "Software Development Engineer at Amazon starting September 2025.",
        "Full-stack development using JavaScript, Java, Python, C++, React, Angular, Node.js, ASP.NET Core, MongoDB, PostgreSQL, Qdrant, Neo4j, Redis, AWS, and Docker."
    ]
    
    answers = []
    contexts = []

    # Run query through LangGraph Agent
    for q in test_questions:
        res = ai_brain.invoke({"question": q})
        answers.append(res.get("final_answer", ""))
        contexts.append(res.get("retrieved_docs", ["No context retrieved"]))

    # Standard dataset column schema for modern Ragas evaluation
    data = {
        "user_input": test_questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths
    }
    dataset = Dataset.from_dict(data)

    # Compute Ragas Metrics
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision]
    )

    # Helper function to extract scalar mean from float, list, or numpy array
    def get_mean_score(val):
        if isinstance(val, (list, tuple, np.ndarray)):
            return float(np.nanmean(val))
        return float(val)

    score_faithfulness = get_mean_score(results["faithfulness"])
    score_relevance = get_mean_score(results["answer_relevancy"])
    score_precision = get_mean_score(results["context_precision"])

    print(f"\n--- RAGAS EVALUATION METRICS ---")
    print(f"Faithfulness Score: {score_faithfulness:.2f}")
    print(f"Answer Relevance:   {score_relevance:.2f}")
    print(f"Context Precision:  {score_precision:.2f}")

    # Assertion Quality Gates
    assert score_faithfulness >= 0.70, f"Faithfulness score {score_faithfulness} below threshold!"
    assert score_relevance >= 0.70, f"Relevance score {score_relevance} below threshold!"