from typing import TypedDict, Sequence, Annotated
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from app.retriever import HybridRetriever

retriever = HybridRetriever()

class AgentState(TypedDict):
    question: str
    next_step: str
    retrieved_docs: list[str]
    final_answer: str

def router_node(state: AgentState):
    question = state["question"].lower()
    if any(k in question for k in ["sales", "revenue", "metrics"]):
        return {"next_step": "sql_agent"}
    return {"next_step": "rag_agent"}

def rag_retrieval_node(state: AgentState):
    docs = retriever.retrieve_context(state["question"])
    return {"retrieved_docs": docs, "next_step": "synthesize"}

def synthesis_node(state: AgentState):
    docs = state.get("retrieved_docs", [])
    context_str = "\n".join([f"- {d}" for d in docs])
    answer = f"Retrieved Context:\n{context_str}"
    return {"final_answer": answer, "next_step": END}

# Build State Graph
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("rag_agent", rag_retrieval_node)
workflow.add_node("synthesize", synthesis_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", lambda x: x["next_step"], {
    "rag_agent": "rag_agent",
    "sql_agent": "synthesize"
})
workflow.add_edge("rag_agent", "synthesize")

ai_brain = workflow.compile()