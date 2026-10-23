import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from cohere import Client as CohereClient

load_dotenv()

class HybridRetriever:
    def __init__(self):
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        self.collection_name = "enterprise_docs"
        
        self.qdrant = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", 
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.cohere = CohereClient(api_key=os.getenv("COHERE_API_KEY"))

    def retrieve_context(self, query: str, top_k: int = 10, top_n: int = 3) -> list[str]:
        # 1. Embed user query
        query_vector = self.embeddings.embed_query(query)

        # 2. Vector Search in Qdrant (Updated method call)
        response = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        )

        candidates = [hit.payload["text"] for hit in response.points if hit.payload and "text" in hit.payload]
        if not candidates:
            return ["No relevant context found in database."]

        # 3. Cohere Rerank v3 Filtering
        rerank_response = self.cohere.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=candidates,
            top_n=top_n
        )

        return [candidates[r.index] for r in rerank_response.results]