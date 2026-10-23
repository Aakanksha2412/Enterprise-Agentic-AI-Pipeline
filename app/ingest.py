import os
import glob
from dotenv import load_dotenv

# Document Loading & Chunking
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings & Vector Store
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Knowledge Graph (Neo4j)
from neo4j import GraphDatabase

# Load environment variables from .env
load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

COLLECTION_NAME = "enterprise_docs"


class IngestionPipeline:
    def __init__(self):
        # Initialize Embeddings Engine
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
        
        # Initialize Qdrant Client
        self.qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Initialize Neo4j Driver
        self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        # Setup Qdrant Collection if not existing
        self._setup_qdrant_collection()

    def _setup_qdrant_collection(self):
        collections = self.qdrant_client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        if not exists:
            self.qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
            print(f"Created Qdrant collection: {COLLECTION_NAME}")
        else:
            print(f"Qdrant collection '{COLLECTION_NAME}' already exists.")

    def load_and_chunk_documents(self, data_folder: str = "./data"):
        print("1. Loading and chunking documents...")
        raw_docs = []
        
        # Load text and markdown files
        for filepath in glob.glob(f"{data_folder}/*.txt") + glob.glob(f"{data_folder}/*.md"):
            try:
                loader = TextLoader(filepath)
                raw_docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
            
        # Load PDF files
        for filepath in glob.glob(f"{data_folder}/*.pdf"):
            try:
                loader = PyPDFLoader(filepath)
                raw_docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

        if not raw_docs:
            print(f"No documents found in {data_folder}/. Place your files inside {data_folder}/ and try again.")
            return []

        # Context-aware recursive chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(raw_docs)
        print(f"Split {len(raw_docs)} source files into {len(chunks)} chunk windows.")
        return chunks

    def ingest_to_qdrant(self, chunks):
        print("2. Generating embeddings and pushing chunks to Qdrant...")
        points = []
        for idx, chunk in enumerate(chunks):
            vector = self.embeddings.embed_query(chunk.page_content)
            points.append(
                PointStruct(
                    id=idx + 1,
                    vector=vector,
                    payload={
                        "text": chunk.page_content,
                        "source": chunk.metadata.get("source", "unknown"),
                        "page": chunk.metadata.get("page", 0)
                    }
                )
            )
        
        self.qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Successfully uploaded {len(points)} vector embeddings to Qdrant.")

    def ingest_to_neo4j(self, chunks):
        print("3. Building document entity nodes in Neo4j Knowledge Graph...")
        with self.neo4j_driver.session() as session:
            for idx, chunk in enumerate(chunks):
                source = chunk.metadata.get("source", "unknown")
                text = chunk.page_content[:100]  # Chunk summary
                
                query = """
                MERGE (d:Document {name: $source})
                CREATE (c:Chunk {id: $chunk_id, text: $text})
                CREATE (d)-[:HAS_CHUNK]->(c)
                """
                session.run(query, source=source, chunk_id=idx + 1, text=text)
        print("Successfully created document-chunk entity topology in Neo4j.")

    def close(self):
        self.neo4j_driver.close()


if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    
    pipeline = IngestionPipeline()
    chunks = pipeline.load_and_chunk_documents(data_folder="./data")
    
    if chunks:
        pipeline.ingest_to_qdrant(chunks)
        pipeline.ingest_to_neo4j(chunks)
        print("\nIngestion Complete! Databases populated successfully.")
    
    pipeline.close()