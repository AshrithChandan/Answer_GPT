from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from sentence_transformers import SentenceTransformer
import uuid

# Load sentence transformer model for embeddings
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Connect to Qdrant (in-memory for now; change to host/port for persistence)
client = QdrantClient(":memory:")  # For real use: host="localhost", port=6333

# Initialize collection if it doesn't exist
def init_collection():
    existing = [col.name for col in client.get_collections().collections]
    if "answergpt" not in existing:
        client.recreate_collection(
            collection_name="answergpt",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

# Add a document to the Qdrant vector store
def add_text(text):
    init_collection()
    vector = embedder.encode(text).tolist()
    point = PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": text})
    client.upsert(collection_name="answergpt", points=[point])

# Query similar documents by semantic similarity
def query_text(query, top_k=3):
    init_collection()
    query_vector = embedder.encode(query).tolist()
    results = client.search(
        collection_name="answergpt",
        query_vector=query_vector,
        limit=top_k
    )
    return [res.payload["text"] for res in results]
