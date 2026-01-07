import chromadb
from chromadb.config import Settings

def get_chroma_client(persist_dir="data/chroma"):
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(
            anonymized_telemetry=False
        )
    )

def get_collection(client, name="photos"):
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )
