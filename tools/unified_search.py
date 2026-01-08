def unified_search(
    collection,
    query_embedding,
    where: dict | None = None,
    k: int = 5
) -> list[dict]:
    """
    Search ChromaDB collection and return normalised photo results.
    
    Args:
        collection: ChromaDB collection instance
        query_embedding: Query embedding vector (already in list format)
        where: Optional metadata filter dictionary
        k: Number of results to return
        
    Returns:
        List of photo dictionaries with cleaned metadata
    """
    # Query ChromaDB
    raw_results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        where=where
    )
    
    # Extract nested lists (ChromaDB returns [[...]] format)
    ids = raw_results.get("ids", [[]])[0]
    metadatas = raw_results.get("metadatas", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]
    
    # Normalise into clean photo dictionaries
    photos = []
    for i in range(len(ids)):
        meta = metadatas[i] if i < len(metadatas) else {}
        distance = distances[i] if i < len(distances) else None
        
        photos.append({
            "photo_id": meta.get("id"),
            "photo_key": meta.get("photo_key"),
            "thumbnail_key": meta.get("thumbnail_key"),
            "bucket": meta.get("bucket"),
            "taken_at": meta.get("taken_at"),
            "period_of_day": meta.get("period_of_day"),
            "year": meta.get("year"),
            "month": meta.get("month"),
            "month_name": meta.get("month_name"),
            "hour": meta.get("hour"),
            "camera_make": meta.get("camera_make"),
            "camera_model": meta.get("camera_model"),
            "distance": distance,
            "similarity_score": round(1 - distance, 3) if distance is not None else None
        })
    
    return photos
