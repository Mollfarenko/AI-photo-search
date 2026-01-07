def unified_search(collection, query_embedding,
    where: dict | None = None,
    k: int = 5
):
    return collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        where=where
    )
