# src/text_embedder.py

import torch

def embed_text(text: str, model, processor, device) -> torch.Tensor:
    inputs = processor(text=text, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        embedding = model.get_text_features(**inputs)

    return embedding / embedding.norm(p=2, dim=-1, keepdim=True)
