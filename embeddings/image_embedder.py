# src/image_embedder.py

import torch
from PIL import Image

def embed_image(image_path: str, model, processor, device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        embedding = model.get_image_features(**inputs)

    return embedding / embedding.norm(p=2, dim=-1, keepdim=True)
