from transformers import CLIPProcessor, CLIPModel
import torch

def load_clip_model():
    """
    Load CLIP model and processor.
    Returns:
        model: CLIPModel
        processor: CLIPProcessor
        device: torch.device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    model.eval()  # important: inference mode

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    return model, processor, device
