from langchain_core.messages import ToolMessage
import json

def extract_photos(messages: List[BaseMessage]) -> list[dict]:
    """Extract normalized photo results from tool messages."""
    photos = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            try:
                content = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                if isinstance(content, list):
                    photos.extend(content)
            except:
                pass
    return photos

