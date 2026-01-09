from typing import List
import logging
from langchain_core.messages import ToolMessage, BaseMessage

def serialize_messages(messages: List[BaseMessage]) -> List[dict]:
    """Serialize LangChain messages to a compact JSON-compatible format."""

    logger = logging.getLogger(__name__)
    serialized = []

    for msg in messages:
        entry = {
            "type": msg.__class__.__name__,
        }

        if isinstance(msg, ToolMessage):
            entry["tool_name"] = msg.name

            try:
                content = msg.content
                if isinstance(content, str):
                    content = json.loads(content)

                if isinstance(content, list):
                    entry["content_summary"] = f"Returned {len(content)} results"
                    # Optional preview
                    if content and isinstance(content[0], dict):
                        entry["preview_keys"] = list(content[0].keys())
                elif isinstance(content, dict):
                    entry["content_summary"] = "Returned structured data"
                    entry["keys"] = list(content.keys())
                else:
                    entry["content_summary"] = "Tool executed successfully"

            except Exception as e:
                logger.exception("Failed to summarize tool message")
                entry["content_summary"] = "Tool result (unavailable)"

        else:
            entry["content"] = msg.content

        if getattr(msg, "tool_calls", None):
            entry["tool_calls"] = [
                {
                    "name": tc.get("name"),
                    "args": tc.get("args"),
                    "id": tc.get("id"),
                }
                for tc in msg.tool_calls
            ]

        serialized.append(entry)

    return serialized
