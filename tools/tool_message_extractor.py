from langchain_core.messages import ToolMessage, BaseMessage, AIMessage
import json

def extract_photos(messages: list[BaseMessage]) -> list[dict]:
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

def extract_tool_calls(messages: list[BaseMessage]) -> list[dict]:
    """Extract tool calls made by the LLM."""
    tool_calls = []

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_calls.extend(msg.tool_calls)

    return tool_calls





