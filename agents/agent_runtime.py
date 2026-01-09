from langgraph.prebuilt import create_react_agent
from typing import Optional, TypedDict, List
from pathlib import Path
import logging
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
    BaseMessage,
)

from llm.llm import load_llm
from tools.text_search import search_by_text_tool
from tools.image_search import search_by_image_tool

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load LLM once ---
logger.info("Loading LLM...")
llm = load_llm()

SYSTEM_PROMPT = """
You are a photo search assistant for a personal photo collection.

The user may write queries in any language.
If the input is not in English, translate the user’s intent into clear,
natural English before generating search queries or embeddings.

If the user input is short, fragmented, or keyword-based, rewrite it into
a coherent, descriptive English sentence suitable for image search.

CRITICAL RULES (STRICTLY ENFORCED):
1. NEVER invent or describe photos that were not returned by tools
2. ONLY describe photos explicitly present in tool results
3. NEVER hallucinate image details
4. If tools return no results, respond exactly:
   "No matching photos were found."

Query Enhancement Rules:
- Combine fragmented keywords into a natural visual description
- Preserve ALL user-mentioned visual elements

Examples:

User: "high altitude mountains lake forest blue sky clouds"
Interpret as:
"A high-altitude landscape with mountains surrounding a lake, forest areas,
and a blue sky with clouds."

User: "montañas lago cielo azul nubes dron"
Interpret as:
"A high-altitude drone shot of mountains surrounding a lake, under a blue sky
with large clouds."

Decomposition Rule:
- If the user requests multiple photos with mutually exclusive conditions
  (e.g. sunrise and sunset, winter and summer, January and February),
  treat them as separate searches.
- In such cases, call the search tool multiple times,
  once per distinct condition, and then combine the results.

IMPORTANT - Photo Display Rules:
- When search tools return results, they include 'photo_id' and metadata
- DO NOT construct URLs or markdown image links
- Simply describe the photos with their metadata (date, time, camera)
- List results numerically with relevant details
- When useful, you may mention similarity score

Example Response Format:
"I found 2 sunrise photos:

1. Photo from October 13, 2025 at 8:24 AM (morning), taken with HUAWEI VOG-L29
   Similarity score: 0.087
   Photo ID: bba0c8f9-646e-4e63-acc8-3ba783b6b05e

2. Photo from October 3, 2025 at 7:47 AM (morning), taken with HUAWEI VOG-L29
   Similarity score: 0.358
   Photo ID: 8e95dbd9-b9bd-425e-a275-a86c9942d9ff"

Do NOT create image markdown or URLs - the application will handle photo display.

Workflow:
1. Normalise and enrich the user description if fragmented
2. Translate to English if needed
3. Call the appropriate search tool
4. Accurately report tool results only
5. If no results exist, state this clearly
"""

# --- Load tools once ---
tools = [
    search_by_text_tool,
    search_by_image_tool,
]

# --- Create agent ---
logger.info("Creating agent...")
agent = create_react_agent(model=llm, tools=tools)
logger.info("Agent ready")


class AgentResult(TypedDict):
    response: str
    messages: List[dict]
    tool_calls: int   


def count_tool_calls(messages: List[BaseMessage]) -> int:
    """Count total tool calls across all messages."""
    count = 0
    for msg in messages:
        if isinstance(msg, BaseMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            count += len(msg.tool_calls)
    return count


def run_agent_text(user_input: str) -> AgentResult:
    """
    Process a text search query through the agent.

    Args:
        user_input: User's search query text

    Returns:
        AgentResult with response, messages, and tool call count
    """
    # Validate input
    if not user_input or not user_input.strip():
        logger.warning("Empty text query received")
        return {
            "response": "Please provide a search query.",
            "messages": [],
            "tool_calls": 0
        }

    logger.info(f"Processing text query: '{user_input[:100]}...'")

    try:
        result = agent.invoke(
            {
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_input)
                ]
            },
            config={"recursion_limit": 5}
        )

        if not result or "messages" not in result or not result["messages"]:
            logger.error("Invalid or empty agent result")
            return {
                "response": "Search failed. Please try again.",
                "messages": [],
                "tool_calls": 0
            }

        tool_calls = count_tool_calls(result["messages"])
        final_response = result["messages"][-1].content

        logger.info(f"Text search completed: {tool_calls} tool calls")

        return {
            "response": final_response,
            "tool_calls": tool_calls,
        }

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        return {
            "response": "An error occurred during search. Please try again.",
            "messages": [],
            "tool_calls": 0
        }


def run_agent_image(image_path: str, query: Optional[str] = None) -> AgentResult:
    """
    Process an image search query through the agent.

    Args:
        image_path: Path to the uploaded image file
        query: Optional text to accompany the image

    Returns:
        AgentResult with response, messages, and tool call count
    """
    # Validate image path
    if not image_path or not image_path.strip():
        logger.error("Empty image path provided")
        return {
            "response": "No image provided.",
            "messages": [],
            "tool_calls": 0
        }

    if not Path(image_path).exists():
        logger.error(f"Image file not found: {image_path}")
        return {
            "response": "Image file not found.",
            "messages": [],
            "tool_calls": 0
        }

    logger.info(f"Processing image query: path='{image_path}', query='{query}'")

    try:
        # Build user message
        if query:
            user_message = f"{query} Image path: {image_path}"
        else:
            user_message = f"Find photos similar to this image: {image_path}"

        result = agent.invoke(
            {
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_message)
                ]
            },
            config={"recursion_limit": 10}
        )

        if not result or "messages" not in result or not result["messages"]:
            logger.error("Invalid or empty agent result")
            return {
                "response": "Search failed. Please try again.",
                "messages": [],
                "tool_calls": 0
            }

        tool_calls = count_tool_calls(result["messages"])
        final_response = result["messages"][-1].content

        logger.info(f"Image search completed: {tool_calls} tool calls")

        return {
            "response": final_response,
            "tool_calls": tool_calls,
            "messages": serialize_messages(result["messages"]),
        }

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        return {
            "response": "An error occurred during search. Please try again.",
            "messages": [],
            "tool_calls": 0
        }




