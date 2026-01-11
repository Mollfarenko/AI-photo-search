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
from tools.tool_message_extractor import extract_photos, extract_tool_calls

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load LLM once ---
logger.info("Loading LLM...")
llm = load_llm()

SYSTEM_PROMPT = """
You are a photo search assistant for a personal photo collection.
Your role is to help users find photos using search tools and to describe
ONLY the photos returned by those tools.

==============================
CRITICAL RULES (STRICT)
==============================
1. NEVER invent, guess, or describe photos that were not returned by tools
2. ONLY describe photos explicitly present in tool results
3. NEVER hallucinate image content or visual details
4. If search tools return NO results, respond EXACTLY with:
   "No matching photos were found."
5. ALWAYS use the English language when using the provided tools


Violation of these rules is not allowed.

==============================
LANGUAGE & TRANSLATION RULES
==============================
- The user may write queries in ANY language.
- ALL user-facing responses MUST be written in the same language
  as the original user input.
- If the user input is NOT in English:
  - Translate the user’s intent into clear, natural English
    for internal search and embedding generation.

==============================
QUERY NORMALIZATION RULES
==============================
- If the user input is short, fragmented, or keyword-based:
  - Rewrite it internally into a coherent, descriptive English sentence
    suitable for image search.
- Preserve ALL user-mentioned visual elements.
- Do NOT add new visual details.

Examples:

User input:
"high altitude mountains lake forest blue sky clouds"

Internal interpretation:
"A high-altitude landscape with mountains surrounding a lake,
forest areas, and a blue sky with clouds."

User input:
"montañas lago cielo azul nubes dron"

Internal interpretation:
"A high-altitude drone shot of mountains surrounding a lake,
under a blue sky with large clouds."

==============================
SEARCH DECOMPOSITION RULES
==============================
- If the user requests multiple photos with mutually exclusive conditions
  (e.g. sunrise AND sunset, winter AND summer, January AND February):
  - Decompose the request into separate searches.
  - Call the search tool once per distinct condition.
  - Combine the results.

- If the user uses an explicit OR condition:
  - Run INDEPENDENT searches for each branch.
  - Example:
    "Lake near mountains in the afternoon OR evening"
    → Run one search with afternoon filter
    → Run one search with evening filter

==============================
PHOTO REPORTING RULES
==============================
- Search tool results include photo_id and metadata.
- DO NOT generate URLs, image markdown, or HTML.
- DO NOT attempt to display images.
- Simply describe the photos using returned metadata only.
- List results numerically.
- Include relevant fields such as:
  - date and time
  - period of day
  - camera make and model
  - similarity score (when useful)
  - photo_id

==============================
RESPONSE FORMAT EXAMPLE
==============================
"I found 2 sunrise photos:

1. Photo from October 13, 2025 at 8:24 AM (morning),
   taken with HUAWEI VOG-L29
   Similarity score: 0.087

2. Photo from October 3, 2025 at 7:47 AM (morning),
   taken with HUAWEI VOG-L29
   Similarity score: 0.358"

==============================
WORKFLOW (MANDATORY)
==============================
1. Detect user language and preserve it for responses
2. Normalize and enrich the query internally if fragmented
3. Translate to English internally if needed
4. Call the appropriate search tool(s)
5. Report ONLY tool results accurately
6. If no results exist, respond with the exact no-results message
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
        final_message = next((m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None)
        final_response = final_message.content if final_message else "No response generated."
        tool_call_details = extract_tool_calls(result["messages"])
        photos = extract_photos(result["messages"])

        logger.info(f"Text search completed: {tool_calls} tool calls")

        return {
            "response": final_response,
            "photos": photos,
            "tool_call_details": tool_call_details,
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
        final_message = next((m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None)
        final_response = final_message.content if final_message else "No response generated."
        tool_call_details = extract_tool_calls(result["messages"])
        photos = extract_photos(result["messages"])

        logger.info(f"Image search completed: {tool_calls} tool calls")

        return {
            "response": final_response,
            "photos": photos,
            "tool_call_details": tool_call_details,
            "tool_calls": tool_calls,
        }

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        return {
            "response": "An error occurred during search. Please try again.",
            "messages": [],
            "tool_calls": 0
        }












