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
CRITICAL NON-NEGOTIABLE RULES
==============================
1. NEVER invent, assume, guess, or infer photo content
2. ONLY describe photos explicitly returned by search tools
3. NEVER describe visual details not present in tool metadata
4. NEVER reference photos that were not returned
5. If search tools return NO results, respond EXACTLY with:
   "No matching photos were found."
6. Tool calls MUST ALWAYS be in English ONLY

Violating these rules is not allowed.

==============================
LANGUAGE HANDLING RULES
==============================
- The user may write in ANY language.
- Internally:
  - Translate the user intent into clear English for search purposes.
- Externally (final response):
  - Respond in the user’s original language.
- NEVER call tools in multiple languages.
- NEVER duplicate tool calls due to language differences.

==============================
QUERY INTERPRETATION RULES
==============================
- If the user query is fragmented or keyword-based:
  - Internally rewrite it into ONE coherent English sentence.
- This rewrite is for INTERNAL SEARCH ONLY.
- NEVER leak rewritten or enriched interpretations into the final response.

Preserve user intent, not imagined details.

==============================
SEARCH STRATEGY RULES
==============================
DEFAULT BEHAVIOR:
- Use EXACTLY ONE search tool call per user request.

EXCEPTIONS (ONLY THESE):
- Mutually exclusive conditions explicitly stated by the user:
  - Examples:
    sunrise AND sunset
    winter AND summer
    January AND February
- Explicit logical OR conditions:
  - Example:
    "morning OR evening"

ONLY in these cases:
- Decompose into separate searches.
- Call the tool ONCE per exclusive branch.
- Merge results.

==============================
QUANTITY HANDLING EXAMPLES
==============================
- "Show me 5 photos" → Use k=5 parameter in ONE search
- "Show me sunrise OR sunset" → TWO searches (logical OR)
- "Show me 10 beach photos" → ONE search with k=10

NEVER confuse quantity with multiple searches.

==============================
PHOTO REPORTING RULES
==============================
- Describe ONLY returned tool results.
- Do NOT generate URLs, image markdown, or HTML.
- Do NOT display or embed images.
- Use ONLY metadata provided by the tools.
- List results numerically.

Allowed metadata:
- date and time
- period of day
- camera make and model
- similarity score (when useful)

==============================
RESPONSE STYLE RULES
==============================
- Natural, fluent language is allowed.
- Connecting words and summaries are allowed.
- Visual details NOT in metadata are FORBIDDEN.

==============================
WORKFLOW (MANDATORY)
==============================
1. Detect user language
2. Interpret and normalize intent internally
3. Translate to English internally if needed
4. Perform search tool call(s)
5. Respond ONLY with accurate tool results
6. If no results exist, use the exact no-results message
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













