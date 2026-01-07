from langchain_core.tools import StructuredTool
import logging
from pathlib import Path
from embeddings.clip_model import load_clip_model
from embeddings.image_embedder import embed_image
from storage.chroma_store import get_chroma_client, get_collection
from tools.metadata_filter import build_where_clause
from tools.unified_search import unified_search

logger = logging.getLogger(__name__)

# --- Load shared resources once ---
_model, _processor, _device = load_clip_model()
_client = get_chroma_client()
_collection = get_collection(_client)


def search_by_image_impl(
    image_path: str,
    year: int | None = None,
    month: int | None = None,
    time_of_day: str | None = None,
    camera_make: str | None = None,
    camera_model: str | None = None,
    k: int = 5
) -> list[dict]:
    """
    Search photos using an image as a query with optional metadata filters.

    Args:
        image_path: Path to the query image file (local or S3)
        year: Filter by year (e.g., 2023)
        month: Filter by month (1-12)
        time_of_day: Filter by time ('morning', 'afternoon', 'evening', 'night')
        k: Number of results to return (max 20)

    Returns:
        List of similar photo dictionaries with metadata and similarity scores
    """
    logger.info(f"Image search: path='{image_path}', filters={{year={year}, month={month}, time={time_of_day}}}, k={k}")

    try:
        # Validate image path
        if not image_path or not image_path.strip():
            logger.error("Empty image path provided")
            return []

        # Check if file exists (for local paths)
        image_file = Path(image_path)
        if not image_file.exists():
            logger.error(f"Image file not found: {image_path}")
            return []

        # Validate filters
        if month is not None and not (1 <= month <= 12):
            logger.error(f"Invalid month: {month}")
            return []

        if time_of_day and time_of_day not in ['morning', 'afternoon', 'evening', 'night']:
            logger.error(f"Invalid time_of_day: {time_of_day}")
            return []

        if camera_make is not None:
            if not isinstance(camera_make, str) or not camera_make.strip():
                logger.error(f"Invalid camera_make: {camera_make}")
                return []
            camera_make = camera_make.strip()

        if camera_model is not None:
            if not isinstance(camera_model, str) or not camera_model.strip():
                logger.error(f"Invalid camera_model: {camera_model}")
                return []
            camera_model = camera_model.strip()

        # Limit k
        k = min(max(k, 1), 20)

        # Generate embedding
        logger.debug(f"Embedding image: {image_path}")
        query_vec = embed_image(image_path, _model, _processor, _device)

        # Build filters
        where = build_where_clause(year=year, month=month, time_of_day=time_of_day, camera_make=camera_make, camera_model=camera_model)

        # Search
        results = unified_search(
            collection=_collection,
            query_embedding=query_vec.cpu().tolist(),
            where=where if where else None,
            k=k
        )

        logger.info(f"Found {len(results)} similar photos")
        return results

    except FileNotFoundError as e:
        logger.error(f"Image file not found: {e}")
        return []
    except Exception as e:
        logger.error(f"Image search failed: {e}", exc_info=True)
        return []


search_by_image_tool = StructuredTool.from_function(
    func=search_by_image_impl,
    name="search_photos_by_image",
    description=(
        "Search for similar photos using an uploaded image as the query. "
        "This finds photos that look visually similar to the provided image.\n\n"
        "Required:\n"
        "- image_path: Full path to the uploaded image file\n\n"
        "Optional metadata filters (only use if user explicitly mentions them):\n"
        "- year: Integer year (e.g., 2023, 2024)\n"
        "- month: Integer 1-12 where 1=January, 12=December\n"
        "- time_of_day: 'morning', 'afternoon', 'evening', or 'night'\n"
        "- camera_make: Camera brand (e.g., 'HUAWEI', 'Apple', 'Canon')\n"
        "- camera_model: Specific camera model (e.g., 'VOG-L29', 'iPhone 14 Pro')\n"
        "- k: Number of similar results (default 5, max 20)\n\n"
        "Use cases:\n"
        "User uploads family photo → Find similar group photos\n\n"
        "The tool compares visual similarity not metadata. "
    )
)
