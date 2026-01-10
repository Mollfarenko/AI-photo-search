from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import List

import tempfile
from pathlib import Path

from agents.agent_runtime import run_agent_text, run_agent_image
from utilities.url_generator import S3PhotoResolver

# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------

app = FastAPI(title="AI Photo Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

s3_resolver = S3PhotoResolver()

# -----------------------------------------------------------------------------
# Request / Response models
# -----------------------------------------------------------------------------

class TextSearchRequest(BaseModel):
    query: str


class PhotoResponse(BaseModel):
    photo_id: str
    thumbnail_url: str | None
    photo_url: str | None
    taken_at: str | None
    period_of_day: str | None
    similarity_score: float | None


class SearchResponse(BaseModel):
    response: str
    photos: List[PhotoResponse]
    tool_calls: int


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def attach_s3_urls(photos: list[dict]) -> list[PhotoResponse]:
    """Generate presigned URLs for photos returned by the agent."""
    enriched = []

    for photo in photos:
        bucket = photo.get("bucket")
        photo_key = photo.get("photo_key")
        thumbnail_key = photo.get("thumbnail_key")

        photo_url = (
            s3_resolver.generate_presigned_url(bucket, photo_key)
            if bucket and photo_key
            else None
        )

        thumbnail_url = (
            s3_resolver.generate_presigned_url(bucket, thumbnail_key)
            if bucket and thumbnail_key
            else None
        )

        enriched.append(
            PhotoResponse(
                photo_id=photo.get("photo_id"),
                thumbnail_url=thumbnail_url,
                photo_url=photo_url,
                taken_at=photo.get("taken_at"),
                period_of_day=photo.get("period_of_day"),
                similarity_score=photo.get("similarity_score"),
            )
        )

    return enriched


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.post("/search/text", response_model=SearchResponse)
async def search_by_text(data: TextSearchRequest):
    result = await run_in_threadpool(run_agent_text, data.query)

    photos_with_urls = attach_s3_urls(result.get("photos", []))

    return SearchResponse(
        response=result.get("response", ""),
        photos=photos_with_urls,
        tool_calls=result.get("tool_calls", 0),
    )


@app.post("/search/image", response_model=SearchResponse)
async def search_by_image(image: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=Path(image.filename).suffix
    ) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        result = await run_in_threadpool(run_agent_image, tmp_path)
        photos_with_urls = attach_s3_urls(result.get("photos", []))

        return SearchResponse(
            response=result.get("response", ""),
            photos=photos_with_urls,
            tool_calls=result.get("tool_calls", 0),
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

