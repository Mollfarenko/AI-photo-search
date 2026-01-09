from langchain_core.messages import ToolMessage
import json

def extract_photos(messages: list[BaseMessage]) -> list[dict]:
    photos = []

    for msg in messages:
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)

                # depends on your unified_search output
                metadatas = data.get("metadatas", [[]])[0]
                distances = data.get("distances", [[]])[0]

                for meta, dist in zip(metadatas, distances):
                    photos.append({
                        "photo_id": meta["id"],
                        "bucket": meta["bucket"],
                        "photo_key": meta["photo_key"],
                        "thumbnail_key": meta.get("thumbnail_key"),
                        "taken_at": meta.get("taken_at"),
                        "period_of_day": meta.get("period_of_day"),
                        "camera_make": meta.get("camera_make"),
                        "camera_model": meta.get("camera_model"),
                        "distance": round(dist, 4),
                    })

            except Exception as e:
                logger.error(f"Failed to parse tool output: {e}")

    return photos
