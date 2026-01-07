import json

def flatten_metadata(metadata: dict) -> dict:
    """
    Flatten metadata so that all nested dicts/lists/bytes are
    stored as a JSON string under 'exif', leaving top-level
    scalars unchanged.
    """
    metadata = metadata.copy()  # avoid mutating original
    exif_data = metadata.pop("exif", {})  # remove exif dict if exists
    if exif_data:
        # Convert exif dict to JSON string
        metadata["exif"] = json.dumps(exif_data)
    else:
        # Ensure the key exists even if empty
        metadata["exif"] = "unknown"
    return metadata

def sanitize_metadata(metadata: dict) -> dict:
    """
    Convert None values to safe Chroma-compatible values.
    """
    sanitized = {}

    for key, value in metadata.items():
        if value is None:
            # Choose sensible defaults
            if key in {"year", "month", "hour"}:
                sanitized[key] = -1
            else:
                sanitized[key] = "unknown"
        else:
            sanitized[key] = value

    return sanitized
