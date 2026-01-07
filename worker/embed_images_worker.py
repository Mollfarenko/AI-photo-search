import json
import time
import boto3
import logging
from pathlib import Path
from embeddings.clip_model import load_clip_model
from embeddings.image_embedder import embed_image
from tools.format_metadata import flatten_metadata, sanitize_metadata
from storage.chroma_store import get_chroma_client, get_collection

# ---- Setup logging ----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---- AWS clients ----
sqs = boto3.client("sqs", region_name="eu-north-1")
s3 = boto3.client("s3", region_name="eu-north-1")

QUEUE_URL = "https://sqs.eu-north-1.amazonaws.com/324236699129/photo-embedding-jobs"
TMP_DIR = Path("/home/ec2-user/app/tmp/images")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load heavy resources ONCE ----
logger.info("Loading CLIP model...")
model, processor, device = load_clip_model()
logger.info(f"Model loaded on device: {device}")

logger.info("Connecting to ChromaDB...")
chroma = get_chroma_client("/home/ec2-user/app/data/chroma")
collection = get_collection(chroma)
logger.info(f"Connected to collection: {collection.name}")

# ---- Graceful shutdown ----
import signal
import sys

shutdown_flag = False

def signal_handler(sig, frame):
    global shutdown_flag
    logger.info("Shutdown signal received, finishing current message...")
    shutdown_flag = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

logger.info("Worker started. Waiting for messages...")

while not shutdown_flag:
    try:
        response = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
            VisibilityTimeout=300  # 5 minutes - ADD THIS
        )

        messages = response.get("Messages", [])
        if not messages:
            continue

        msg = messages[0]
        receipt_handle = msg["ReceiptHandle"]

        try:
            body = json.loads(msg["Body"])
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in message: {e}")
            # Delete malformed message
            sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=receipt_handle)
            continue

        # Validate message structure
        required_fields = ["bucket", "image_key", "photo_id", "metadata_key"]
        if not all(field in body for field in required_fields):
            logger.error(f"Missing required fields in message: {body}")
            sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=receipt_handle)
            continue

        bucket = body["bucket"]
        image_key = body["image_key"]
        photo_id = body["photo_id"]
        metadata_key = body["metadata_key"]

        # Prepare local paths
        local_path = TMP_DIR / f"{photo_id}.jpg"
        local_metadata_path = TMP_DIR / f"{photo_id}.json"

        try:
            # 1. Download image
            logger.info(f"Downloading {photo_id} from s3://{bucket}/{image_key}")
            s3.download_file(bucket, image_key, str(local_path))

            # 2. Download metadata JSON
            logger.info(f"Downloading metadata for {photo_id} from s3://{bucket}/{metadata_key}")
            s3.download_file(bucket, metadata_key, str(local_metadata_path))
            with open(local_metadata_path, "r") as f:
                metadata = json.load(f)
            
            # 2.1. Flatten the metadata
            metadata = flatten_metadata(metadata)
            metadata = sanitize_metadata(metadata)

            # 3. Embed
            logger.info(f"Generating embedding for {photo_id}")
            embedding = embed_image(
                image_path=str(local_path),
                model=model,
                processor=processor,
                device=device
            )

            # 4. Store in Chroma
            logger.info(f"Storing embedding for {photo_id}")
            collection.add(
                ids=[photo_id],
                embeddings=embedding.cpu().tolist(),
                metadatas=[metadata]
            )

            # 5. Delete SQS message (success)
            sqs.delete_message(
                QueueUrl=QUEUE_URL,
                ReceiptHandle=receipt_handle
            )

            logger.info(f"✓ Successfully embedded photo {photo_id}")

        except Exception as e:
            logger.error(f"Error processing {photo_id}: {e}", exc_info=True)
            # Don't delete message - it will return to queue after visibility timeout
            # Consider adding to DLQ after X retries

        finally:
            # Clean up temp file
            if local_path.exists():
                local_path.unlink()
                logger.debug(f"Cleaned up temp file: {local_path}")

    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
        time.sleep(5)  # Wait before retrying

logger.info("Worker shutdown complete")
sys.exit(0)
