import boto3
import logging

logger = logging.getLogger(__name__)

class S3PhotoResolver:
    def __init__(self, region='eu-north-1'):
        self.s3_client = boto3.client('s3', region_name=region)

    def generate_presigned_url(
        self,
        bucket: str,
        key: str,
        expires_in: int = 180
    ) -> str | None:
        try:
            return self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=expires_in
            )
        except Exception as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None

