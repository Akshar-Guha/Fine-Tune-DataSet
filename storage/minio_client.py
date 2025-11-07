"""MinIO client for S3-compatible object storage."""

import os
from typing import Optional, BinaryIO
from pathlib import Path
from minio import Minio
from minio.error import S3Error
import logging

logger = logging.getLogger(__name__)


class MinIOClient:
    """S3-compatible object storage client using MinIO."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        secure: bool = False,
    ):
        self.endpoint = endpoint or os.getenv("MINIO_ENDPOINT", "localhost:9000")
        # Remove http:// or https:// prefix if present
        if "://" in self.endpoint:
            self.endpoint = self.endpoint.split("://")[1]
        
        self.access_key = access_key or os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = secret_key or os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.secure = secure

        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure,
        )
        
        logger.info(f"MinIO client initialized for {self.endpoint}")

    def ensure_bucket(self, bucket_name: str) -> None:
        """Create bucket if it doesn't exist."""
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logger.info(f"Created bucket: {bucket_name}")
            else:
                logger.debug(f"Bucket exists: {bucket_name}")
        except S3Error as e:
            logger.error(f"Error ensuring bucket {bucket_name}: {e}")
            raise

    def upload_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: str,
        content_type: Optional[str] = None,
    ) -> str:
        """Upload file to MinIO."""
        self.ensure_bucket(bucket_name)
        
        try:
            file_size = Path(file_path).stat().st_size
            self.client.fput_object(
                bucket_name,
                object_name,
                file_path,
                content_type=content_type,
            )
            uri = f"s3://{bucket_name}/{object_name}"
            logger.info(f"Uploaded {file_size} bytes to {uri}")
            return uri
        except S3Error as e:
            logger.error(f"Error uploading {file_path}: {e}")
            raise

    def upload_data(
        self,
        bucket_name: str,
        object_name: str,
        data: BinaryIO,
        length: int,
        content_type: Optional[str] = None,
    ) -> str:
        """Upload binary data to MinIO."""
        self.ensure_bucket(bucket_name)
        
        try:
            self.client.put_object(
                bucket_name,
                object_name,
                data,
                length,
                content_type=content_type,
            )
            uri = f"s3://{bucket_name}/{object_name}"
            logger.info(f"Uploaded {length} bytes to {uri}")
            return uri
        except S3Error as e:
            logger.error(f"Error uploading data to {object_name}: {e}")
            raise

    def download_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: str,
    ) -> str:
        """Download file from MinIO."""
        try:
            self.client.fget_object(bucket_name, object_name, file_path)
            logger.info(f"Downloaded s3://{bucket_name}/{object_name} to {file_path}")
            return file_path
        except S3Error as e:
            logger.error(f"Error downloading {object_name}: {e}")
            raise

    def download_data(self, bucket_name: str, object_name: str) -> bytes:
        """Download object data as bytes."""
        try:
            response = self.client.get_object(bucket_name, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            logger.info(f"Downloaded {len(data)} bytes from s3://{bucket_name}/{object_name}")
            return data
        except S3Error as e:
            logger.error(f"Error downloading data from {object_name}: {e}")
            raise

    def delete_object(self, bucket_name: str, object_name: str) -> None:
        """Delete object from MinIO."""
        try:
            self.client.remove_object(bucket_name, object_name)
            logger.info(f"Deleted s3://{bucket_name}/{object_name}")
        except S3Error as e:
            logger.error(f"Error deleting {object_name}: {e}")
            raise

    def list_objects(self, bucket_name: str, prefix: str = "") -> list[str]:
        """List objects in bucket with optional prefix."""
        try:
            objects = self.client.list_objects(bucket_name, prefix=prefix, recursive=True)
            object_names = [obj.object_name for obj in objects]
            logger.debug(f"Found {len(object_names)} objects in {bucket_name}/{prefix}")
            return object_names
        except S3Error as e:
            logger.error(f"Error listing objects in {bucket_name}: {e}")
            raise

    def object_exists(self, bucket_name: str, object_name: str) -> bool:
        """Check if object exists."""
        try:
            self.client.stat_object(bucket_name, object_name)
            return True
        except S3Error:
            return False

    def get_presigned_url(
        self,
        bucket_name: str,
        object_name: str,
        expires_seconds: int = 3600,
    ) -> str:
        """Get presigned URL for temporary access."""
        try:
            from datetime import timedelta
            url = self.client.presigned_get_object(
                bucket_name,
                object_name,
                expires=timedelta(seconds=expires_seconds),
            )
            logger.debug(f"Generated presigned URL for {object_name}")
            return url
        except S3Error as e:
            logger.error(f"Error generating presigned URL: {e}")
            raise
