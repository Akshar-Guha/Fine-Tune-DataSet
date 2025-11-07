"""Storage layer tests."""
import pytest
from storage.minio_client import MinIOClient


@pytest.fixture
def minio_client():
    """MinIO client fixture."""
    return MinIOClient()


def test_minio_client_init(minio_client):
    """Test MinIO client initialization."""
    assert minio_client is not None
    assert minio_client.endpoint is not None
