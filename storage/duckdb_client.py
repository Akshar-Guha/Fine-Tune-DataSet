"""DuckDB client for analytics queries."""

import os
from typing import Optional
import duckdb
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DuckDBClient:
    """In-process analytics database for fast queries."""

    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path or ":memory:"
        self.conn = duckdb.connect(self.database_path)
        
        # Install and load extensions
        self._install_extensions()
        
        logger.info(f"DuckDB client initialized (path: {self.database_path})")

    def _install_extensions(self) -> None:
        """Install required extensions."""
        extensions = ["delta", "httpfs", "parquet"]
        
        for ext in extensions:
            try:
                self.conn.execute(f"INSTALL {ext}")
                self.conn.execute(f"LOAD {ext}")
                logger.debug(f"Loaded DuckDB extension: {ext}")
            except Exception as e:
                logger.warning(f"Failed to load extension {ext}: {e}")

    def configure_s3(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ) -> None:
        """Configure S3/MinIO access."""
        endpoint = endpoint or os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
        access_key = access_key or os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = secret_key or os.getenv("MINIO_SECRET_KEY", "minioadmin")
        
        self.conn.execute(f"""
            SET s3_endpoint='{endpoint}';
            SET s3_access_key_id='{access_key}';
            SET s3_secret_access_key='{secret_key}';
            SET s3_use_ssl=false;
        """)
        
        logger.info("DuckDB S3 access configured")

    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        try:
            result = self.conn.execute(sql).fetchdf()
            logger.debug(f"Query returned {len(result)} rows")
            return result
        except Exception as e:
            logger.error(f"Query error: {e}")
            raise

    def query_parquet(self, parquet_path: str, sql: str = None) -> pd.DataFrame:
        """Query Parquet file directly."""
        if sql is None:
            sql = f"SELECT * FROM '{parquet_path}'"
        else:
            sql = sql.replace("{{path}}", f"'{parquet_path}'")
        
        return self.query(sql)

    def query_delta(self, delta_path: str, sql: str = None) -> pd.DataFrame:
        """Query Delta Lake table."""
        if sql is None:
            sql = f"SELECT * FROM delta_scan('{delta_path}')"
        else:
            sql = sql.replace("{{path}}", f"delta_scan('{delta_path}')")
        
        return self.query(sql)

    def create_view(self, name: str, sql: str) -> None:
        """Create view from query."""
        self.conn.execute(f"CREATE OR REPLACE VIEW {name} AS {sql}")
        logger.info(f"Created view: {name}")

    def export_to_parquet(self, sql: str, output_path: str) -> None:
        """Export query results to Parquet."""
        self.conn.execute(f"COPY ({sql}) TO '{output_path}' (FORMAT PARQUET)")
        logger.info(f"Exported query results to {output_path}")

    def get_table_info(self, table_name: str) -> pd.DataFrame:
        """Get table schema information."""
        return self.query(f"DESCRIBE {table_name}")

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
        logger.info("DuckDB connection closed")
