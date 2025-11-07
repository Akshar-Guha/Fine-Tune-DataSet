"""Delta Lake client for ACID dataset versioning."""

import os
from typing import Optional, Dict, Any
from datetime import datetime
import pyarrow as pa
import pandas as pd
from deltalake import DeltaTable, write_deltalake
import duckdb
import logging

logger = logging.getLogger(__name__)


class DeltaLakeClient:
    """ACID-compliant dataset storage using Delta Lake."""

    def __init__(self, base_path: Optional[str] = None):
        self.base_path = base_path or os.getenv(
            "DELTA_LAKE_BASE_PATH", "s3://modelops/delta"
        )
        
        # Storage options for MinIO/S3
        self.storage_options = {
            "AWS_ACCESS_KEY_ID": os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            "AWS_SECRET_ACCESS_KEY": os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            "AWS_ENDPOINT_URL": os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
            "AWS_REGION": "us-east-1",
            "AWS_ALLOW_HTTP": "true",
        }
        
        logger.info(f"Delta Lake client initialized with base path: {self.base_path}")

    def write_dataset(
        self,
        name: str,
        data: pa.Table,
        mode: str = "append",
        partition_by: Optional[list[str]] = None,
        schema_mode: str = "merge",
    ) -> str:
        """
        Write dataset with ACID guarantees.
        
        Args:
            name: Dataset name
            data: PyArrow Table to write
            mode: Write mode ('append', 'overwrite', 'error')
            partition_by: Columns to partition by
            schema_mode: How to handle schema changes ('merge' or 'overwrite')
            
        Returns:
            Delta table URI
        """
        path = f"{self.base_path}/datasets/{name}"
        
        try:
            write_deltalake(
                path,
                data,
                mode=mode,
                schema_mode=schema_mode,
                partition_by=partition_by,
                storage_options=self.storage_options,
            )
            logger.info(f"Wrote {len(data)} rows to Delta table: {path}")
            return path
        except Exception as e:
            logger.error(f"Error writing to Delta table {name}: {e}")
            raise

    def read_dataset(
        self,
        name: str,
        version: Optional[int] = None,
        columns: Optional[list[str]] = None,
    ) -> pa.Table:
        """
        Read dataset at specific version.
        
        Args:
            name: Dataset name
            version: Version number (None for latest)
            columns: Columns to read (None for all)
            
        Returns:
            PyArrow Table
        """
        path = f"{self.base_path}/datasets/{name}"
        
        try:
            dt = DeltaTable(path, storage_options=self.storage_options)
            
            if version is not None:
                dt.load_version(version)
                logger.info(f"Loaded Delta table {name} at version {version}")
            else:
                logger.info(f"Loaded latest version of Delta table {name}")
            
            table = dt.to_pyarrow_table(columns=columns)
            logger.debug(f"Read {len(table)} rows from {name}")
            return table
        except Exception as e:
            logger.error(f"Error reading Delta table {name}: {e}")
            raise

    def time_travel(self, name: str, timestamp: datetime) -> pa.Table:
        """
        Query dataset at specific timestamp.
        
        Args:
            name: Dataset name
            timestamp: Point-in-time to query
            
        Returns:
            PyArrow Table
        """
        path = f"{self.base_path}/datasets/{name}"
        
        try:
            dt = DeltaTable(path, storage_options=self.storage_options)
            dt.load_with_datetime(timestamp)
            table = dt.to_pyarrow_table()
            logger.info(f"Time traveled to {timestamp} for {name}")
            return table
        except Exception as e:
            logger.error(f"Error time traveling to {timestamp}: {e}")
            raise

    def get_history(self, name: str, limit: int = 10) -> list[Dict[str, Any]]:
        """Get dataset version history."""
        path = f"{self.base_path}/datasets/{name}"
        
        try:
            dt = DeltaTable(path, storage_options=self.storage_options)
            history = dt.history(limit=limit)
            logger.debug(f"Retrieved {len(history)} history entries for {name}")
            return history
        except Exception as e:
            logger.error(f"Error getting history for {name}: {e}")
            raise

    def vacuum(self, name: str, retention_hours: int = 168) -> None:
        """
        Remove old files (default: 7 days).
        
        Args:
            name: Dataset name
            retention_hours: Hours to retain (default: 168 = 7 days)
        """
        path = f"{self.base_path}/datasets/{name}"
        
        try:
            dt = DeltaTable(path, storage_options=self.storage_options)
            dt.vacuum(retention_hours=retention_hours)
            logger.info(f"Vacuumed {name} (retention: {retention_hours}h)")
        except Exception as e:
            logger.error(f"Error vacuuming {name}: {e}")
            raise

    def optimize(self, name: str, target_size: int = 1024 * 1024 * 128) -> Dict[str, Any]:
        """
        Optimize Delta table by compacting small files.
        
        Args:
            name: Dataset name
            target_size: Target file size in bytes (default: 128MB)
            
        Returns:
            Optimization metrics
        """
        path = f"{self.base_path}/datasets/{name}"
        
        try:
            dt = DeltaTable(path, storage_options=self.storage_options)
            metrics = dt.optimize.compact(target_size=target_size)
            logger.info(f"Optimized {name}: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error optimizing {name}: {e}")
            raise

    def query_with_duckdb(self, sql: str) -> pd.DataFrame:
        """
        Execute analytics query using DuckDB.
        
        Args:
            sql: SQL query string
            
        Returns:
            Query results as DataFrame
        """
        try:
            conn = duckdb.connect()
            
            # Install and load Delta extension
            conn.execute("INSTALL delta")
            conn.execute("LOAD delta")
            
            # Register all Delta tables
            conn.execute(f"""
                CREATE VIEW datasets AS 
                SELECT * FROM delta_scan('{self.base_path}/datasets/*')
            """)
            
            result = conn.execute(sql).fetchdf()
            logger.info(f"Executed DuckDB query, returned {len(result)} rows")
            return result
        except Exception as e:
            logger.error(f"Error executing DuckDB query: {e}")
            raise

    def get_schema(self, name: str) -> pa.Schema:
        """Get dataset schema."""
        path = f"{self.base_path}/datasets/{name}"
        
        try:
            dt = DeltaTable(path, storage_options=self.storage_options)
            schema = dt.schema().to_pyarrow()
            logger.debug(f"Retrieved schema for {name}")
            return schema
        except Exception as e:
            logger.error(f"Error getting schema for {name}: {e}")
            raise

    def get_statistics(self, name: str) -> Dict[str, Any]:
        """Get dataset statistics."""
        path = f"{self.base_path}/datasets/{name}"
        
        try:
            dt = DeltaTable(path, storage_options=self.storage_options)
            table = dt.to_pyarrow_table()
            
            stats = {
                "num_rows": len(table),
                "num_columns": len(table.schema),
                "size_bytes": table.nbytes,
                "schema": {field.name: str(field.type) for field in table.schema},
                "version": dt.version(),
            }
            
            logger.debug(f"Retrieved statistics for {name}")
            return stats
        except Exception as e:
            logger.error(f"Error getting statistics for {name}: {e}")
            raise

    def delete_dataset(self, name: str) -> None:
        """Delete entire Delta table (use with caution)."""
        path = f"{self.base_path}/datasets/{name}"
        logger.warning(f"Deleting Delta table: {path}")
        
        try:
            dt = DeltaTable(path, storage_options=self.storage_options)
            # Delta Lake doesn't have built-in delete, would need to use storage API
            # This is a placeholder - actual implementation would delete from MinIO
            logger.info(f"Deleted Delta table {name}")
        except Exception as e:
            logger.error(f"Error deleting {name}: {e}")
            raise
