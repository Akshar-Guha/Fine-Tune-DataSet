"""Dataset management using DuckDB for local storage."""
import os
from typing import Dict, Any, List, Optional

import duckdb
import pandas as pd


class DatasetManager:
    """Manage datasets using DuckDB (embedded, no server required)."""

    def __init__(self, db_path: str = "./data/datasets.duckdb"):
        """Initialize dataset manager.

        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = duckdb.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                description TEXT,
                file_path VARCHAR NOT NULL,
                format VARCHAR NOT NULL,
                rows INTEGER,
                columns INTEGER,
                size_bytes BIGINT,
                schema JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def add_dataset(
        self,
        dataset_id: str,
        name: str,
        file_path: str,
        description: str = "",
        format: str = "parquet"
    ) -> Dict[str, Any]:
        """Register a new dataset.

        Args:
            dataset_id: Unique dataset identifier
            name: Dataset name
            file_path: Path to dataset file
            description: Dataset description
            format: File format (parquet, csv, json)

        Returns:
            Dataset metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Count rows
        if format == "parquet":
            df = pd.read_parquet(file_path)
        elif format == "csv":
            df = pd.read_csv(file_path)
        elif format == "json":
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        rows = len(df)
        columns = len(df.columns) if hasattr(df, "columns") else 0
        size_bytes = os.path.getsize(file_path)
        schema = [
            {"name": column, "dtype": str(dtype)}
            for column, dtype in getattr(df, "dtypes", {}).items()
        ] if hasattr(df, "dtypes") else []

        # Insert metadata
        self.conn.execute("""
            INSERT INTO datasets (
                id,
                name,
                description,
                file_path,
                format,
                rows,
                columns,
                size_bytes,
                schema
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            dataset_id,
            name,
            description,
            file_path,
            format,
            rows,
            columns,
            size_bytes,
            schema
        ])

        return {
            "id": dataset_id,
            "name": name,
            "file_path": file_path,
            "format": format,
            "rows": rows,
            "columns": columns,
            "size_bytes": size_bytes,
            "schema": schema
        }

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get dataset metadata.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Dataset metadata or None
        """
        result = self.conn.execute(
            """
            SELECT
                id,
                name,
                description,
                file_path,
                format,
                rows,
                columns,
                size_bytes,
                schema,
                created_at
            FROM datasets
            WHERE id = ?
            """,
            [dataset_id]
        ).fetchone()

        if not result:
            return None

        return {
            "id": result[0],
            "name": result[1],
            "description": result[2],
            "file_path": result[3],
            "format": result[4],
            "rows": result[5],
            "columns": result[6],
            "size_bytes": result[7],
            "schema": result[8],
            "created_at": result[9]
        }

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all datasets.

        Returns:
            List of dataset metadata
        """
        results = self.conn.execute(
            """
            SELECT
                id,
                name,
                description,
                file_path,
                format,
                rows,
                columns,
                size_bytes,
                schema,
                created_at
            FROM datasets
            ORDER BY created_at DESC
            """
        ).fetchall()

        return [
            {
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "file_path": row[3],
                "format": row[4],
                "rows": row[5],
                "columns": row[6],
                "size_bytes": row[7],
                "schema": row[8],
                "created_at": row[9]
            }
            for row in results
        ]

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete dataset metadata.

        Args:
            dataset_id: Dataset identifier

        Returns:
            True if a dataset was deleted, False otherwise
        """
        result = self.conn.execute(
            "DELETE FROM datasets WHERE id = ?",
            [dataset_id]
        )
        return result.rowcount > 0

    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query on datasets.

        Args:
            sql: SQL query

        Returns:
            Query results as DataFrame
        """
        return self.conn.execute(sql).fetchdf()

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
