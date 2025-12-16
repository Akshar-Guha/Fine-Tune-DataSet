"""Dataset management using DuckDB for local storage."""
import json
import os
from datetime import datetime
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
        directory = os.path.dirname(os.path.abspath(db_path))
        if directory and db_path not in {":memory:", "memory"}:
            os.makedirs(directory, exist_ok=True)
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
                quality_status VARCHAR,
                quality_score DOUBLE,
                validation_report_path VARCHAR,
                last_validated TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Ensure new columns exist for quality metadata
        self.conn.execute(
            "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS quality_status VARCHAR"
        )
        self.conn.execute(
            "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS quality_score DOUBLE"
        )
        self.conn.execute(
            "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS validation_report_path VARCHAR"
        )
        self.conn.execute(
            "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS last_validated TIMESTAMP"
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_validation_reports (
                dataset_id VARCHAR,
                dataset_name VARCHAR,
                status VARCHAR,
                quality_score DOUBLE,
                report_path VARCHAR,
                issues JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    def add_dataset(
        self,
        dataset_id: str,
        name: str,
        file_path: str,
        description: str = "",
        format: str = "parquet",
        *,
        quality_status: Optional[str] = None,
        quality_score: Optional[float] = None,
        validation_report_path: Optional[str] = None,
        last_validated: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Register a new dataset.

        Args:
            dataset_id: Unique dataset identifier
            name: Dataset name
            file_path: Path to dataset file
            description: Dataset description
            format: File format (parquet, csv, json)
            quality_status: Latest validation status
            quality_score: Validation quality score
            validation_report_path: Path to stored validation report
            last_validated: Timestamp for last validation

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
        schema_json = json.dumps(schema)

        # Ensure we replace any previous entry for this dataset
        self.conn.execute(
            "DELETE FROM datasets WHERE id = ?",
            [dataset_id]
        )

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
                schema,
                quality_status,
                quality_score,
                validation_report_path,
                last_validated
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            dataset_id,
            name,
            description,
            file_path,
            format,
            rows,
            columns,
            size_bytes,
            schema_json,
            quality_status,
            quality_score,
            validation_report_path,
            last_validated
        ])
        record = self.get_dataset(dataset_id)

        if record is None:
            raise RuntimeError("Failed to fetch dataset metadata after insert")

        return record

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
                quality_status,
                quality_score,
                validation_report_path,
                last_validated,
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
            "schema": self._deserialize_json(result[8]),
            "quality_status": result[9],
            "quality_score": result[10],
            "validation_report_path": result[11],
            "last_validated": result[12],
            "created_at": result[13]
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
                quality_status,
                quality_score,
                validation_report_path,
                last_validated,
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
                "schema": self._deserialize_json(row[8]),
                "quality_status": row[9],
                "quality_score": row[10],
                "validation_report_path": row[11],
                "last_validated": row[12],
                "created_at": row[13]
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

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def record_validation_result(
        self,
        dataset_id: str,
        *,
        dataset_name: Optional[str],
        status: str,
        quality_score: Optional[float],
        report_path: str,
        issues: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Persist validation run outcomes."""

        payload = json.dumps(issues or [])
        self.conn.execute(
            """
            INSERT INTO dataset_validation_reports (
                dataset_id,
                dataset_name,
                status,
                quality_score,
                report_path,
                issues
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [dataset_id, dataset_name, status, quality_score, report_path, payload],
        )

    def get_latest_validation_result(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the latest validation record for a dataset."""

        result = self.conn.execute(
            """
            SELECT
                dataset_id,
                dataset_name,
                status,
                quality_score,
                report_path,
                issues,
                created_at
            FROM dataset_validation_reports
            WHERE dataset_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [dataset_id],
        ).fetchone()

        if not result:
            return None

        return {
            "dataset_id": result[0],
            "dataset_name": result[1],
            "status": result[2],
            "quality_score": result[3],
            "report_path": result[4],
            "issues": self._deserialize_json(result[5]) or [],
            "created_at": result[6],
        }

    def list_validation_results(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent validation records."""

        results = self.conn.execute(
            """
            SELECT
                dataset_id,
                dataset_name,
                status,
                quality_score,
                report_path,
                issues,
                created_at
            FROM dataset_validation_reports
            ORDER BY created_at DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()

        return [
            {
                "dataset_id": row[0],
                "dataset_name": row[1],
                "status": row[2],
                "quality_score": row[3],
                "report_path": row[4],
                "issues": self._deserialize_json(row[5]) or [],
                "created_at": row[6],
            }
            for row in results
        ]

    def get_validation_history(
        self,
        dataset_id: str,
        *,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return validation history for a dataset."""

        query = [
            """
            SELECT
                dataset_id,
                dataset_name,
                status,
                quality_score,
                report_path,
                issues,
                created_at
            FROM dataset_validation_reports
            WHERE dataset_id = ?
            ORDER BY created_at DESC
            """
        ]
        params: List[Any] = [dataset_id]

        if limit is not None:
            query.append("LIMIT ?")
            params.append(limit)

        results = self.conn.execute("\n".join(query), params).fetchall()

        return [
            {
                "dataset_id": row[0],
                "dataset_name": row[1],
                "status": row[2],
                "quality_score": row[3],
                "report_path": row[4],
                "issues": self._deserialize_json(row[5]) or [],
                "created_at": row[6],
            }
            for row in results
        ]

    def update_dataset_quality(
        self,
        dataset_id: str,
        *,
        status: Optional[str],
        quality_score: Optional[float],
        validation_report_path: Optional[str],
        last_validated: Optional[datetime] = None,
    ) -> bool:
        """Update quality metadata for an existing dataset."""

        timestamp = last_validated or datetime.utcnow()
        result = self.conn.execute(
            """
            UPDATE datasets
            SET
                quality_status = ?,
                quality_score = ?,
                validation_report_path = ?,
                last_validated = ?
            WHERE id = ?
            """,
            [status, quality_score, validation_report_path, timestamp, dataset_id],
        )

        return result.rowcount > 0

    def _deserialize_json(self, payload: Any) -> Any:
        if payload is None:
            return None
        if isinstance(payload, (dict, list)):
            return payload
        if isinstance(payload, str):
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return payload
        return payload
