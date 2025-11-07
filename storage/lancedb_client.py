"""LanceDB client for fast vector search."""

import os
from typing import Optional, List, Dict, Any
import lancedb
from fastembed import TextEmbedding
import pyarrow as pa
import logging

logger = logging.getLogger(__name__)


class LanceDBClient:
    """Disk-based vector database with SQL interface."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ):
        self.db_path = db_path or os.getenv("LANCEDB_PATH", "s3://modelops/lancedb")
        
        # Configure storage for S3
        storage_options = {
            "aws_access_key_id": os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            "aws_secret_access_key": os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            "aws_endpoint": os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
            "aws_region": "us-east-1",
        }
        
        try:
            self.db = lancedb.connect(self.db_path, storage_options=storage_options)
            logger.info(f"LanceDB connected to: {self.db_path}")
        except Exception as e:
            logger.warning(f"S3 connection failed, using local storage: {e}")
            self.db = lancedb.connect("./lancedb_local")
        
        # Initialize embedding model
        self.embedding_model = TextEmbedding(
            model_name=embedding_model, max_length=512
        )
        logger.info(f"Loaded embedding model: {embedding_model}")

    def create_index(
        self,
        name: str,
        documents: List[Dict[str, Any]],
        metadata_schema: Optional[Dict] = None,
        metric: str = "cosine",
    ) -> str:
        """
        Create vector index with metadata.
        
        Args:
            name: Index name
            documents: List of documents with 'text' field
            metadata_schema: Optional schema for metadata
            metric: Distance metric ('cosine', 'l2', 'dot')
            
        Returns:
            Index ID
        """
        try:
            # Generate embeddings (10x faster than sentence-transformers)
            texts = [doc["text"] for doc in documents]
            embeddings = list(self.embedding_model.embed(texts))
            
            # Prepare data
            data = []
            for doc, emb in zip(documents, embeddings):
                record = {
                    "id": doc["id"],
                    "text": doc["text"],
                    "vector": emb,
                }
                # Add metadata fields
                if "metadata" in doc:
                    record.update(doc["metadata"])
                data.append(record)
            
            # Create table
            table = self.db.create_table(name, data=data, mode="overwrite")
            
            # Create vector index (IVF-PQ for large scale)
            table.create_index(metric=metric, num_partitions=256, num_sub_vectors=96)
            
            logger.info(f"Created LanceDB index '{name}' with {len(data)} documents")
            return name
        except Exception as e:
            logger.error(f"Error creating index {name}: {e}")
            raise

    def add_documents(self, table_name: str, documents: List[Dict[str, Any]]) -> None:
        """Add documents to existing index."""
        try:
            table = self.db.open_table(table_name)
            
            # Generate embeddings
            texts = [doc["text"] for doc in documents]
            embeddings = list(self.embedding_model.embed(texts))
            
            # Prepare data
            data = []
            for doc, emb in zip(documents, embeddings):
                record = {
                    "id": doc["id"],
                    "text": doc["text"],
                    "vector": emb,
                }
                if "metadata" in doc:
                    record.update(doc["metadata"])
                data.append(record)
            
            # Add to table
            table.add(data)
            logger.info(f"Added {len(data)} documents to {table_name}")
        except Exception as e:
            logger.error(f"Error adding documents to {table_name}: {e}")
            raise

    def search(
        self,
        table_name: str,
        query: str,
        k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Search with SQL-like filtering.
        
        Args:
            table_name: Index name
            query: Query text
            k: Number of results
            filters: Metadata filters (e.g., {'year': 2023})
            
        Returns:
            List of search results
        """
        try:
            table = self.db.open_table(table_name)
            
            # Embed query
            query_vector = list(self.embedding_model.embed([query]))[0]
            
            # Build filter SQL
            search = table.search(query_vector).limit(k)
            
            if filters:
                conditions = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        conditions.append(f"{key} = '{value}'")
                    else:
                        conditions.append(f"{key} = {value}")
                filter_sql = " AND ".join(conditions)
                search = search.where(filter_sql)
            
            # Execute search
            results = search.to_list()
            logger.debug(f"Search returned {len(results)} results from {table_name}")
            return results
        except Exception as e:
            logger.error(f"Error searching {table_name}: {e}")
            raise

    def hybrid_search(
        self,
        table_name: str,
        query: str,
        k: int = 10,
        fts_weight: float = 0.3,
    ) -> List[Dict]:
        """
        Hybrid vector + full-text search.
        
        Args:
            table_name: Index name
            query: Query text
            k: Number of results
            fts_weight: Weight for full-text search (0-1)
            
        Returns:
            Reranked results
        """
        try:
            table = self.db.open_table(table_name)
            
            # Vector search
            query_vector = list(self.embedding_model.embed([query]))[0]
            vector_results = table.search(query_vector).limit(k * 2).to_list()
            
            # Full-text search (built into LanceDB)
            fts_results = table.search(query, query_type="fts").limit(k * 2).to_list()
            
            # Combine scores using reciprocal rank fusion
            combined = self._rrf_fusion(vector_results, fts_results, fts_weight)
            
            logger.debug(f"Hybrid search returned {len(combined[:k])} results")
            return combined[:k]
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            raise

    def _rrf_fusion(
        self,
        vector_results: List[Dict],
        fts_results: List[Dict],
        fts_weight: float,
    ) -> List[Dict]:
        """Reciprocal Rank Fusion for combining results."""
        scores = {}
        k = 60  # RRF constant
        
        # Score vector results
        for rank, result in enumerate(vector_results):
            doc_id = result["id"]
            scores[doc_id] = scores.get(doc_id, 0) + (1 - fts_weight) / (k + rank + 1)
        
        # Score FTS results
        for rank, result in enumerate(fts_results):
            doc_id = result["id"]
            scores[doc_id] = scores.get(doc_id, 0) + fts_weight / (k + rank + 1)
        
        # Create combined results
        all_results = {r["id"]: r for r in vector_results + fts_results}
        combined = [
            {**all_results[doc_id], "fusion_score": score}
            for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return combined

    def update_documents(self, table_name: str, updates: List[Dict]) -> None:
        """Update existing documents."""
        try:
            table = self.db.open_table(table_name)
            
            for update in updates:
                # Re-embed if text changed
                if "text" in update:
                    embedding = list(self.embedding_model.embed([update["text"]]))[0]
                    update["vector"] = embedding
                
                table.update(where=f"id = '{update['id']}'", values=update)
            
            logger.info(f"Updated {len(updates)} documents in {table_name}")
        except Exception as e:
            logger.error(f"Error updating documents: {e}")
            raise

    def delete_documents(self, table_name: str, doc_ids: List[str]) -> None:
        """Delete documents by ID."""
        try:
            table = self.db.open_table(table_name)
            
            for doc_id in doc_ids:
                table.delete(f"id = '{doc_id}'")
            
            logger.info(f"Deleted {len(doc_ids)} documents from {table_name}")
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise

    def get_stats(self, table_name: str) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            table = self.db.open_table(table_name)
            count = table.count_rows()
            
            stats = {
                "table_name": table_name,
                "num_documents": count,
                "schema": str(table.schema),
            }
            
            logger.debug(f"Retrieved stats for {table_name}")
            return stats
        except Exception as e:
            logger.error(f"Error getting stats for {table_name}: {e}")
            raise

    def list_tables(self) -> List[str]:
        """List all tables in database."""
        try:
            tables = self.db.table_names()
            logger.debug(f"Found {len(tables)} tables")
            return tables
        except Exception as e:
            logger.error(f"Error listing tables: {e}")
            raise

    def drop_table(self, table_name: str) -> None:
        """Drop table."""
        try:
            self.db.drop_table(table_name)
            logger.info(f"Dropped table {table_name}")
        except Exception as e:
            logger.error(f"Error dropping table {table_name}: {e}")
            raise
