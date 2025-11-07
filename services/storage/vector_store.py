"""Vector storage using LanceDB (embedded, no server)."""
import os
from typing import List, Dict, Any, Optional

try:
    import lancedb
    from sentence_transformers import SentenceTransformer
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False
    print("Warning: lancedb or sentence-transformers not installed")


class VectorStore:
    """Embedded vector store using LanceDB."""

    def __init__(
        self,
        db_path: str = "./data/lancedb",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """Initialize vector store.

        Args:
            db_path: Path to LanceDB database
            embedding_model: Sentence transformer model name
        """
        if not LANCEDB_AVAILABLE:
            raise ImportError(
                "lancedb not installed. Install with: pip install lancedb"
            )

        os.makedirs(db_path, exist_ok=True)
        self.db = lancedb.connect(db_path)
        self.embedding_model = SentenceTransformer(embedding_model)

    def create_collection(
        self,
        name: str,
        schema: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create a new collection.

        Args:
            name: Collection name
            schema: Optional schema definition
        """
        # LanceDB creates tables automatically on first insert
        pass

    def add_documents(
        self,
        collection: str,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Add documents to collection.

        Args:
            collection: Collection name
            documents: List of text documents
            metadata: Optional metadata for each document
            ids: Optional document IDs

        Returns:
            Operation result
        """
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()

        # Prepare data
        data = []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            item = {
                "id": ids[i] if ids else f"doc_{i}",
                "text": doc,
                "vector": emb
            }
            if metadata and i < len(metadata):
                item.update(metadata[i])
            data.append(item)

        # Insert into LanceDB
        table = self.db.create_table(
            collection,
            data=data,
            mode="overwrite" if not self._table_exists(collection) else "append"
        )

        return {
            "collection": collection,
            "documents_added": len(documents),
            "status": "success"
        }

    def search(
        self,
        collection: str,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents.

        Args:
            collection: Collection name
            query: Search query text
            top_k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of search results with documents and scores
        """
        if not self._table_exists(collection):
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()

        # Search in LanceDB
        table = self.db.open_table(collection)
        results = table.search(query_embedding).limit(top_k).to_list()

        return [
            {
                "id": r.get("id"),
                "text": r.get("text"),
                "score": r.get("_distance", 0),
                "metadata": {k: v for k, v in r.items()
                           if k not in ["id", "text", "vector", "_distance"]}
            }
            for r in results
        ]

    def _table_exists(self, name: str) -> bool:
        """Check if table exists.

        Args:
            name: Table name

        Returns:
            True if exists
        """
        try:
            self.db.open_table(name)
            return True
        except Exception:
            return False
