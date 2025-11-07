"""Artifact registry manager with signing and governance."""

import json
from typing import List, Optional
from datetime import datetime
import logging
from artifacts.schemas.base import (
    ArtifactManifest,
    ArtifactType,
    GovernanceStatus,
)
from modelops.platform.security.signing import SignatureManager

logger = logging.getLogger(__name__)


class ArtifactRegistry:
    """Central registry for model artifacts with full lineage tracking."""

    def __init__(self, db_url: Optional[str] = None):
        # Defer heavy imports to avoid startup-time dependency issues
        from modelops.storage.postgres_client import PostgresClient  # local import
        from modelops.storage.lancedb_client import LanceDBClient   # local import

        self.pg_client = PostgresClient(db_url)
        self.lance_client = LanceDBClient()
        self.signer = SignatureManager()
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS artifacts (
            id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            type VARCHAR(50) NOT NULL,
            manifest_json JSONB NOT NULL,
            governance_status VARCHAR(50) NOT NULL,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            tags TEXT[],
            CONSTRAINT unique_name_version UNIQUE (name, manifest_json->>'provenance')
        );
        
        CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(type);
        CREATE INDEX IF NOT EXISTS idx_artifacts_status ON artifacts(governance_status);
        CREATE INDEX IF NOT EXISTS idx_artifacts_created ON artifacts(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_artifacts_tags ON artifacts USING GIN(tags);
        CREATE INDEX IF NOT EXISTS idx_artifacts_manifest ON artifacts USING GIN(manifest_json);
        """
        
        try:
            self.pg_client.execute_query(schema_sql)
            logger.info("Artifact registry schema initialized")
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")

    def register(self, manifest: ArtifactManifest) -> str:
        """Register artifact with cryptographic signature."""
        # Sign manifest
        manifest_json = manifest.json()
        manifest.signature = self.signer.sign(manifest_json)
        
        # Insert to PostgreSQL
        insert_sql = """
        INSERT INTO artifacts (
            id, name, type, manifest_json, governance_status, created_at, tags
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            manifest_json = EXCLUDED.manifest_json,
            updated_at = CURRENT_TIMESTAMP
        """
        
        try:
            self.pg_client.execute_query(
                insert_sql,
                (
                    manifest.artifact_id,
                    manifest.name,
                    manifest.type.value,
                    manifest_json,
                    manifest.governance.status.value,
                    manifest.provenance.created_at,
                    manifest.tags,
                ),
            )
            
            # Index for semantic search in LanceDB
            self._index_for_search(manifest)
            
            logger.info(f"Registered artifact: {manifest.artifact_id}")
            return manifest.artifact_id
        except Exception as e:
            logger.error(f"Failed to register artifact: {e}")
            raise

    def _index_for_search(self, manifest: ArtifactManifest) -> None:
        """Index artifact for semantic search."""
        try:
            # Create searchable document
            search_doc = {
                "id": manifest.artifact_id,
                "text": f"{manifest.name} {manifest.description or ''} {' '.join(manifest.tags)}",
                "metadata": {
                    "type": manifest.type.value,
                    "status": manifest.governance.status.value,
                    "created_at": manifest.provenance.created_at.isoformat(),
                },
            }
            
            # Try to add to existing index
            tables = self.lance_client.list_tables()
            if "artifacts" in tables:
                self.lance_client.add_documents("artifacts", [search_doc])
            else:
                self.lance_client.create_index("artifacts", [search_doc])
                
        except Exception as e:
            logger.warning(f"Failed to index artifact for search: {e}")

    def get(self, artifact_id: str) -> Optional[ArtifactManifest]:
        """Get artifact by ID."""
        query_sql = "SELECT manifest_json FROM artifacts WHERE id = %s"
        
        try:
            results = self.pg_client.execute_query(query_sql, (artifact_id,))
            if results:
                return ArtifactManifest.parse_raw(results[0]["manifest_json"])
            return None
        except Exception as e:
            logger.error(f"Failed to get artifact {artifact_id}: {e}")
            raise

    def search(
        self,
        query: Optional[str] = None,
        artifact_type: Optional[ArtifactType] = None,
        status: Optional[GovernanceStatus] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[ArtifactManifest]:
        """Search artifacts with filters."""
        conditions = []
        params = []
        
        if artifact_type:
            conditions.append("type = %s")
            params.append(artifact_type.value)
        
        if status:
            conditions.append("governance_status = %s")
            params.append(status.value)
        
        if tags:
            conditions.append("tags && %s")
            params.append(tags)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        search_sql = f"""
            SELECT manifest_json 
            FROM artifacts 
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT %s
        """
        
        try:
            results = self.pg_client.execute_query(
                search_sql, tuple(params + [limit])
            )
            return [ArtifactManifest.parse_raw(r["manifest_json"]) for r in results]
        except Exception as e:
            logger.error(f"Failed to search artifacts: {e}")
            raise

    def promote(
        self,
        artifact_id: str,
        to_status: GovernanceStatus,
        promoter: str,
    ) -> None:
        """Promote artifact through governance stages."""
        manifest = self.get(artifact_id)
        if not manifest:
            raise ValueError(f"Artifact {artifact_id} not found")
        
        # Validation rules
        if to_status == GovernanceStatus.PROD:
            if manifest.governance.status != GovernanceStatus.STAGING:
                raise ValueError(
                    "Can only promote to PROD from STAGING"
                )
            if not all(manifest.governance.promotion_checklist.values()):
                raise ValueError(
                    "All checklist items must be completed before promotion"
                )
        
        # Update status
        manifest.governance.status = to_status
        manifest.governance.promoted_by = promoter
        manifest.governance.promoted_at = datetime.now()
        
        # Re-sign
        manifest.signature = self.signer.sign(manifest.json())
        
        # Update database
        update_sql = """
        UPDATE artifacts 
        SET manifest_json = %s, governance_status = %s, updated_at = CURRENT_TIMESTAMP
        WHERE id = %s
        """
        
        try:
            self.pg_client.execute_query(
                update_sql, (manifest.json(), to_status.value, artifact_id)
            )
            logger.info(f"Promoted {artifact_id} to {to_status.value}")
        except Exception as e:
            logger.error(f"Failed to promote artifact: {e}")
            raise

    def verify_signature(self, artifact_id: str) -> bool:
        """Verify artifact signature."""
        manifest = self.get(artifact_id)
        if not manifest:
            return False
        
        # Create manifest without signature
        manifest_copy = manifest.copy(deep=True)
        sig_info = manifest_copy.signature
        manifest_copy.signature = None
        
        manifest_json = manifest_copy.json()
        return self.signer.verify(manifest_json, sig_info)

    def get_lineage(self, artifact_id: str) -> dict:
        """Get full artifact lineage."""
        manifest = self.get(artifact_id)
        if not manifest:
            raise ValueError(f"Artifact {artifact_id} not found")
        
        lineage = {
            "artifact_id": artifact_id,
            "name": manifest.name,
            "type": manifest.type.value,
            "created_at": manifest.provenance.created_at.isoformat(),
            "created_by": manifest.provenance.created_by,
            "git_commit": manifest.provenance.git_commit,
            "parent_artifacts": [],
        }
        
        # Recursive parent lookup
        if manifest.provenance.parent_artifact_id:
            try:
                parent_lineage = self.get_lineage(
                    manifest.provenance.parent_artifact_id
                )
                lineage["parent_artifacts"].append(parent_lineage)
            except Exception as e:
                logger.warning(f"Could not retrieve parent lineage: {e}")
        
        return lineage

    def list_versions(self, name: str) -> List[ArtifactManifest]:
        """List all versions of an artifact."""
        query_sql = """
        SELECT manifest_json FROM artifacts 
        WHERE name = %s 
        ORDER BY created_at DESC
        """
        
        try:
            results = self.pg_client.execute_query(query_sql, (name,))
            return [ArtifactManifest.parse_raw(r["manifest_json"]) for r in results]
        except Exception as e:
            logger.error(f"Failed to list versions for {name}: {e}")
            raise

    def delete(self, artifact_id: str) -> None:
        """Delete artifact (use with caution)."""
        delete_sql = "DELETE FROM artifacts WHERE id = %s"
        
        try:
            self.pg_client.execute_query(delete_sql, (artifact_id,))
            logger.warning(f"Deleted artifact: {artifact_id}")
        except Exception as e:
            logger.error(f"Failed to delete artifact: {e}")
            raise
