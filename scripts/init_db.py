#!/usr/bin/env python3
"""Initialize database schema for ModelOps."""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from storage.postgres_client import PostgresClient
from artifacts.registry.manager import ArtifactRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_database():
    """Initialize all database schemas."""
    logger.info("Initializing ModelOps database...")
    
    # Initialize PostgreSQL connection
    pg_client = PostgresClient()
    
    # Create MLflow database
    logger.info("Creating MLflow database...")
    try:
        pg_client.execute_query("""
            CREATE DATABASE mlflow;
        """)
        logger.info("MLflow database created")
    except Exception as e:
        logger.warning(f"MLflow database might already exist: {e}")
    
    # Initialize artifact registry schema
    logger.info("Initializing artifact registry...")
    registry = ArtifactRegistry()
    logger.info("Artifact registry initialized")
    
    # Create additional tables
    logger.info("Creating additional tables...")
    
    # Jobs table
    pg_client.execute_query("""
        CREATE TABLE IF NOT EXISTS jobs (
            id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            type VARCHAR(50) NOT NULL,
            status VARCHAR(50) NOT NULL,
            config JSONB,
            workflow_id VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            error TEXT,
            metrics JSONB
        );
        
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
        CREATE INDEX IF NOT EXISTS idx_jobs_type ON jobs(type);
        CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
    """)
    
    # Datasets table
    pg_client.execute_query("""
        CREATE TABLE IF NOT EXISTS datasets (
            id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) UNIQUE NOT NULL,
            description TEXT,
            delta_uri VARCHAR(500),
            version INTEGER DEFAULT 1,
            num_rows BIGINT,
            num_columns INTEGER,
            schema JSONB,
            tags TEXT[],
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name);
        CREATE INDEX IF NOT EXISTS idx_datasets_tags ON datasets USING GIN(tags);
    """)
    
    # Deployments table
    pg_client.execute_query("""
        CREATE TABLE IF NOT EXISTS deployments (
            id VARCHAR(255) PRIMARY KEY,
            artifact_id VARCHAR(255) REFERENCES artifacts(id),
            backend VARCHAR(50) NOT NULL,
            status VARCHAR(50) NOT NULL,
            endpoint VARCHAR(500),
            replicas INTEGER DEFAULT 1,
            config JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_deployments_artifact ON deployments(artifact_id);
        CREATE INDEX IF NOT EXISTS idx_deployments_status ON deployments(status);
    """)
    
    logger.info("✅ Database initialization complete!")
    logger.info("\n📊 Access services at:")
    logger.info("  - API: http://localhost:8000")
    logger.info("  - Temporal UI: http://localhost:8088")
    logger.info("  - MLflow: http://localhost:5000")
    logger.info("  - Grafana: http://localhost:3000 (admin/admin)")
    logger.info("  - Prometheus: http://localhost:9090")
    logger.info("  - Jaeger: http://localhost:16686")
    logger.info("  - MinIO Console: http://localhost:9001 (minioadmin/minioadmin)")


if __name__ == "__main__":
    init_database()
