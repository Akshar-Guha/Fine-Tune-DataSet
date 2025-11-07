"""PostgreSQL client for metadata storage."""

import os
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()


class PostgresClient:
    """PostgreSQL client for job metadata, artifact registry, and lineage."""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:password@localhost:5432/modelops"
        )
        
        # SQLAlchemy engine
        self.engine = create_engine(self.database_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        
        logger.info("PostgreSQL client initialized")

    @contextmanager
    def get_session(self) -> Session:
        """Get database session context manager."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def init_db(self) -> None:
        """Initialize database schema."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database schema initialized")

    def execute_query(
        self, query: str, params: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """Execute raw SQL query."""
        conn = psycopg2.connect(self.database_url)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cursor.execute(query, params)
            if cursor.description:
                results = cursor.fetchall()
                return [dict(row) for row in results]
            conn.commit()
            return []
        except Exception as e:
            conn.rollback()
            logger.error(f"Query execution error: {e}")
            raise
        finally:
            cursor.close()
            conn.close()

    def execute_many(self, query: str, params_list: List[tuple]) -> None:
        """Execute query with multiple parameter sets."""
        conn = psycopg2.connect(self.database_url)
        cursor = conn.cursor()
        
        try:
            cursor.executemany(query, params_list)
            conn.commit()
            logger.debug(f"Executed batch query with {len(params_list)} rows")
        except Exception as e:
            conn.rollback()
            logger.error(f"Batch execution error: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
