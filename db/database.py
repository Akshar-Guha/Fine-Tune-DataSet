"""Database connection management and session utilities."""

import os
from typing import Generator
import logging
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import Pool

from .models import Base

# Configure logging
logger = logging.getLogger(__name__)

# Get database URL from environment variable or use default
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:password@localhost:5432/modelops"
)

# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=False,  # Set to True to see SQL queries
    pool_pre_ping=True,  # Test connection before using
    pool_size=5,
    max_overflow=10
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Add event listeners to log connection issues
@event.listens_for(Pool, "checkout")
def ping_connection(dbapi_connection, connection_record, connection_proxy):
    """Ping database connection to ensure it's still active."""
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("SELECT 1")
    except Exception:
        # Disconnect and try reconnecting
        logger.warning("Database connection was stale, reconnecting")
        connection_proxy._pool.dispose()
        raise
    cursor.close()


def get_db() -> Generator[Session, None, None]:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database with all models."""
    logger.info("Creating database tables if they don't exist")
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialization complete")


# Function for testing only
def drop_all_tables() -> None:
    """Drop all tables (use with caution, testing only)."""
    logger.warning("Dropping all database tables - FOR TESTING ONLY!")
    Base.metadata.drop_all(bind=engine)
    logger.warning("All tables have been dropped")
