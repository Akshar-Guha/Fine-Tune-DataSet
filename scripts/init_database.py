#!/usr/bin/env python3
"""
Initialize the complete ModelOps database schema.
This script creates all required tables for storing LLM, dataset, metrics and logging data.
"""

import os
import sys
from pathlib import Path
import logging
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import init_db, drop_all_tables
from db.models import (
    User, LLM, LLMVersion, Dataset, DatasetVersion,
    TrainingParameter, TrainingLog, InferenceLog, 
    Tag, AuditLog, SystemMetric, Alert, APIKey
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_data(init_admin_user=True):
    """Create sample data for development and testing."""
    from db.database import SessionLocal
    from datetime import datetime
    import uuid
    import bcrypt
    
    logger.info("Creating sample data...")
    db = SessionLocal()
    try:
        # Create admin user
        if init_admin_user:
            salt = bcrypt.gensalt()
            hashed_password = bcrypt.hashpw("admin123".encode(), salt).decode()
            admin_user = User(
                username="admin",
                email="admin@example.com",
                password_hash=hashed_password,
                first_name="Admin",
                last_name="User",
                is_admin=True
            )
            db.add(admin_user)
        
        # Create sample LLM model
        sample_llm = LLM(
            name="GPT-4-Turbo-Fine-Tuned",
            description="GPT-4 Turbo model fine-tuned for specialized tasks",
            base_model="gpt-4-turbo"
        )
        db.add(sample_llm)
        
        # Create sample dataset
        sample_dataset = Dataset(
            name="Customer Support Conversations",
            description="Dataset of customer support conversations for fine-tuning",
            source="internal_crm"
        )
        db.add(sample_dataset)
        
        # Create some tags
        tags = [
            Tag(name="production"),
            Tag(name="development"),
            Tag(name="test"),
            Tag(name="customer-service"),
            Tag(name="finance"),
            Tag(name="healthcare")
        ]
        for tag in tags:
            db.add(tag)
        
        # Create sample training parameters
        parameters = [
            TrainingParameter(
                name="learning_rate",
                description="Learning rate for optimizer",
                type="float",
                default_value="5e-5",
                min_value="1e-6",
                max_value="1e-2"
            ),
            TrainingParameter(
                name="batch_size",
                description="Batch size for training",
                type="int",
                default_value="8",
                min_value="1",
                max_value="128"
            ),
            TrainingParameter(
                name="num_epochs",
                description="Number of training epochs",
                type="int",
                default_value="3",
                min_value="1",
                max_value="50"
            )
        ]
        for param in parameters:
            db.add(param)
        
        db.commit()
        logger.info("Sample data created successfully")
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating sample data: {e}")
        raise
    finally:
        db.close()


def main():
    """Initialize database schema."""
    parser = argparse.ArgumentParser(description="Initialize ModelOps database")
    parser.add_argument("--reset", action="store_true", 
                        help="Reset database (drop all tables first)")
    parser.add_argument("--sample-data", action="store_true",
                        help="Create sample data for development")
    args = parser.parse_args()
    
    try:
        if args.reset:
            logger.warning("Resetting database (dropping all tables)")
            drop_all_tables()
        
        logger.info("Initializing ModelOps database schema...")
        init_db()
        
        if args.sample_data:
            create_sample_data()
        
        logger.info("✅ Database initialization complete!")
        logger.info("\n📊 Database Schema Created For:")
        logger.info("  - LLM models and versions")
        logger.info("  - Training and inference datasets")
        logger.info("  - Training logs and metrics")
        logger.info("  - Inference logs and performance metrics")
        logger.info("  - System monitoring and alerts")
        logger.info("  - User authentication and API keys")
        logger.info("  - Audit logging and security tracking")
    
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
