#!/usr/bin/env python3
"""Test database connectivity and data persistence."""

import sys
import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, continue without it

# Ensure stdlib has priority over project packages (avoid 'platform' shadowing)
project_root = Path(__file__).parent
project_parent = project_root.parent
try:
    import sysconfig
    stdlib_path = sysconfig.get_paths().get("stdlib")
    if stdlib_path:
        if stdlib_path in sys.path:
            sys.path.remove(stdlib_path)
        sys.path.insert(0, stdlib_path)
except Exception:
    pass

# Add project parent first so absolute imports like 'modelops.*' resolve
if str(project_parent) not in sys.path:
    sys.path.insert(1, str(project_parent))

# Add project root at the end to avoid shadowing stdlib modules
if str(project_root) in sys.path:
    try:
        sys.path.remove(str(project_root))
    except ValueError:
        pass
sys.path.append(str(project_root))

from db.database import SessionLocal, engine
from db.models import LLM, LLMVersion, Dataset, Job
from datetime import datetime

def test_database_connectivity():
    """Test basic database connectivity."""
    print("Testing database connectivity...")

    # Test connection
    try:
        with engine.connect() as conn:
            result = conn.execute("SELECT 1 as test")
            row = result.fetchone()
            assert row[0] == 1
        print("✓ Database connection successful")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

    return True

def test_data_persistence():
    """Test data creation and persistence."""
    print("\nTesting data persistence...")

    # Create test data
    db = SessionLocal()
    try:
        # Create a test LLM
        test_llm = LLM(
            name="Test Model",
            description="A test LLM model",
            base_model="test-base-model"
        )
        db.add(test_llm)
        db.commit()
        db.refresh(test_llm)

        print(f"✓ Created test LLM with ID: {test_llm.id}")

        # Create a test LLM version
        test_version = LLMVersion(
            llm_id=test_llm.id,
            version="1.0.0",
            status="completed",
            created_by="test-user"
        )
        db.add(test_version)
        db.commit()
        db.refresh(test_version)

        print(f"✓ Created test LLM version with ID: {test_version.id}")

        # Create a test dataset
        test_dataset = Dataset(
            name="Test Dataset",
            description="A test dataset",
            source="test-source"
        )
        db.add(test_dataset)
        db.commit()
        db.refresh(test_dataset)

        print(f"✓ Created test dataset with ID: {test_dataset.id}")

        # Create a test job
        test_job = Job(
            name="Test Job",
            job_type="evaluation",
            status="completed",
            config={"test": "config"},
            dataset_id="test-dataset-id",
            base_model="test-model"
        )
        db.add(test_job)
        db.commit()
        db.refresh(test_job)

        print(f"✓ Created test job with ID: {test_job.id}")

        # Query the data back
        llms = db.query(LLM).all()
        versions = db.query(LLMVersion).all()
        datasets = db.query(Dataset).all()
        jobs = db.query(Job).all()

        print(f"✓ Queried back {len(llms)} LLMs, {len(versions)} versions, {len(datasets)} datasets, {len(jobs)} jobs")

        # Verify data integrity
        assert len(llms) >= 1
        assert len(versions) >= 1
        assert len(datasets) >= 1
        assert len(jobs) >= 1

        assert llms[0].name == "Test Model"
        assert versions[0].version == "1.0.0"
        assert datasets[0].name == "Test Dataset"
        assert jobs[0].name == "Test Job"

        print("✓ Data persistence and integrity verified")

        # Clean up test data
        db.delete(test_job)
        db.delete(test_dataset)
        db.delete(test_version)
        db.delete(test_llm)
        db.commit()

        print("✓ Test data cleaned up")

    except Exception as e:
        print(f"✗ Data persistence test failed: {e}")
        db.rollback()
        return False
    finally:
        db.close()

    return True

def main():
    """Run all database tests."""
    print("=" * 60)
    print("ModelOps Database Connectivity & Persistence Tests")
    print("=" * 60)

    success = True

    success &= test_database_connectivity()
    success &= test_data_persistence()

    print("\n" + "=" * 60)
    if success:
        print("🎉 All database tests passed!")
    else:
        print("❌ Some database tests failed!")
    print("=" * 60)

    return success

if __name__ == '__main__':
    main()
