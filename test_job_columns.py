"""
Simple test script to verify job CRUD operations with new columns.
This bypasses the platform shadowing issue.
"""

import sys
import os

# Fix platform shadowing issue
_cwd = os.getcwd()
_modelops_paths = [
    p for p in sys.path
    if 'modelops' in p.lower() or p == _cwd or p == ''
]
for p in _modelops_paths:
    if p in sys.path:
        sys.path.remove(p)

# Add the correct path
sys.path.insert(0, 'S:\\projects\\Fine Tunning\\modelops')

# Restore modelops paths for our imports
sys.path.extend(_modelops_paths)

print("Testing job operations with new columns...")

try:
    # Test imports
    from db.database import get_db
    from db.repository import JobRepository
    from db.models import JobStatus, JobType

    print("✓ Imports successful")

    # Get database session
    db = next(get_db())
    repo = JobRepository(db)

    print("✓ Database connection successful")

    # Test job creation with new fields
    job_data = {
        "name": "Test Job with New Columns",
        "job_type": JobType.SFT_TRAINING,
        "config": {"learning_rate": 0.001, "epochs": 10},
        "dataset_id": "test-dataset-123",
        "base_model": "gpt-2",
        "priority": 5,
        "workflow_id": "workflow-456"
    }

    print("Creating job with new fields...")
    created_job = repo.create_job(job_data)
    print(f"✓ Job created: {created_job.id}")
    print(f"  - dataset_id: {created_job.dataset_id}")
    print(f"  - base_model: {created_job.base_model}")
    print(f"  - priority: {created_job.priority}")
    print(f"  - workflow_id: {created_job.workflow_id}")
    print(f"  - artifacts: {created_job.artifacts}")
    print(f"  - updated_at: {created_job.updated_at}")

    # Test job retrieval
    retrieved_job = repo.get_by_id(created_job.id)
    assert retrieved_job is not None
    print("✓ Job retrieval successful")

    # Test job listing
    jobs, total = repo.list_jobs()
    assert total >= 1
    print(f"✓ Job listing successful, found {total} jobs")

    # Test job update
    update_data = {"priority": 10, "base_model": "gpt-3"}
    updated_job = repo.update_job(created_job.id, update_data)
    assert updated_job.priority == 10
    assert updated_job.base_model == "gpt-3"
    print("✓ Job update successful")

    # Test job cancellation
    cancelled_job = repo.cancel_job(created_job.id)
    assert cancelled_job.status == JobStatus.CANCELLED
    assert cancelled_job.completed_at is not None
    print("✓ Job cancellation successful")

    print("\n🎉 All job operations with new columns work correctly!")

except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
