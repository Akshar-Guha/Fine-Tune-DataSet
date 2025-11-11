"""Data Lifecycle Management - cleanup, archival, retention policies."""

import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import json


class LifecycleManager:
    """Manages data lifecycle - cleanup, archival, retention."""

    def __init__(self, data_dir: str = "./data", archive_dir: str = "./archive"):
        self.data_dir = Path(data_dir)
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def cleanup_old_data(
        self, days_threshold: int = 30, dry_run: bool = True
    ) -> Dict:
        """Delete data older than threshold."""
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        deleted_count = 0
        deleted_size = 0

        for file_path in self.data_dir.rglob("*"):
            if file_path.is_file():
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mtime < cutoff_date:
                    size = file_path.stat().st_size
                    if not dry_run:
                        file_path.unlink()
                    deleted_count += 1
                    deleted_size += size

        return {
            "deleted_count": deleted_count,
            "deleted_size_mb": round(deleted_size / (1024 * 1024), 2),
            "dry_run": dry_run,
        }

    def archive_data(self, source_pattern: str, days_old: int = 30) -> Dict:
        """Archive old data to archive directory."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        archived_count = 0

        for file_path in self.data_dir.rglob(source_pattern):
            if file_path.is_file():
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mtime < cutoff_date:
                    # Create archive path
                    rel_path = file_path.relative_to(self.data_dir)
                    archive_path = self.archive_dir / rel_path
                    archive_path.parent.mkdir(parents=True, exist_ok=True)

                    # Move to archive
                    shutil.move(str(file_path), str(archive_path))
                    archived_count += 1

        return {"archived_count": archived_count, "archive_dir": str(self.archive_dir)}


# Example usage
if __name__ == "__main__":
    lm = LifecycleManager()
    result = lm.cleanup_old_data(days_threshold=30, dry_run=True)
    print(f"Cleanup: {result}")
