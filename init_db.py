#!/usr/bin/env python3
"""Initialize database tables."""

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

from db.database import engine
from db.models import Base

def main():
    print("Creating database tables...")
    print(f"Using database URL: {os.getenv('DATABASE_URL')}")
    print(f"Engine: {engine}")

    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

    # List all tables that were created
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"Created tables: {tables}")

if __name__ == '__main__':
    main()
