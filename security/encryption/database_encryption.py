"""
Database Encryption for PostgreSQL.

Provides column-level encryption using AES-256 via pgcrypto extension.
"""

from typing import Optional, Any
import psycopg2
from psycopg2.extras import RealDictCursor
from .key_manager import KeyManager


class DatabaseEncryption:
    """PostgreSQL database encryption using pgcrypto."""

    def __init__(self, db_config: dict, key_manager: KeyManager):
        """
        Initialize database encryption.

        Args:
            db_config: PostgreSQL connection config
            key_manager: KeyManager instance for encryption keys
        """
        self.db_config = db_config
        self.key_manager = key_manager
        self.conn = None

    def connect(self):
        """Establish database connection."""
        self.conn = psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)

    def setup_encryption(self):
        """Set up pgcrypto extension and encryption functions."""
        if not self.conn:
            self.connect()

        with self.conn.cursor() as cur:
            # Enable pgcrypto extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

            # Create encryption keys table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS encryption_keys (
                    id SERIAL PRIMARY KEY,
                    key_name VARCHAR(100) UNIQUE NOT NULL,
                    encrypted_key BYTEA NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    rotated_at TIMESTAMP,
                    status VARCHAR(20) DEFAULT 'active'
                );
            """)

            # Create encrypt function
            cur.execute("""
                CREATE OR REPLACE FUNCTION encrypt_data(data TEXT, key_text TEXT)
                RETURNS BYTEA AS $$
                BEGIN
                    RETURN pgp_sym_encrypt(data, key_text);
                END;
                $$ LANGUAGE plpgsql SECURITY DEFINER;
            """)

            # Create decrypt function
            cur.execute("""
                CREATE OR REPLACE FUNCTION decrypt_data(encrypted_data BYTEA, key_text TEXT)
                RETURNS TEXT AS $$
                BEGIN
                    RETURN pgp_sym_decrypt(encrypted_data, key_text);
                END;
                $$ LANGUAGE plpgsql SECURITY DEFINER;
            """)

            self.conn.commit()
            print("✅ Database encryption setup complete")

    def encrypt_column(
        self, table: str, column: str, key_name: str = "database_encryption"
    ) -> None:
        """
        Encrypt an existing column in-place.

        Args:
            table: Table name
            column: Column name to encrypt
            key_name: Name of encryption key to use
        """
        if not self.conn:
            self.connect()

        # Get encryption key
        key = self.key_manager.get_key(key_name)
        key_hex = key.hex()

        with self.conn.cursor() as cur:
            # Add temporary encrypted column
            encrypted_col = f"{column}_encrypted"
            cur.execute(f"""
                ALTER TABLE {table}
                ADD COLUMN {encrypted_col} BYTEA;
            """)

            # Encrypt data and copy to new column
            cur.execute(f"""
                UPDATE {table}
                SET {encrypted_col} = encrypt_data({column}::TEXT, %s)
                WHERE {column} IS NOT NULL;
            """, (key_hex,))

            # Drop original column
            cur.execute(f"""
                ALTER TABLE {table}
                DROP COLUMN {column};
            """)

            # Rename encrypted column
            cur.execute(f"""
                ALTER TABLE {table}
                RENAME COLUMN {encrypted_col} TO {column};
            """)

            self.conn.commit()
            print(f"✅ Encrypted column: {table}.{column}")

    def decrypt_value(self, encrypted_value: bytes, key_name: str = "database_encryption") -> str:
        """
        Decrypt a single value.

        Args:
            encrypted_value: Encrypted bytes
            key_name: Name of encryption key

        Returns:
            Decrypted string
        """
        if not self.conn:
            self.connect()

        key = self.key_manager.get_key(key_name)
        key_hex = key.hex()

        with self.conn.cursor() as cur:
            cur.execute("SELECT decrypt_data(%s, %s) as decrypted;", (encrypted_value, key_hex))
            result = cur.fetchone()
            return result["decrypted"] if result else None

    def create_encrypted_table(self, table_name: str, schema: dict, key_name: str = "database_encryption"):
        """
        Create a new table with encrypted columns.

        Args:
            table_name: Name of the table
            schema: Dict of column_name -> (type, encrypted)
            key_name: Encryption key name

        Example:
            schema = {
                'id': ('UUID PRIMARY KEY', False),
                'username': ('VARCHAR(100)', False),
                'email': ('TEXT', True),  # Encrypted
                'api_key': ('TEXT', True),  # Encrypted
            }
        """
        if not self.conn:
            self.connect()

        columns = []
        for col_name, (col_type, encrypted) in schema.items():
            if encrypted:
                columns.append(f"{col_name} BYTEA")
            else:
                columns.append(f"{col_name} {col_type}")

        create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)});"

        with self.conn.cursor() as cur:
            cur.execute(create_sql)
            self.conn.commit()
            print(f"✅ Created encrypted table: {table_name}")

    def insert_encrypted(
        self,
        table: str,
        data: dict,
        encrypted_columns: list,
        key_name: str = "database_encryption"
    ) -> None:
        """
        Insert data with automatic encryption of specified columns.

        Args:
            table: Table name
            data: Dict of column_name -> value
            encrypted_columns: List of columns to encrypt
            key_name: Encryption key name
        """
        if not self.conn:
            self.connect()

        key = self.key_manager.get_key(key_name)
        key_hex = key.hex()

        # Build INSERT statement
        columns = list(data.keys())
        values = []
        placeholders = []

        for col in columns:
            if col in encrypted_columns:
                placeholders.append("encrypt_data(%s, %s)")
                values.extend([data[col], key_hex])
            else:
                placeholders.append("%s")
                values.append(data[col])

        insert_sql = f"""
            INSERT INTO {table} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)});
        """

        with self.conn.cursor() as cur:
            cur.execute(insert_sql, values)
            self.conn.commit()

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


# Example usage
if __name__ == "__main__":
    # Initialize components
    km = KeyManager()

    # Generate database encryption key
    try:
        db_key = km.generate_key("database_encryption", "aes-256")
    except ValueError:
        db_key = km.get_key("database_encryption")

    # Database config
    db_config = {
        "host": "localhost",
        "port": 5432,
        "database": "modelops",
        "user": "postgres",
        "password": "postgres",
    }

    # Initialize database encryption
    db_enc = DatabaseEncryption(db_config, km)

    # Setup encryption
    db_enc.setup_encryption()

    # Create encrypted table
    schema = {
        "user_id": ("UUID PRIMARY KEY", False),
        "username": ("VARCHAR(100) NOT NULL", False),
        "email": ("TEXT", True),  # Encrypted
        "api_key": ("TEXT", True),  # Encrypted
        "created_at": ("TIMESTAMP DEFAULT NOW()", False),
    }

    try:
        db_enc.create_encrypted_table("user_credentials", schema)
        print("✅ Database encryption example complete")
    except Exception as e:
        print(f"⚠️ Database encryption example failed: {e}")
    finally:
        db_enc.close()
