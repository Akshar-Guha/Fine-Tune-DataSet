"""
Key Management System for ModelOps.

Handles encryption key generation, storage, rotation, and retrieval.
Uses AES-256 for symmetric encryption with secure key derivation.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64


class KeyManager:
    """Manages encryption keys with rotation and secure storage."""
    
    def __init__(self, key_store_path: Optional[str] = None):
        """
        Initialize Key Manager.
        
        Args:
            key_store_path: Path to store encryption keys (default: ./keys)
        """
        self.key_store_path = Path(key_store_path or "./keys")
        self.key_store_path.mkdir(parents=True, exist_ok=True)
        self.master_key_file = self.key_store_path / "master.key"
        self.keys_metadata_file = self.key_store_path / "keys_metadata.json"
        
        # Initialize master key
        self._initialize_master_key()
        
        # Load keys metadata
        self.keys_metadata = self._load_keys_metadata()
    
    def _initialize_master_key(self):
        """Initialize or load the master encryption key."""
        if not self.master_key_file.exists():
            # Generate new master key
            master_key = Fernet.generate_key()
            with open(self.master_key_file, "wb") as f:
                f.write(master_key)
            # Secure the file (read-only for owner)
            os.chmod(self.master_key_file, 0o400)
            print(f"✅ Generated new master key: {self.master_key_file}")
        
        # Load master key
        with open(self.master_key_file, "rb") as f:
            self.master_key = f.read()
        
        self.fernet = Fernet(self.master_key)
    
    def _load_keys_metadata(self) -> Dict:
        """Load keys metadata from storage."""
        if self.keys_metadata_file.exists():
            with open(self.keys_metadata_file, "r") as f:
                return json.load(f)
        return {}
    
    def _save_keys_metadata(self):
        """Save keys metadata to storage."""
        with open(self.keys_metadata_file, "w") as f:
            json.dump(self.keys_metadata, f, indent=2, default=str)
    
    def generate_key(self, key_name: str, key_type: str = "aes-256") -> bytes:
        """
        Generate a new encryption key.
        
        Args:
            key_name: Unique name for the key
            key_type: Type of key (aes-256, fernet)
            
        Returns:
            Generated encryption key
        """
        if key_name in self.keys_metadata:
            raise ValueError(f"Key '{key_name}' already exists")
        
        # Generate key based on type
        if key_type == "fernet":
            key = Fernet.generate_key()
        else:  # aes-256
            key = os.urandom(32)  # 256 bits
        
        # Encrypt key with master key
        encrypted_key = self.fernet.encrypt(key)
        
        # Store encrypted key
        key_file = self.key_store_path / f"{key_name}.key"
        with open(key_file, "wb") as f:
            f.write(encrypted_key)
        os.chmod(key_file, 0o400)
        
        # Update metadata
        self.keys_metadata[key_name] = {
            "type": key_type,
            "created_at": datetime.utcnow().isoformat(),
            "status": "active",
            "file": str(key_file)
        }
        self._save_keys_metadata()
        
        print(f"✅ Generated key: {key_name} ({key_type})")
        return key
    
    def get_key(self, key_name: str) -> bytes:
        """
        Retrieve an encryption key.
        
        Args:
            key_name: Name of the key to retrieve
            
        Returns:
            Decrypted encryption key
        """
        if key_name not in self.keys_metadata:
            raise ValueError(f"Key '{key_name}' not found")
        
        metadata = self.keys_metadata[key_name]
        
        if metadata["status"] != "active":
            raise ValueError(f"Key '{key_name}' is not active")
        
        # Read encrypted key
        with open(metadata["file"], "rb") as f:
            encrypted_key = f.read()
        
        # Decrypt with master key
        key = self.fernet.decrypt(encrypted_key)
        return key
    
    def rotate_key(self, key_name: str) -> bytes:
        """
        Rotate an encryption key (generate new key, mark old as rotated).
        
        Args:
            key_name: Name of the key to rotate
            
        Returns:
            New encryption key
        """
        if key_name not in self.keys_metadata:
            raise ValueError(f"Key '{key_name}' not found")
        
        # Mark old key as rotated
        old_metadata = self.keys_metadata[key_name]
        old_metadata["status"] = "rotated"
        old_metadata["rotated_at"] = datetime.utcnow().isoformat()
        
        # Generate new key
        new_key_name = f"{key_name}_v{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        new_key = self.generate_key(new_key_name, old_metadata["type"])
        
        print(f"✅ Rotated key: {key_name} -> {new_key_name}")
        return new_key
    
    def derive_key(self, password: str, salt: Optional[bytes] = None) -> tuple[bytes, bytes]:
        """
        Derive an encryption key from a password using PBKDF2HMAC.

        Args:
            password: Password to derive key from
            salt: Salt for key derivation (generated if not provided)

        Returns:
            Tuple of (derived_key, salt)
        """
        if salt is None:
            salt = os.urandom(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = kdf.derive(password.encode())
        return key, salt
    
    def encrypt_data(self, data: bytes, key: bytes) -> bytes:
        """
        Encrypt data using AES-256.
        
        Args:
            data: Data to encrypt
            key: Encryption key
            
        Returns:
            Encrypted data
        """
        f = Fernet(base64.urlsafe_b64encode(key))
        return f.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes, key: bytes) -> bytes:
        """
        Decrypt data using AES-256.
        
        Args:
            encrypted_data: Data to decrypt
            key: Decryption key
            
        Returns:
            Decrypted data
        """
        f = Fernet(base64.urlsafe_b64encode(key))
        return f.decrypt(encrypted_data)
    
    def list_keys(self) -> Dict:
        """List all encryption keys and their metadata."""
        return self.keys_metadata
    
    def delete_key(self, key_name: str, force: bool = False):
        """
        Delete an encryption key (soft delete by default).
        
        Args:
            key_name: Name of the key to delete
            force: If True, permanently delete the key file
        """
        if key_name not in self.keys_metadata:
            raise ValueError(f"Key '{key_name}' not found")
        
        metadata = self.keys_metadata[key_name]
        
        if force:
            # Permanently delete key file
            key_file = Path(metadata["file"])
            if key_file.exists():
                key_file.unlink()
            del self.keys_metadata[key_name]
            print(f"🗑️ Permanently deleted key: {key_name}")
        else:
            # Soft delete (mark as deleted)
            metadata["status"] = "deleted"
            metadata["deleted_at"] = datetime.utcnow().isoformat()
            print(f"🗑️ Soft deleted key: {key_name}")
        
        self._save_keys_metadata()


# Example usage
if __name__ == "__main__":
    # Initialize key manager
    km = KeyManager()
    
    # Generate database encryption key
    db_key = km.generate_key("database_encryption", "aes-256")
    print(f"Database key: {db_key.hex()[:32]}...")
    
    # Generate MinIO encryption key
    minio_key = km.generate_key("minio_encryption", "aes-256")
    print(f"MinIO key: {minio_key.hex()[:32]}...")
    
    # Test encryption/decryption
    data = b"Sensitive data to encrypt"
    encrypted = km.encrypt_data(data, db_key)
    decrypted = km.decrypt_data(encrypted, db_key)
    assert data == decrypted
    print("✅ Encryption/Decryption test passed")
    
    # List all keys
    print("\n📋 All encryption keys:")
    for name, metadata in km.list_keys().items():
        print(f"  - {name}: {metadata['status']} ({metadata['type']})")
