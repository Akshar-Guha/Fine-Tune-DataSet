"""Artifact signing with Ed25519."""

import os
from datetime import datetime
from typing import Optional
import base64
import logging
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.backends import default_backend
from artifacts.schemas.base import SignatureInfo

logger = logging.getLogger(__name__)


class SignatureManager:
    """Cryptographic signing for artifact manifests."""

    def __init__(self, private_key_path: Optional[str] = None):
        """Initialize with private key or generate new keypair."""
        if private_key_path and os.path.exists(private_key_path):
            try:
                with open(private_key_path, "rb") as f:
                    self.private_key = serialization.load_pem_private_key(
                        f.read(),
                        password=None,
                        backend=default_backend()
                    )
                logger.info(f"Loaded private key from {private_key_path}")
            except Exception as e:
                logger.warning(f"Failed to load private key: {e}")
                self.private_key = ed25519.Ed25519PrivateKey.generate()
        else:
            # Generate new keypair
            self.private_key = ed25519.Ed25519PrivateKey.generate()
            logger.info("Generated new Ed25519 keypair")
        
        self.public_key = self.private_key.public_key()

    def sign(self, data: str) -> SignatureInfo:
        """Sign data and return signature info."""
        try:
            signature = self.private_key.sign(data.encode())
            
            public_key_bytes = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            return SignatureInfo(
                algorithm="ed25519",
                public_key=base64.b64encode(public_key_bytes).decode(),
                signature=base64.b64encode(signature).decode(),
                signed_at=datetime.now()
            )
        except Exception as e:
            logger.error(f"Failed to sign data: {e}")
            raise

    def verify(self, data: str, signature_info: SignatureInfo) -> bool:
        """Verify signature."""
        try:
            public_key_bytes = base64.b64decode(signature_info.public_key)
            public_key = serialization.load_pem_public_key(
                public_key_bytes,
                backend=default_backend()
            )
            
            signature = base64.b64decode(signature_info.signature)
            public_key.verify(signature, data.encode())
            return True
        except Exception as e:
            logger.warning(f"Signature verification failed: {e}")
            return False

    def save_private_key(self, path: str) -> None:
        """Save private key to file."""
        private_pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(private_pem)
        
        logger.info(f"Saved private key to {path}")

    def save_public_key(self, path: str) -> None:
        """Save public key to file."""
        public_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(public_pem)
        
        logger.info(f"Saved public key to {path}")
