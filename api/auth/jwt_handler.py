"""JWT token handling."""
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import jwt
from passlib.context import CryptContext
from pydantic import BaseModel


class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str  # subject (user_id)
    exp: int  # expiration
    iat: int  # issued at
    role: str
    permissions: list[str]


class JWTHandler:
    """Handle JWT token creation and verification."""

    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7
    ):
        """Initialize JWT handler."""
        self.secret_key = secret_key or os.getenv(
            "JWT_SECRET_KEY",
            "your-secret-key-change-in-production"
        )
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        self.pwd_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto"
        )

    def create_access_token(
        self,
        user_id: str,
        role: str,
        permissions: list[str],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token."""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.access_token_expire_minutes
            )

        payload = {
            "sub": user_id,
            "exp": int(expire.timestamp()),
            "iat": int(datetime.utcnow().timestamp()),
            "role": role,
            "permissions": permissions,
            "type": "access"
        }

        return jwt.encode(
            payload,
            self.secret_key,
            algorithm=self.algorithm
        )

    def create_refresh_token(
        self,
        user_id: str,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT refresh token."""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                days=self.refresh_token_expire_days
            )

        payload = {
            "sub": user_id,
            "exp": int(expire.timestamp()),
            "iat": int(datetime.utcnow().timestamp()),
            "type": "refresh"
        }

        return jwt.encode(
            payload,
            self.secret_key,
            algorithm=self.algorithm
        )

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.JWTError:
            raise ValueError("Invalid token")

    def hash_password(self, password: str) -> str:
        """Hash a password."""
        return self.pwd_context.hash(password)

    def verify_password(
        self,
        plain_password: str,
        hashed_password: str
    ) -> bool:
        """Verify a password against hash."""
        return self.pwd_context.verify(plain_password, hashed_password)

    def get_user_from_token(self, token: str) -> Optional[str]:
        """Extract user ID from token."""
        try:
            payload = self.verify_token(token)
            return payload.get("sub")
        except ValueError:
            return None
