"""
Session Management using Redis.

Handles user sessions, device tracking, and session expiration.
"""

import json
import redis
from typing import Optional, Dict
from datetime import datetime, timedelta
import secrets
import hashlib


class SessionManager:
    """Manages user sessions with Redis."""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        session_ttl: int = 3600,
    ):
        """
        Initialize Session Manager.

        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            session_ttl: Session TTL in seconds (default: 1 hour)
        """
        self.redis_client = redis.Redis(
            host=redis_host, port=redis_port, db=redis_db, decode_responses=True
        )
        self.session_ttl = session_ttl

    def create_session(
        self, user_id: str, user_data: Dict, device_info: Optional[Dict] = None
    ) -> str:
        """
        Create a new session for a user.

        Args:
            user_id: User ID
            user_data: User information
            device_info: Device fingerprint info

        Returns:
            Session token
        """
        # Generate secure session token
        session_token = secrets.token_urlsafe(32)

        # Create session data
        session_data = {
            "user_id": user_id,
            "user_data": user_data,
            "device_info": device_info or {},
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
        }

        # Store in Redis with TTL
        session_key = f"session:{session_token}"
        self.redis_client.setex(
            session_key, self.session_ttl, json.dumps(session_data, default=str)
        )

        # Track active sessions for user
        user_sessions_key = f"user_sessions:{user_id}"
        self.redis_client.sadd(user_sessions_key, session_token)
        self.redis_client.expire(user_sessions_key, self.session_ttl * 2)

        print(f"✅ Created session for user {user_id}")
        return session_token

    def get_session(self, session_token: str) -> Optional[Dict]:
        """
        Retrieve session data.

        Args:
            session_token: Session token

        Returns:
            Session data or None if not found
        """
        session_key = f"session:{session_token}"
        session_data = self.redis_client.get(session_key)

        if session_data:
            data = json.loads(session_data)
            # Update last activity
            data["last_activity"] = datetime.utcnow().isoformat()
            self.redis_client.setex(
                session_key, self.session_ttl, json.dumps(data, default=str)
            )
            return data

        return None

    def validate_session(self, session_token: str) -> bool:
        """
        Validate if session exists and is active.

        Args:
            session_token: Session token

        Returns:
            True if valid, False otherwise
        """
        session_data = self.get_session(session_token)
        return session_data is not None

    def refresh_session(self, session_token: str) -> bool:
        """
        Refresh session TTL.

        Args:
            session_token: Session token

        Returns:
            True if refreshed, False if session not found
        """
        session_key = f"session:{session_token}"

        if self.redis_client.exists(session_key):
            self.redis_client.expire(session_key, self.session_ttl)
            return True

        return False

    def delete_session(self, session_token: str) -> bool:
        """
        Delete a session (logout).

        Args:
            session_token: Session token

        Returns:
            True if deleted, False if not found
        """
        session_key = f"session:{session_token}"

        # Get session data to find user_id
        session_data = self.redis_client.get(session_key)
        if session_data:
            data = json.loads(session_data)
            user_id = data.get("user_id")

            # Remove from user's active sessions
            if user_id:
                user_sessions_key = f"user_sessions:{user_id}"
                self.redis_client.srem(user_sessions_key, session_token)

            # Delete session
            self.redis_client.delete(session_key)
            print(f"✅ Deleted session for user {user_id}")
            return True

        return False

    def delete_all_user_sessions(self, user_id: str) -> int:
        """
        Delete all sessions for a user (logout from all devices).

        Args:
            user_id: User ID

        Returns:
            Number of sessions deleted
        """
        user_sessions_key = f"user_sessions:{user_id}"
        session_tokens = self.redis_client.smembers(user_sessions_key)

        count = 0
        for token in session_tokens:
            session_key = f"session:{token}"
            if self.redis_client.delete(session_key):
                count += 1

        # Clear user sessions set
        self.redis_client.delete(user_sessions_key)

        print(f"✅ Deleted {count} sessions for user {user_id}")
        return count

    def get_user_sessions(self, user_id: str) -> list:
        """
        Get all active sessions for a user.

        Args:
            user_id: User ID

        Returns:
            List of session data
        """
        user_sessions_key = f"user_sessions:{user_id}"
        session_tokens = self.redis_client.smembers(user_sessions_key)

        sessions = []
        for token in session_tokens:
            session_data = self.get_session(token)
            if session_data:
                sessions.append({"token": token, **session_data})

        return sessions

    def check_device_fingerprint(self, session_token: str, current_device: Dict) -> bool:
        """
        Check if device matches session device.

        Args:
            session_token: Session token
            current_device: Current device info

        Returns:
            True if device matches, False otherwise
        """
        session_data = self.get_session(session_token)

        if not session_data:
            return False

        stored_device = session_data.get("device_info", {})

        # Simple fingerprint comparison
        device_hash = self._hash_device(stored_device)
        current_hash = self._hash_device(current_device)

        return device_hash == current_hash

    def _hash_device(self, device_info: Dict) -> str:
        """
        Create hash of device info.

        Args:
            device_info: Device information

        Returns:
            Device hash
        """
        device_str = json.dumps(device_info, sort_keys=True)
        return hashlib.sha256(device_str.encode()).hexdigest()


# Example usage
if __name__ == "__main__":
    # Initialize session manager
    sm = SessionManager(session_ttl=3600)

    # Create session
    user_data = {"username": "john_doe", "email": "john@example.com", "role": "data_scientist"}

    device_info = {
        "user_agent": "Mozilla/5.0...",
        "ip_address": "192.168.1.1",
        "platform": "Windows",
    }

    session_token = sm.create_session(
        user_id="user_123", user_data=user_data, device_info=device_info
    )

    print(f"Session Token: {session_token}")

    # Validate session
    is_valid = sm.validate_session(session_token)
    print(f"Session Valid: {is_valid}")

    # Get session data
    session = sm.get_session(session_token)
    print(f"Session Data: {json.dumps(session, indent=2)}")

    # Get user sessions
    user_sessions = sm.get_user_sessions("user_123")
    print(f"User Sessions: {len(user_sessions)}")

    # Device fingerprint check
    device_match = sm.check_device_fingerprint(session_token, device_info)
    print(f"Device Match: {device_match}")

    # Delete session
    deleted = sm.delete_session(session_token)
    print(f"Session Deleted: {deleted}")

    print("\n✅ Session management test complete")
