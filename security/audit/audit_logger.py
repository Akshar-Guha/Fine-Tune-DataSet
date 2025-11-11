"""
Comprehensive Audit Logging System.

Logs security events, data access, model changes for compliance.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum


class AuditEventType(Enum):
    """Types of audit events."""

    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"
    PASSWORD_CHANGE = "password_change"

    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHANGE = "permission_change"

    # Data events
    DATA_READ = "data_read"
    DATA_WRITE = "data_write"
    DATA_DELETE = "data_delete"
    DATA_EXPORT = "data_export"

    # Model events
    MODEL_TRAIN = "model_train"
    MODEL_DEPLOY = "model_deploy"
    MODEL_DELETE = "model_delete"
    MODEL_UPDATE = "model_update"

    # System events
    CONFIG_CHANGE = "config_change"
    KEY_ROTATION = "key_rotation"
    BACKUP_CREATED = "backup_created"
    SYSTEM_ERROR = "system_error"


class AuditLogger:
    """Structured audit logging for compliance and security."""

    def __init__(self, log_dir: str = "./logs/audit"):
        """
        Initialize Audit Logger.

        Args:
            log_dir: Directory for audit logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set up separate loggers for different event types
        self.loggers = {}
        self._setup_loggers()

    def _setup_loggers(self):
        """Set up specialized loggers for different audit categories."""
        categories = {
            "security": [
                AuditEventType.LOGIN_SUCCESS,
                AuditEventType.LOGIN_FAILURE,
                AuditEventType.LOGOUT,
                AuditEventType.MFA_ENABLED,
                AuditEventType.MFA_DISABLED,
                AuditEventType.PASSWORD_CHANGE,
            ],
            "access": [
                AuditEventType.ACCESS_GRANTED,
                AuditEventType.ACCESS_DENIED,
                AuditEventType.PERMISSION_CHANGE,
            ],
            "data": [
                AuditEventType.DATA_READ,
                AuditEventType.DATA_WRITE,
                AuditEventType.DATA_DELETE,
                AuditEventType.DATA_EXPORT,
            ],
            "model": [
                AuditEventType.MODEL_TRAIN,
                AuditEventType.MODEL_DEPLOY,
                AuditEventType.MODEL_DELETE,
                AuditEventType.MODEL_UPDATE,
            ],
            "system": [
                AuditEventType.CONFIG_CHANGE,
                AuditEventType.KEY_ROTATION,
                AuditEventType.BACKUP_CREATED,
                AuditEventType.SYSTEM_ERROR,
            ],
        }

        for category, event_types in categories.items():
            logger = logging.getLogger(f"audit.{category}")
            logger.setLevel(logging.INFO)

            # Create file handler
            log_file = self.log_dir / f"{category}.log"
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)

            # JSON formatter
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"category": "' + category + '", "message": %(message)s}'
            )
            handler.setFormatter(formatter)

            logger.addHandler(handler)
            self.loggers[category] = logger

            # Map event types to logger
            for event_type in event_types:
                self.loggers[event_type] = logger

    def log_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        status: str = "success",
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
    ):
        """
        Log an audit event.

        Args:
            event_type: Type of event
            user_id: User performing the action
            resource: Resource being accessed
            action: Action being performed
            status: Status (success/failure)
            details: Additional details
            ip_address: User's IP address
        """
        event_data = {
            "event_type": event_type.value,
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "status": status,
            "ip_address": ip_address,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details or {},
        }

        # Get appropriate logger
        logger = self.loggers.get(event_type, self.loggers.get("system"))

        # Log as JSON
        logger.info(json.dumps(event_data))

    # Convenience methods for common events
    def log_login(
        self, user_id: str, success: bool, ip_address: str, details: Optional[Dict] = None
    ):
        """Log login attempt."""
        event_type = AuditEventType.LOGIN_SUCCESS if success else AuditEventType.LOGIN_FAILURE
        self.log_event(
            event_type=event_type,
            user_id=user_id,
            action="login",
            status="success" if success else "failure",
            details=details,
            ip_address=ip_address,
        )

    def log_data_access(
        self,
        user_id: str,
        resource: str,
        action: str,
        status: str = "success",
        details: Optional[Dict] = None,
    ):
        """Log data access event."""
        event_map = {
            "read": AuditEventType.DATA_READ,
            "write": AuditEventType.DATA_WRITE,
            "delete": AuditEventType.DATA_DELETE,
            "export": AuditEventType.DATA_EXPORT,
        }

        event_type = event_map.get(action.lower(), AuditEventType.DATA_READ)

        self.log_event(
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            action=action,
            status=status,
            details=details,
        )

    def log_model_operation(
        self,
        user_id: str,
        model_id: str,
        action: str,
        status: str = "success",
        details: Optional[Dict] = None,
    ):
        """Log model operation."""
        event_map = {
            "train": AuditEventType.MODEL_TRAIN,
            "deploy": AuditEventType.MODEL_DEPLOY,
            "delete": AuditEventType.MODEL_DELETE,
            "update": AuditEventType.MODEL_UPDATE,
        }

        event_type = event_map.get(action.lower(), AuditEventType.MODEL_UPDATE)

        self.log_event(
            event_type=event_type,
            user_id=user_id,
            resource=model_id,
            action=action,
            status=status,
            details=details,
        )

    def log_access_control(
        self, user_id: str, resource: str, granted: bool, details: Optional[Dict] = None
    ):
        """Log access control decision."""
        event_type = AuditEventType.ACCESS_GRANTED if granted else AuditEventType.ACCESS_DENIED

        self.log_event(
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            action="access",
            status="success" if granted else "denied",
            details=details,
        )

    def query_logs(
        self,
        category: str = "security",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
    ) -> list:
        """
        Query audit logs.

        Args:
            category: Log category
            start_time: Start time filter
            end_time: End time filter
            user_id: User ID filter

        Returns:
            List of matching log entries
        """
        log_file = self.log_dir / f"{category}.log"

        if not log_file.exists():
            return []

        logs = []
        with open(log_file, "r") as f:
            for line in f:
                try:
                    log_entry = json.loads(line)

                    # Apply filters
                    if user_id and log_entry.get("user_id") != user_id:
                        continue

                    if start_time or end_time:
                        log_time = datetime.fromisoformat(log_entry["timestamp"])
                        if start_time and log_time < start_time:
                            continue
                        if end_time and log_time > end_time:
                            continue

                    logs.append(log_entry)
                except json.JSONDecodeError:
                    continue

        return logs


# Example usage
if __name__ == "__main__":
    # Initialize audit logger
    audit = AuditLogger()

    # Log authentication events
    audit.log_login(
        user_id="user_123",
        success=True,
        ip_address="192.168.1.100",
        details={"method": "password", "mfa": True},
    )

    audit.log_login(
        user_id="user_456",
        success=False,
        ip_address="192.168.1.101",
        details={"method": "password", "reason": "invalid_password"},
    )

    # Log data access
    audit.log_data_access(
        user_id="user_123",
        resource="dataset:sensitive_data",
        action="read",
        details={"rows_accessed": 1000, "columns": ["email", "phone"]},
    )

    # Log model operations
    audit.log_model_operation(
        user_id="user_123",
        model_id="model_v1",
        action="deploy",
        details={"environment": "production", "version": "1.0.0"},
    )

    # Log access control
    audit.log_access_control(
        user_id="user_456",
        resource="admin_panel",
        granted=False,
        details={"required_role": "admin", "user_role": "user"},
    )

    print("✅ Audit logging test complete")
    print(f"📁 Logs stored in: {audit.log_dir}")

    # Query logs
    security_logs = audit.query_logs(category="security", user_id="user_123")
    print(f"📊 Found {len(security_logs)} security logs for user_123")
