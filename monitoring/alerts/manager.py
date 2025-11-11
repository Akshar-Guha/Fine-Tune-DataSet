"""Alert Manager for multi-channel notifications."""

from typing import Dict, List, Optional
from enum import Enum
import smtplib
from email.mime.text import MIMEText


class AlertChannel(Enum):
    """Alert notification channels."""

    EMAIL = "email"
    SLACK = "slack"
    SMS = "sms"
    WEBHOOK = "webhook"


class AlertManager:
    """Manages alerts across multiple channels."""

    def __init__(self, config: Dict):
        self.config = config
        self.alert_history = []

    def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "medium",
        channels: Optional[List[AlertChannel]] = None,
    ):
        """Send alert to specified channels."""
        channels = channels or [AlertChannel.EMAIL]

        alert = {
            "title": title,
            "message": message,
            "severity": severity,
            "channels": [c.value for c in channels],
        }

        for channel in channels:
            if channel == AlertChannel.EMAIL:
                self._send_email(title, message)
            elif channel == AlertChannel.SLACK:
                self._send_slack(title, message)

        self.alert_history.append(alert)
        print(f"✅ Alert sent: {title}")

    def _send_email(self, title: str, message: str):
        """Send email alert."""
        # Simplified email sending
        print(f"📧 Email: {title} - {message}")

    def _send_slack(self, title: str, message: str):
        """Send Slack alert."""
        # Simplified Slack sending
        print(f"💬 Slack: {title} - {message}")


# Example
if __name__ == "__main__":
    am = AlertManager(config={})
    am.send_alert(
        title="Model Accuracy Drop",
        message="Model accuracy dropped below 80%",
        severity="high",
        channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
    )
    print("✅ Alert manager test complete")
