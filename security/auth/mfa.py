"""
Multi-Factor Authentication (MFA) using TOTP.

Implements Time-based One-Time Password (TOTP) for 2FA.
"""

import pyotp
import qrcode
from io import BytesIO
from typing import Optional
import base64


class MFAManager:
    """Manages MFA (TOTP) for users."""

    def __init__(self, issuer_name: str = "ModelOps"):
        """
        Initialize MFA Manager.

        Args:
            issuer_name: Name shown in authenticator apps
        """
        self.issuer_name = issuer_name

    def generate_secret(self) -> str:
        """
        Generate a new TOTP secret for a user.

        Returns:
            Base32-encoded secret
        """
        return pyotp.random_base32()

    def get_provisioning_uri(self, user_email: str, secret: str) -> str:
        """
        Generate provisioning URI for QR code.

        Args:
            user_email: User's email address
            secret: TOTP secret

        Returns:
            Provisioning URI
        """
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(name=user_email, issuer_name=self.issuer_name)

    def generate_qr_code(self, user_email: str, secret: str) -> str:
        """
        Generate QR code for authenticator app setup.

        Args:
            user_email: User's email
            secret: TOTP secret

        Returns:
            Base64-encoded QR code image
        """
        uri = self.get_provisioning_uri(user_email, secret)

        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(uri)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")

        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return f"data:image/png;base64,{img_str}"

    def verify_token(self, secret: str, token: str, valid_window: int = 1) -> bool:
        """
        Verify a TOTP token.

        Args:
            secret: User's TOTP secret
            token: 6-digit token from authenticator app
            valid_window: Number of time windows to check (default: 1 = ±30 seconds)

        Returns:
            True if token is valid, False otherwise
        """
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=valid_window)

    def get_current_token(self, secret: str) -> str:
        """
        Get current valid token (for testing).

        Args:
            secret: TOTP secret

        Returns:
            Current 6-digit token
        """
        totp = pyotp.TOTP(secret)
        return totp.now()

    def enable_mfa(self, user_email: str) -> dict:
        """
        Enable MFA for a user.

        Args:
            user_email: User's email

        Returns:
            Dict with secret and QR code
        """
        secret = self.generate_secret()
        qr_code = self.generate_qr_code(user_email, secret)

        return {"secret": secret, "qr_code": qr_code, "backup_codes": self._generate_backup_codes()}

    def _generate_backup_codes(self, count: int = 10) -> list:
        """
        Generate backup codes for recovery.

        Args:
            count: Number of backup codes to generate

        Returns:
            List of backup codes
        """
        import secrets

        codes = []
        for _ in range(count):
            code = "-".join(
                [secrets.token_hex(2).upper() for _ in range(4)]
            )
            codes.append(code)

        return codes


# Example usage
if __name__ == "__main__":
    mfa = MFAManager(issuer_name="ModelOps")

    # Enable MFA for user
    user_email = "user@example.com"
    mfa_data = mfa.enable_mfa(user_email)

    print("=== MFA Setup ===")
    print(f"Secret: {mfa_data['secret']}")
    print(f"QR Code (Base64): {mfa_data['qr_code'][:50]}...")
    print(f"\nBackup Codes:")
    for code in mfa_data["backup_codes"]:
        print(f"  - {code}")

    # Test token verification
    current_token = mfa.get_current_token(mfa_data["secret"])
    print(f"\nCurrent Token: {current_token}")

    is_valid = mfa.verify_token(mfa_data["secret"], current_token)
    print(f"Token Valid: {is_valid}")

    # Test invalid token
    is_valid = mfa.verify_token(mfa_data["secret"], "000000")
    print(f"Invalid Token Valid: {is_valid}")

    print("\n✅ MFA test complete")
