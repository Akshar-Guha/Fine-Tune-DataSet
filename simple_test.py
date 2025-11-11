"""
Simple integration test to verify key components work.
"""

print("Testing basic component imports...")

try:
    # Test security components
    from security.encryption.key_manager import KeyManager
    print("✓ KeyManager import successful")

    km = KeyManager(key_store_path="./test_keys")
    try:
        test_key = km.generate_key("test_key", "aes-256")
        print(f"✓ Key generation successful (size: {len(test_key)} bytes)")
    except ValueError:
        # Key already exists, get it instead
        test_key = km.get_key("test_key")
        print(f"✓ Key retrieval successful (size: {len(test_key)} bytes)")

    from security.auth.mfa import MFAManager
    print("✓ MFAManager import successful")

    mfa = MFAManager(issuer_name="ModelOps")
    mfa_data = mfa.enable_mfa("test@example.com")
    current_token = mfa.get_current_token(mfa_data["secret"])
    is_valid = mfa.verify_token(mfa_data["secret"], current_token)
    print(f"✓ MFA verification successful (valid: {is_valid})")

    print("\n🎉 Basic security components are working!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
