"""Quick test script to verify API functionality."""
import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("\n=== Testing Health Endpoint ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    return True


def test_root():
    """Test root endpoint."""
    print("\n=== Testing Root Endpoint ===")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    return True


def test_metrics():
    """Test metrics endpoint."""
    print("\n=== Testing Metrics Endpoint ===")
    response = requests.get(f"{BASE_URL}/metrics")
    print(f"Status: {response.status_code}")
    print(f"Metrics available: {len(response.text)} bytes")
    assert response.status_code == 200
    return True


def test_docs():
    """Test API documentation."""
    print("\n=== Testing API Documentation ===")
    response = requests.get(f"{BASE_URL}/docs")
    print(f"Status: {response.status_code}")
    print("Documentation available at: http://localhost:8000/docs")
    assert response.status_code == 200
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("ModelOps Lightweight Edition - API Tests")
    print("=" * 50)

    tests = [
        test_health,
        test_root,
        test_metrics,
        test_docs
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
                print("✓ PASSED")
        except Exception as e:
            failed += 1
            print(f"✗ FAILED: {e}")

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed == 0:
        print("\n🎉 All tests passed! API is working correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Check the errors above.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error running tests: {e}")
        print("\nMake sure the API server is running:")
        print("  poetry run uvicorn api.rest.main:app --reload --port 8000")
