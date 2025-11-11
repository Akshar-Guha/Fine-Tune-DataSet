#!/usr/bin/env python3
"""Generate test JWT token for API testing."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from api.auth.jwt_handler import JWTHandler
from api.auth.permissions import get_permissions_for_role, Role

def main():
    handler = JWTHandler()
    permissions = get_permissions_for_role(Role.ADMIN)
    token = handler.create_access_token('test-user', 'admin', permissions)
    print(f'Test JWT Token: {token}')

if __name__ == '__main__':
    main()
