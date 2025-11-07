"""Role-based access control (RBAC)."""
from enum import Enum
from typing import List, Set
from functools import wraps

from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials



class Role(str, Enum):
    """User roles."""
    ADMIN = "admin"
    DATA_SCIENTIST = "data_scientist"
    ML_ENGINEER = "ml_engineer"
    VIEWER = "viewer"


class Permission(str, Enum):
    """Granular permissions."""
    # Dataset permissions
    DATASET_READ = "dataset:read"
    DATASET_WRITE = "dataset:write"
    DATASET_DELETE = "dataset:delete"

    # Job permissions
    JOB_READ = "job:read"
    JOB_SUBMIT = "job:submit"
    JOB_CANCEL = "job:cancel"

    # Artifact permissions
    ARTIFACT_READ = "artifact:read"
    ARTIFACT_WRITE = "artifact:write"
    ARTIFACT_DELETE = "artifact:delete"
    ARTIFACT_PROMOTE = "artifact:promote"

    # Deployment permissions
    DEPLOY_READ = "deploy:read"
    DEPLOY_CREATE = "deploy:create"
    DEPLOY_DELETE = "deploy:delete"

    # Admin permissions
    USER_MANAGE = "user:manage"
    SYSTEM_CONFIG = "system:config"


# Role to permissions mapping
ROLE_PERMISSIONS: dict[Role, Set[Permission]] = {
    Role.ADMIN: set(Permission),  # All permissions
    Role.DATA_SCIENTIST: {
        Permission.DATASET_READ,
        Permission.DATASET_WRITE,
        Permission.JOB_READ,
        Permission.JOB_SUBMIT,
        Permission.ARTIFACT_READ,
        Permission.ARTIFACT_WRITE,
        Permission.DEPLOY_READ,
    },
    Role.ML_ENGINEER: {
        Permission.DATASET_READ,
        Permission.JOB_READ,
        Permission.JOB_SUBMIT,
        Permission.JOB_CANCEL,
        Permission.ARTIFACT_READ,
        Permission.ARTIFACT_WRITE,
        Permission.ARTIFACT_PROMOTE,
        Permission.DEPLOY_READ,
        Permission.DEPLOY_CREATE,
        Permission.DEPLOY_DELETE,
    },
    Role.VIEWER: {
        Permission.DATASET_READ,
        Permission.JOB_READ,
        Permission.ARTIFACT_READ,
        Permission.DEPLOY_READ,
    }
}


security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """Get current authenticated user from JWT token."""
    token = credentials.credentials
    from .jwt_handler import JWTHandler
    jwt_handler = JWTHandler()

    try:
        payload = jwt_handler.verify_token(token)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "user_id": payload["sub"],
        "role": payload["role"],
        "permissions": payload["permissions"]
    }


def require_permissions(*required_permissions: Permission):
    """Decorator to require specific permissions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, current_user: dict = Depends(get_current_user), **kwargs):
            user_permissions = set(current_user.get("permissions", []))
            required = {perm.value for perm in required_permissions}

            if not required.issubset(user_permissions):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )

            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator


def require_role(*allowed_roles: Role):
    """Decorator to require specific role."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, current_user: dict = Depends(get_current_user), **kwargs):
            user_role = current_user.get("role")

            if user_role not in [role.value for role in allowed_roles]:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role {user_role} not allowed"
                )

            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator


def get_permissions_for_role(role: Role) -> List[str]:
    """Get all permissions for a role."""
    return [perm.value for perm in ROLE_PERMISSIONS.get(role, set())]
