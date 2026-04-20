"""
Authentication and Role-Based Access Control (RBAC) module.
Provides JWT-based authentication, user management, and
role-based authorization for the medical imaging system.
"""

import os
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from pathlib import Path


# JWT handling - falls back to simple token if jose not available
try:
    from jose import jwt, JWTError
    HAS_JOSE = True
except ImportError:
    HAS_JOSE = False

try:
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    HAS_PASSLIB = True
except ImportError:
    HAS_PASSLIB = False


# Default secret key (override in production via environment variable)
SECRET_KEY = os.environ.get('XRAY_SECRET_KEY', secrets.token_hex(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours


# Role definitions
ROLES = {
    'admin': {
        'description': 'Full system access',
        'permissions': [
            'view_predictions', 'make_predictions', 'view_patients',
            'manage_patients', 'view_reports', 'generate_reports',
            'manage_users', 'view_audit_logs', 'manage_models',
            'batch_process', 'export_data', 'view_dashboard',
            'manage_settings', 'federated_learning',
        ]
    },
    'radiologist': {
        'description': 'Clinical diagnosis and feedback',
        'permissions': [
            'view_predictions', 'make_predictions', 'view_patients',
            'manage_patients', 'view_reports', 'generate_reports',
            'provide_feedback', 'batch_process', 'view_dashboard',
        ]
    },
    'technician': {
        'description': 'Upload images and view results',
        'permissions': [
            'view_predictions', 'make_predictions', 'view_patients',
            'view_reports', 'batch_process',
        ]
    },
    'viewer': {
        'description': 'View-only access',
        'permissions': [
            'view_predictions', 'view_patients', 'view_reports',
            'view_dashboard',
        ]
    }
}


def _hash_password(password: str) -> str:
    """Hash a password."""
    if HAS_PASSLIB:
        return pwd_context.hash(password)
    # Fallback to SHA-256 (less secure, but functional)
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
    return f"sha256:{salt}:{hashed}"


def _verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    if HAS_PASSLIB and not hashed_password.startswith('sha256:'):
        return pwd_context.verify(plain_password, hashed_password)
    # Fallback verification
    if hashed_password.startswith('sha256:'):
        parts = hashed_password.split(':')
        salt = parts[1]
        expected = parts[2]
        actual = hashlib.sha256(f"{salt}:{plain_password}".encode()).hexdigest()
        return actual == expected
    return False


class UserManager:
    """
    Manages user accounts and authentication.
    Uses a JSON file for simplicity (use a proper DB in production).
    """
    
    def __init__(self, users_file: str = 'data/users.json'):
        self.users_file = users_file
        os.makedirs(os.path.dirname(users_file) or '.', exist_ok=True)
        self.users = self._load_users()
        
        # Create default admin if no users exist
        if not self.users:
            self.create_user('admin', 'admin123', 'admin', 'System Administrator')
            print("Default admin user created (username: admin, password: admin123)")
    
    def _load_users(self) -> Dict:
        """Load users from JSON file."""
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_users(self):
        """Save users to JSON file."""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2, default=str)
    
    def create_user(self, username: str, password: str,
                    role: str = 'viewer', full_name: str = '') -> Dict:
        """
        Create a new user.
        
        Args:
            username: Unique username
            password: Plain text password (will be hashed)
            role: User role (admin, radiologist, technician, viewer)
            full_name: Display name
        
        Returns:
            User info dict or error dict
        """
        if username in self.users:
            return {'status': 'error', 'message': 'Username already exists'}
        
        if role not in ROLES:
            return {'status': 'error', 'message': f'Invalid role: {role}'}
        
        self.users[username] = {
            'username': username,
            'password_hash': _hash_password(password),
            'role': role,
            'full_name': full_name,
            'is_active': True,
            'created_at': datetime.utcnow().isoformat(),
            'last_login': None,
        }
        self._save_users()
        
        return {
            'status': 'success',
            'username': username,
            'role': role,
        }
    
    def authenticate(self, username: str, password: str) -> Optional[Dict]:
        """
        Authenticate a user.
        
        Returns:
            User dict if successful, None otherwise
        """
        user = self.users.get(username)
        if not user:
            return None
        
        if not user.get('is_active', True):
            return None
        
        if not _verify_password(password, user['password_hash']):
            return None
        
        # Update last login
        user['last_login'] = datetime.utcnow().isoformat()
        self._save_users()
        
        return {
            'username': user['username'],
            'role': user['role'],
            'full_name': user['full_name'],
        }
    
    def get_user(self, username: str) -> Optional[Dict]:
        """Get user info (excluding password)."""
        user = self.users.get(username)
        if user:
            return {k: v for k, v in user.items() if k != 'password_hash'}
        return None
    
    def list_users(self) -> List[Dict]:
        """List all users (excluding passwords)."""
        return [
            {k: v for k, v in user.items() if k != 'password_hash'}
            for user in self.users.values()
        ]
    
    def update_user(self, username: str, **kwargs) -> bool:
        """Update user fields."""
        if username not in self.users:
            return False
        
        allowed_fields = {'role', 'full_name', 'is_active'}
        for key, value in kwargs.items():
            if key in allowed_fields:
                self.users[username][key] = value
        
        self._save_users()
        return True
    
    def change_password(self, username: str, old_password: str,
                        new_password: str) -> bool:
        """Change user password."""
        user = self.users.get(username)
        if not user:
            return False
        
        if not _verify_password(old_password, user['password_hash']):
            return False
        
        user['password_hash'] = _hash_password(new_password)
        self._save_users()
        return True
    
    def delete_user(self, username: str) -> bool:
        """Soft-delete a user (deactivate)."""
        if username in self.users:
            self.users[username]['is_active'] = False
            self._save_users()
            return True
        return False


def create_access_token(data: Dict, expires_delta: timedelta = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Token payload (should include 'sub' for username)
        expires_delta: Token expiration time
    
    Returns:
        Encoded JWT string
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({'exp': expire})
    
    if HAS_JOSE:
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    else:
        # Fallback: simple token (not as secure, but functional)
        token_data = json.dumps(to_encode, default=str)
        token_hash = hashlib.sha256(f"{token_data}:{SECRET_KEY}".encode()).hexdigest()
        import base64
        encoded = base64.b64encode(token_data.encode()).decode()
        return f"{encoded}.{token_hash}"


def verify_token(token: str) -> Optional[Dict]:
    """
    Verify and decode a JWT token.
    
    Returns:
        Decoded payload dict, or None if invalid
    """
    try:
        if HAS_JOSE:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        else:
            # Fallback verification
            import base64
            parts = token.split('.')
            if len(parts) != 2:
                return None
            encoded, provided_hash = parts
            token_data = base64.b64decode(encoded).decode()
            expected_hash = hashlib.sha256(
                f"{token_data}:{SECRET_KEY}".encode()
            ).hexdigest()
            if provided_hash != expected_hash:
                return None
            payload = json.loads(token_data)
            exp = datetime.fromisoformat(payload.get('exp', '2000-01-01'))
            if exp < datetime.utcnow():
                return None
            return payload
    except Exception:
        return None


def check_permission(user_role: str, required_permission: str) -> bool:
    """
    Check if a role has a specific permission.
    
    Args:
        user_role: The user's role
        required_permission: The permission to check
    
    Returns:
        True if the role has the permission
    """
    role_info = ROLES.get(user_role, {})
    return required_permission in role_info.get('permissions', [])


def require_permission(permission: str):
    """
    Decorator for functions requiring specific permissions.
    Works with Streamlit session state.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Try to get user from kwargs or session state
            user = kwargs.get('current_user', None)
            
            if user is None:
                try:
                    import streamlit as st
                    user = st.session_state.get('current_user', None)
                except ImportError:
                    pass
            
            if user is None:
                raise PermissionError("Authentication required")
            
            if not check_permission(user.get('role', ''), permission):
                raise PermissionError(
                    f"Permission denied: '{permission}' required. "
                    f"Your role '{user.get('role', 'unknown')}' does not have this permission."
                )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Global user manager instance
_user_manager = None


def get_user_manager(users_file: str = 'data/users.json') -> UserManager:
    """Get or create the global user manager instance."""
    global _user_manager
    if _user_manager is None:
        _user_manager = UserManager(users_file=users_file)
    return _user_manager


if __name__ == "__main__":
    manager = UserManager(users_file='data/users_test.json')
    
    # Test user creation
    result = manager.create_user('dr_smith', 'password123', 'radiologist', 'Dr. Smith')
    print(f"Created user: {result}")
    
    # Test authentication
    auth = manager.authenticate('dr_smith', 'password123')
    print(f"Authenticated: {auth}")
    
    # Test token
    token = create_access_token({'sub': 'dr_smith', 'role': 'radiologist'})
    print(f"Token: {token[:50]}...")
    
    decoded = verify_token(token)
    print(f"Decoded: {decoded}")
    
    # Test permissions
    print(f"\nRadiologist can make predictions: {check_permission('radiologist', 'make_predictions')}")
    print(f"Technician can manage users: {check_permission('technician', 'manage_users')}")
    print(f"Admin can manage users: {check_permission('admin', 'manage_users')}")
    
    # Cleanup
    os.remove('data/users_test.json')
    print("\nAuth module tested successfully!")
