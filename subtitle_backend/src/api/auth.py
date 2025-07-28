from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import hashlib

# In-memory user store for demonstration; replace with persistent DB!
users_db = {}

SECRET_KEY = "development_only_secret_DO_NOT_USE_IN_PRODUCTION"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

# PUBLIC_INTERFACE
def get_password_hash(password: str) -> str:
    """Hash a password (dummy sha256)."""
    return hashlib.sha256(password.encode()).hexdigest()

# PUBLIC_INTERFACE
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password vs. hash."""
    return get_password_hash(plain_password) == hashed_password

# PUBLIC_INTERFACE
def authenticate_user(email: str, password: str) -> Optional[dict]:
    """Authenticate user in the demo DB."""
    user = users_db.get(email)
    if not user:
        return None
    if not verify_password(password, user['password']):
        return None
    return user

# PUBLIC_INTERFACE
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT with expiry."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# PUBLIC_INTERFACE
def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get user from JWT token or raise."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None or email not in users_db:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return users_db[email]
