"""
API Module for AAUto trading platform.

This module provides REST API endpoints with comprehensive authentication,
versioning, request validation, and response formatting. It integrates with
core components including uncertainty quantification, monitoring, and adaptation.
"""
import asyncio
import functools
import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast

from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, ValidationError, validator
import jwt
from jwt.exceptions import InvalidTokenError

# Internal core systems imports
try:
    from src.core.uncertainty import UncertaintyQuantifier
    from src.core.monitoring import SystemMonitor
    from src.core.adaptation import AdaptationManager
    from src.core.validation import RequestValidator
    from src.core.coordination import Coordinator
    from src.core.recovery import RecoveryManager
    from src.analytics import AnalyticsEngine
except ImportError:
    # Placeholder implementations for development
    class UncertaintyQuantifier:
        @staticmethod
        async def quantify_uncertainty(data: Any) -> Dict[str, float]:
            return {"aleatory": 0.1, "epistemic": 0.05, "total": 0.15}

    class SystemMonitor:
        @staticmethod
        async def record_api_request(endpoint: str, latency: float, status_code: int) -> None:
            pass

    class AdaptationManager:
        @staticmethod
        async def check_adaptation_status() -> Dict[str, Any]:
            return {"status": "stable", "last_adaptation": "2023-01-01T00:00:00Z"}

    class RequestValidator:
        @staticmethod
        async def validate(data: Any, schema: Type) -> Tuple[bool, List[str]]:
            return True, []

    class Coordinator:
        @staticmethod
        async def register_api_component() -> str:
            return "api-component-id"

    class RecoveryManager:
        @staticmethod
        async def is_system_healthy() -> bool:
            return True

    class AnalyticsEngine:
        @staticmethod
        async def log_api_interaction(data: Dict[str, Any]) -> None:
            pass

# Configuration
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"
SECRET_KEY = "REPLACE_WITH_SECURE_KEY_FROM_ENV_OR_VAULT"  # In production, use environment or vault
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize FastAPI app
app = FastAPI(
    title="AAUto Trading API",
    description="Advanced algorithmic trading platform with self-improving capabilities",
    version=API_VERSION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize oauth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{API_PREFIX}/token")

# Initialize core components
uncertainty_quantifier = UncertaintyQuantifier()
system_monitor = SystemMonitor()
adaptation_manager = AdaptationManager()
request_validator = RequestValidator()
coordinator = Coordinator()
recovery_manager = RecoveryManager()
analytics_engine = AnalyticsEngine()


# --- Models ---

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    permissions: List[str] = Field(default_factory=list)


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None
    permissions: List[str] = Field(default_factory=list)


class TradeRequest(BaseModel):
    symbol: str
    quantity: float
    side: str  # buy or sell
    order_type: str = "market"
    limit_price: Optional[float] = None
    
    @validator("side")
    def validate_side(cls, v):
        if v.lower() not in ["buy", "sell"]:
            raise ValueError("side must be either 'buy' or 'sell'")
        return v.lower()
    
    @validator("order_type")
    def validate_order_type(cls, v):
        valid_types = ["market", "limit", "stop", "stop_limit"]
        if v.lower() not in valid_types:
            raise ValueError(f"order_type must be one of: {', '.join(valid_types)}")
        return v.lower()
    
    @validator("limit_price")
    def validate_limit_price(cls, v, values):
        if values.get("order_type") in ["limit", "stop_limit"] and v is None:
            raise ValueError("limit_price is required for limit and stop_limit orders")
        return v


class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    uncertainty: Optional[Dict[str, float]] = None
    version: str = API_VERSION
    timestamp: float = Field(default_factory=lambda: time.time())


class ErrorDetail(BaseModel):
    loc: List[str]
    msg: str
    type: str


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    errors: List[ErrorDetail] = Field(default_factory=list)
    version: str = API_VERSION
    timestamp: float = Field(default_factory=lambda: time.time())


# --- Authentication ---

# Mock user database - In production, replace with actual database access
fake_users_db = {
    "alice": {
        "username": "alice",
        "full_name": "Alice Wonderland",
        "email": "alice@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "disabled": False,
        "permissions": ["read:trades", "write:trades", "read:portfolio"]
    },
    "bob": {
        "username": "bob",
        "full_name": "Bob Builder",
        "email": "bob@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "disabled": False,
        "permissions": ["read:trades", "read:portfolio"]
    }
}


def verify_password(plain_password, hashed_password):
    # In production, use proper password hashing library
    return plain_password == "secret"  # Simplified for example


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return User(**user_dict)


def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, fake_db[username]["hashed_password"]):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, permissions=payload.get("permissions", []))
    except InvalidTokenError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def has_permission(required_permission: str):
    async def check_permission(current_user: User = Depends(get_current_active_user)):
        if required_permission not in current_user.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {required_permission} required"
            )
        return current_user
    return check_permission


# --- Middleware ---

@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    start_time = time.time()
    
    # Check system health before processing
    is_healthy = await recovery_manager.is_system_healthy()
    if not is_healthy:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"success": False, "message": "System is recovering, please try again later"}
        )
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Record metrics
        await system_monitor.record_api_request(
            endpoint=str(request.url.path),
            latency=process_time,
            status_code=response.status_code
        )
        
        return response
    except Exception as e:
        # Log the exception
        logging.exception("Error processing request")
        process_time = time.time() - start_time
        
        # Record the failure
        await system_monitor.record_api_request(
            endpoint=str(request.url.path),
            latency=process_time,
            status_code=500
        )
        
        # Return error response
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "message": "Internal server error",
                "version": API_VERSION,
                "timestamp": time.time()
            }
        )


# --- Decorators ---

def with_uncertainty(func):
    """Decorator to add uncertainty estimates to API responses."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)
        
        # Only add uncertainty to successful responses
        if isinstance(result, dict) and result.get("success", False):
            # Extract the relevant data for uncertainty estimation
            data_for_uncertainty = result.get("data", {})
            
            # Get uncertainty estimates
            uncertainty = await uncertainty_quantifier.quantify_uncertainty(data_for_uncertainty)
            
            # Add uncertainty to the response
            result["uncertainty"] = uncertainty
        
        return result
    return wrapper


def log_api_interaction(func):
    """Decorator to log API interactions for analytics."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract request details if available
        request = None
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break
        
        # Get function details
        interaction_data = {
            "endpoint": func.__name__,
            "timestamp": time.time(),
        }
        
        if request:
            interaction_data["method"] = request.method
            interaction_data["path"] = str(request.url.path)
            interaction_data["client"] = request.client.host if request.client else "unknown"
        
        # Execute the function
        result = await func(*args, **kwargs)
        
        # Add response data
        if isinstance(result, dict):
            interaction_data["status"] = "success" if result.get("success", False) else "failure"
        
        # Log the interaction
        await analytics_engine.log_api_interaction(interaction_data)
        
        return result
    return wrapper


# --- Authentication Endpoints ---

@app.post(f"{API_PREFIX}/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create the access token with permissions
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "permissions": user.permissions},
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


# --- API Endpoints ---

@app.get(f"{API_PREFIX}/health")
async def health_check():
    """Public health check endpoint that doesn't require authentication."""
    is_healthy = await recovery_manager.is_system_healthy()
    adaptation_status = await adaptation_manager.check_adaptation_status()
    
    if is_healthy:
        return {
            "success": True,
            "message": "System is healthy",
            "data": {
                "status": "online",
                "adaptation": adaptation_status,
                "version": API_VERSION
            }
        }
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "success": False,
                "message": "System is recovering",
                "data": {
                    "status": "recovering",
                    "adaptation": adaptation_status,
                    "version": API_VERSION
                }
            }
        )


@app.get(f"{API_PREFIX}/profile", response_model=ApiResponse)
@log_api_interaction
async def get_profile(current_user: User = Depends(get_current_active_user)):
    """Get the profile of the currently authenticated user."""
    return {
        "success": True,
        "message": "User profile retrieved successfully",
        "data": {
            "username": current_user.username,
            "email": current_user.email,
            "full_name": current_user.full_name,
            "permissions": current_user.permissions
        }
    }


@app.get(f"{API_PREFIX}/portfolio", response_model=ApiResponse)
@with_uncertainty
@log_api_interaction
async def get_portfolio(
    request: Request,
    current_user: User = Depends(has_permission("read:portfolio"))
):
    """Get the user's portfolio with uncertainty estimates."""
    # Mock portfolio data - in production, fetch from database
    portfolio_data = {
        "total_value": 125000.50,

