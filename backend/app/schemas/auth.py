"""Public authentication-domain schemas without credential material."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PublicUser(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    display_name: Optional[str]
    auth_provider: str
    created_at: datetime


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=15, max_length=128)
    display_name: Optional[str] = Field(default=None, max_length=255)


class LoginRequest(BaseModel):
    email: str
    password: str = Field(min_length=1, max_length=128)


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: PublicUser
