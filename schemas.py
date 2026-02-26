from typing import Optional, List
from pydantic import BaseModel
import datetime

class UserBase(BaseModel):
    phone_number: str

class UserCreate(UserBase):
    name: str
    password: str
    area: Optional[str] = None

class User(UserBase):
    id: int
    name: Optional[str] = None
    area: Optional[str] = None
    is_active: bool

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    phone_number: Optional[str] = None

class ComplaintBase(BaseModel):
    title: str
    description: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    # Assuming user uploads file, API will handle it and return URL, or they send URL directly
    image_url: str 
    
class ComplaintCreate(ComplaintBase):
    department: Optional[str] = None   # AI will auto-fill if not provided
    subcategory: Optional[str] = None  # AI will auto-fill if not provided
    voice_url: Optional[str] = None

class ComplaintAIResponse(ComplaintBase):
    id: int
    status: str
    created_at: datetime.datetime
    reporter_id: int
    department: Optional[str] = None
    voice_url: Optional[str] = None
    issue_type: Optional[str] = None # This is subcategory select
    severity_score: Optional[float] = None
    confidence_score: Optional[float] = None
    department_suggested: Optional[str] = None
    votes: int = 0

    class Config:
        from_attributes = True

class Worker(BaseModel):
    id: int
    name: str
    department: str
    status: str
    phone: str
    location: str
    rating: float
    active_tasks: int
    completed_tasks: int

    class Config:
        from_attributes = True
