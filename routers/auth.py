from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database import get_db
from models import User
from auth import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


class AuthRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/register", response_model=TokenResponse)
def register(request: AuthRequest, db: Session = Depends(get_db)):
    username = request.username.strip().lower()
    if not username or len(request.password) < 4:
        raise HTTPException(status_code=400, detail="Invalid username or password")
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    user = User(username=username, hashed_password=hash_password(request.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return TokenResponse(access_token=create_access_token({"sub": user.username}))


@router.post("/login", response_model=TokenResponse)
def login(request: AuthRequest, db: Session = Depends(get_db)):
    username = request.username.strip().lower()
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenResponse(access_token=create_access_token({"sub": user.username}))
