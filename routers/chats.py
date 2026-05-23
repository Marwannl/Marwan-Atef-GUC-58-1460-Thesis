from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from typing import Optional
from pydantic import BaseModel
from database import get_db
from models import User, Chat, Message
from auth import get_current_user

router = APIRouter(prefix="/chats", tags=["chats"])


def _get_user(username: str, db: Session) -> User:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


@router.get("")
def list_chats(db: Session = Depends(get_db), username: str = Depends(get_current_user)):
    user = _get_user(username, db)
    chats = (
        db.query(Chat)
        .filter(Chat.user_id == user.id)
        .order_by(Chat.updated_at.desc())
        .all()
    )
    return [{"id": c.id, "title": c.title, "updated_at": str(c.updated_at), "model": c.model, "system_prompt": c.system_prompt} for c in chats]


class CreateChatRequest(BaseModel):
    model: Optional[str] = None
    system_prompt: Optional[str] = None


@router.post("")
def create_chat(
    request: CreateChatRequest = Body(default=CreateChatRequest()),
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    user = _get_user(username, db)
    chat = Chat(
        user_id=user.id,
        title="New Chat",
        model=request.model or None,
        system_prompt=request.system_prompt or None,
    )
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return {"id": chat.id, "title": chat.title, "model": chat.model}


class PatchChatRequest(BaseModel):
    model: Optional[str] = None
    system_prompt: Optional[str] = None


@router.patch("/{chat_id}")
def patch_chat(
    chat_id: int,
    request: PatchChatRequest,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    user = _get_user(username, db)
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user.id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    if request.model is not None:
        chat.model = request.model
    if request.system_prompt is not None:
        chat.system_prompt = request.system_prompt
    db.commit()
    return {"id": chat.id, "model": chat.model, "system_prompt": chat.system_prompt}


@router.delete("/{chat_id}")
def delete_chat(
    chat_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    user = _get_user(username, db)
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user.id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    db.delete(chat)
    db.commit()
    return {"status": "deleted"}


@router.get("/{chat_id}/messages")
def get_messages(
    chat_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    user = _get_user(username, db)
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    if chat.user_id != user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    messages = (
        db.query(Message)
        .filter(Message.chat_id == chat_id)
        .order_by(Message.created_at)
        .all()
    )
    return [{"role": m.role, "content": m.content} for m in messages]
