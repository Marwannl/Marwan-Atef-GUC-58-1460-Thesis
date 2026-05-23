from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from database import engine, Base
from routers import auth as auth_router
from routers import chats as chats_router
from routers import chat as chat_router
from routers import files as files_router
from routers import demo as demo_router

Base.metadata.create_all(bind=engine)

from sqlalchemy import text
_NEW_DEFAULT = "qwen/qwen3-next-80b-a3b-instruct:free"
_OLD_MODELS = (
    "llama3.2", "qwen2.5", "llama3", "mistral",
    "qwen/qwen-2.5-72b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
)
with engine.connect() as _conn:
    for _col, _type in [("model", "TEXT"), ("system_prompt", "TEXT")]:
        try:
            _conn.execute(text(f"ALTER TABLE chats ADD COLUMN {_col} {_type}"))
            _conn.commit()
        except Exception:
            pass
    for old in _OLD_MODELS:
        _conn.execute(text(
            f"UPDATE chats SET model = '{_NEW_DEFAULT}' WHERE model = :m"
        ), {"m": old})
    _conn.commit()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(auth_router.router)
app.include_router(chats_router.router)
app.include_router(chat_router.router)
app.include_router(files_router.router)
app.include_router(demo_router.router)


@app.get("/")
async def root():
    return FileResponse(
        "static/index.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )
