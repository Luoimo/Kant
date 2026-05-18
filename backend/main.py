from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.admin import router as admin_router
from api.auth import router as auth_router
from api.chat import router as chat_router
from api.books import router as books_router
from api.conversations import router as conversations_router
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

@asynccontextmanager
async def lifespan(app: FastAPI):
    from agents.deepread_agent import DeepReadAgent
    from agents.note_agent import NoteAgent
    from agents.followup_agent import FollowupAgent
    from agents.router_agent import RouterAgent
    from agents.critic_agent import CriticAgent
    from memory.mem0_store import Mem0Store
    from config import get_settings
    from storage.checkpoint_store import get_checkpoint_store
    from storage.postgres import ensure_postgres_schema

    settings = get_settings()
    ensure_postgres_schema(settings)
    checkpoint_store = get_checkpoint_store(settings)

    print("[DEBUG] Initializing DeepReadAgent...")
    app.state.agent = DeepReadAgent(checkpoint_store=checkpoint_store)
    print("[DEBUG] Initializing NoteAgent...")
    app.state.note_agent = NoteAgent()
    print("[DEBUG] Initializing FollowupAgent...")
    app.state.followup_agent = FollowupAgent()
    print("[DEBUG] Initializing RouterAgent...")
    app.state.router_agent = RouterAgent()
    print("[DEBUG] Initializing CriticAgent...")
    app.state.critic_agent = CriticAgent()
    print("[DEBUG] Initializing Mem0Store...")
    app.state.mem0 = Mem0Store()
    print("[DEBUG] Initialization complete.")

    print("[main] app started")
    yield
    print("[main] app stopped")


app = FastAPI(title="Kant Reading Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(books_router)
app.include_router(auth_router)
app.include_router(conversations_router)
app.include_router(admin_router)

# 向后兼容：如果仍在使用本地文件系统（OSS 未启用 / 老数据 source 是本地路径），
# 保留 /covers 与 /ebooks 静态资源路由。启用 OSS 后，接口层会直接返回签名 URL，
# 前端不再命中这两个路由。
from config import get_settings as _get_settings
_settings = _get_settings()
if not _settings.oss_access_key_id or not _settings.oss_secret_access_key:
    _covers_dir = Path(_settings.covers_dir)
    _covers_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/covers", StaticFiles(directory=str(_covers_dir)), name="covers")

    _books_dir = Path(_settings.books_data_dir)
    _books_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/ebooks", StaticFiles(directory=str(_books_dir)), name="ebooks")
