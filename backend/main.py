from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.chat import router as chat_router
from api.books import router as books_router
from api.notes import router as notes_router
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
    from storage.note_vector_store import make_note_vector_store
    from config import get_settings

    settings = get_settings()
    note_vector_store = make_note_vector_store(settings)

    app.state.agent = DeepReadAgent(note_vector_store=note_vector_store)
    app.state.note_agent = NoteAgent(note_vector_store=note_vector_store)
    app.state.followup_agent = FollowupAgent()
    app.state.router_agent = RouterAgent()
    app.state.critic_agent = CriticAgent()
    app.state.mem0 = Mem0Store()

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
app.include_router(notes_router)

# Serve extracted cover images
_covers_dir = Path("data/covers")
_covers_dir.mkdir(parents=True, exist_ok=True)
app.mount("/covers", StaticFiles(directory=str(_covers_dir)), name="covers")

# Serve EPUB files for in-browser reader
_books_dir = Path("data/books")
_books_dir.mkdir(parents=True, exist_ok=True)
app.mount("/ebooks", StaticFiles(directory=str(_books_dir)), name="ebooks")
