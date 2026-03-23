from fastapi import FastAPI

from backend.api.chat import router as chat_router
from backend.api.books import router as books_router
from backend.api.notes import router as notes_router
from backend.api.reader import router as reader_router

app = FastAPI(title="Kant Reading Agent")

app.include_router(chat_router)
app.include_router(books_router)
app.include_router(notes_router)
app.include_router(reader_router)
