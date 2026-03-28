'''
Description: 
version: v1.0.0
Author: GaoMingze
Date: 2026-03-25 22:49:35
LastEditors: GaoMingze
LastEditTime: 2026-03-25 22:52:53
'''
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.api.chat import router as chat_router
from backend.api.books import router as books_router
from backend.api.notes import router as notes_router
from backend.api.reader import router as reader_router
from backend.memory.mem0_store import Mem0Store
from backend.team.team import AgentTeam
from backend.team.dispatcher import Dispatcher

_team: AgentTeam | None = None
_dispatcher: Dispatcher | None = None

def get_dispatcher() -> Dispatcher:                                                                                                           
    if _dispatcher is None:                                                                                                                   
        raise RuntimeError("App not started — lifespan not complete")                                                                         
    return _dispatcher                                                                                                                        
                                                                                                                                             

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _team, _dispatcher
    _team = AgentTeam()
    _team.startup()
    mem0 = Mem0Store()
    _dispatcher = Dispatcher(_team, mem0)
    app.state.dispatcher = _dispatcher
    yield
    _team.shutdown()


app = FastAPI(title="Kant Reading Agent", lifespan=lifespan)

app.include_router(chat_router)
app.include_router(books_router)
app.include_router(notes_router)
app.include_router(reader_router)
