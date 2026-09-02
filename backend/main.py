from fastapi import FastAPI
from contextlib import asynccontextmanager

from timer import timing_middleware
from fastapi.middleware.cors import CORSMiddleware
from model_store import load_models
from predict import router as predict_router
from application import router as applicant_router
@asynccontextmanager
async def lifespan(app:FastAPI):
    load_models()
    yield
app=FastAPI(lifespan=lifespan)
app.middleware ("http")(timing_middleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(predict_router)
app.include_router(applicant_router)
