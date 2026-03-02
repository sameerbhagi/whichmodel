"""WhichModel API — FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.models import router as models_router
from routes.recommend import router as recommend_router
from routes.upload import router as upload_router

import os

app = FastAPI(
    title="WhichModel API",
    description="Compare frontier AI LLM models and get personalized recommendations",
    version="1.0.0",
)

# CORS — allow frontend dev server and production domains
_cors_origins = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else []
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "https://whichmodel.sammyinnovations.com",
] + [o.strip() for o in _cors_origins if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(models_router)
app.include_router(recommend_router)
app.include_router(upload_router)


@app.get("/")
def root():
    return {
        "name": "WhichModel API",
        "version": "1.0.0",
        "docs": "/docs",
    }
