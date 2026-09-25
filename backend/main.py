from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.explanation import router as explanation_router

app = FastAPI(title="LabLens AI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(explanation_router)


@app.get("/")
def read_root():
    return {"message": "LabLens AI backend is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}
