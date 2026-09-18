from fastapi import FastAPI

app = FastAPI(title="LabLens AI API", version="0.1.0")


@app.get("/")
def read_root():
    return {"message": "LabLens AI backend is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}
