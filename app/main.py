from fastapi import FastAPI

app = FastAPI(title="ContextForge")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
