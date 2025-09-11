from fastapi import FastAPI

from src.api.routers import recommender

app = FastAPI(title="LLM Recommender API", version="1.0", lifespan=recommender.lifespan)

app.include_router(recommender.router)

@app.get("/")
def root():
    return {"message": "Welcome to the Fashion Recommender API."}

