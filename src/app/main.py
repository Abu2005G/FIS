from fastapi import FastAPI
from src.app.routes import router
from src.app.database import engine
from src.app import models

models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="My SaaS API", description="A FastAPI SaaS service", version="1.0.0"
)

app.include_router(router)


@app.get("/")
def root():
    return {"message": "API is live 🚀"}
