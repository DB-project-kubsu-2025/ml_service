from fastapi import APIRouter

from app.api import healthcheck, prediction

api_router = APIRouter()

api_router.include_router(healthcheck.router, tags=["Healthcheck"])
api_router.include_router(prediction.router, tags=["Prediction"])
