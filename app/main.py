from fastapi import FastAPI, Response, status

from app.api.router import api_router
from app.logger import get_logger
logger = get_logger(__name__)

app = FastAPI(
    title='ML Service',
    version='0.1.0',
    docs_url='/swagger',
)
app.include_router(api_router)