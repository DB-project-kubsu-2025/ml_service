from fastapi import APIRouter, Response, status, Request

from app.model.predictor import PredictionAPI
from app.schemas import ErrorResponse, SuccessPredictionResponse, PredictionRequest

router = APIRouter()


@router.post(
    '/prediction/one_category/',
    response_model=SuccessPredictionResponse,
    summary='Прогноз продаж по одной категории',
    tags=['Prediction'],
    responses={
        status.HTTP_400_BAD_REQUEST: {'model': ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {'model': ErrorResponse},
    },
)
async def prediction_by_one_category(request: PredictionRequest):
    prediction_api = PredictionAPI()

    result = prediction_api.get_prediction(
        category_store_id=request.category_store_id,
        historical_data=request.historical_data,
        forecast_days=request.forecast_days,
    )

    return SuccessPredictionResponse(
        status=result['status'],
        timestamp='2026-01-08T12:00:00Z',
        category_store_id=request.category_store_id,
        forecast_days=request.forecast_days,
        predictions=result['predictions'],
        model_used=result['model_used'],
        statistics=result['statistics'],
        metadata=result['metadata'],
        report=result['report'],
    )


@router.post(
    '/prediction/package/',
)
async def package_prediction(request: Request):
    """Пакетное прогнозирование"""
    return Response(status_code=status.HTTP_200_OK)
