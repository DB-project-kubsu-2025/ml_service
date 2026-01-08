import datetime

from fastapi import APIRouter, Response, status, Request

from app.model.predictor import PredictionAPI
from app.schemas import ErrorResponse, SuccessPredictionResponse, PredictionRequest, DataForPrediction, \
    BatchPredictionAPIResponse

router = APIRouter()


@router.post(
    '/prediction/one_category/',
    summary='Прогноз продаж по одной категории',
    tags=['Prediction'],
    response_model=SuccessPredictionResponse,
)
async def prediction_by_one_category(request: PredictionRequest) -> SuccessPredictionResponse:
    """Прогнозирование по одной категории"""
    result = PredictionAPI().get_prediction(
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
    "/prediction/package/",
    summary="Прогноз продаж по всем категориям",
    tags=["Prediction"],
    response_model=SuccessPredictionResponse,
)
async def package_prediction(data_for_prediction: DataForPrediction) -> BatchPredictionAPIResponse:
    """Пакетное прогнозирование"""
    result = PredictionAPI().get_batch_predictions(data_for_prediction.model_dump()['data_for_prediction_items'])
    return BatchPredictionAPIResponse(
        status=result['status'],
        timestamp=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'),
        results=result['results'],
        raw_results=result['raw_results'],
    )
