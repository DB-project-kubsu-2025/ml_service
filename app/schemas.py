import datetime
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field

from app.types import BatchPredictionSummary


class HistoricalDataPoint(BaseModel):
    date: str = Field(examples=['2024-01-01'])
    sales: float = Field(examples=[123.45])


def historical_example(start: datetime.date, days: int = 28):
    return [
        {
            'date': (start + datetime.timedelta(days=i)).isoformat(),
            'sales': round(100 + i * 2.5, 2),
        }
        for i in range(days)
    ]


class PredictionRequest(BaseModel):
    category_store_id: str
    historical_data: list[HistoricalDataPoint]
    forecast_days: int

    model_config = {
        'json_schema_extra': {
            'example': {
                'category_store_id': 'FOODS/CA/1',
                'forecast_days': 7,
                'historical_data': historical_example(
                    start=datetime.date(2024, 1, 1),
                    days=28,
                ),
            },
        },
    }


class PredictionPoint(BaseModel):
    date: Optional[str] = None
    value: Optional[float] = None


class PredictionStatistics(BaseModel):
    mae: Optional[float] = None
    rmse: Optional[float] = None


class DataForPredictionItem(BaseModel):
    category_store_id: str = Field(examples=['FOODS/CA/1'])
    historical_data: List[HistoricalDataPoint]
    forecast_days: int = Field(ge=1, le=365, examples=[7])


class DataForPrediction(BaseModel):
    data_for_prediction_items: list[DataForPredictionItem] = Field()

    model_config = {
        'json_schema_extra': {
            'example': {
                'data_for_prediction_items': [
                    {
                        'category_store_id': 'FOODS/CA/1',
                        'forecast_days': 7,
                        'historical_data': historical_example(datetime.date(2024, 1, 1)),
                    },
                ],
            },
        },
    }


class SuccessPredictionResponse(BaseModel):
    status: Optional[str] = None
    timestamp: Optional[str] = None

    category_store_id: Optional[str] = None
    forecast_days: Optional[int] = None

    predictions: Optional[list[PredictionPoint]] = None
    model_used: Optional[str] = None
    statistics: Optional[PredictionStatistics] = None
    metadata: Optional[Dict[str, Any]] = None
    report: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    error: str = Field(examples=['Invalid input data'])


class BatchPredictionResultItem(BaseModel):
    category_store_id: Optional[str] = Field(
        default=None,
        examples=['FOODS/CA/1'],
    )
    forecast_days: Optional[int] = Field(
        default=None,
        examples=[7],
    )
    total_predicted: Optional[float] = Field(
        default=None,
        examples=[12345.67],
        description='Суммарный прогноз продаж',
    )
    error: Optional[str] = Field(
        default=None,
        examples=['Not enough historical data'],
        description='Описание ошибки, если прогноз не выполнен',
    )


class BatchPredictionResponse(BaseModel):
    summary: BatchPredictionSummary = Field(
        description='Сводная статистика по пакету',
    )
    successful_predictions: List[BatchPredictionResultItem] = Field(
        default_factory=list,
        description='Успешные прогнозы',
    )
    failed_predictions: List[BatchPredictionResultItem] = Field(
        default_factory=list,
        description='Неуспешные прогнозы',
    )


class BatchPredictionAPIResponse(BaseModel):
    status: str = Field(
        default=None,
        examples=['success'],
        description='Статус выполнения пакетного прогноза',
    )
    timestamp: str = Field(
        default=None,
        description='Время формирования ответа',
    )
    results: BatchPredictionResponse = Field(
        default=None,
        description='Агрегированные результаты пакетного прогнозирования',
    )
    raw_results: Dict[str, SuccessPredictionResponse] = Field(
        default=None,
        description='Сырые результаты по каждой категории',
    )
