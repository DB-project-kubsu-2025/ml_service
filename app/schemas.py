import datetime
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field


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
    date: str
    value: float


class PredictionStatistics(BaseModel):
    mae: float
    rmse: float


class SuccessPredictionResponse(BaseModel):
    status: str = Field(examples=['success'])
    timestamp: str
    category_store_id: str
    forecast_days: int
    predictions: list[PredictionPoint]
    model_used: str
    statistics: PredictionStatistics
    metadata: Optional[Dict[str, Any]] = None
    report: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    error: str = Field(examples=['Invalid input data'])
