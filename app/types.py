from typing import TypedDict, List, Optional, Dict, Any, Union
from datetime import datetime
import numpy as np
import pandas as pd



class PredictionPoint(TypedDict):
    """Тип для точки прогноза"""
    date: str
    predicted_sales: float


class PredictionStatistics(TypedDict):
    """Тип для статистики прогноза"""
    mean: float
    min: float
    max: float
    total: float


class ModelMetadata(TypedDict, total=False):
    """Тип для метаданных модели"""
    model_type: str
    category_store_id: str
    saved_at: str
    loaded_at: str
    performance_metrics: Dict[str, float]
    feature_columns: List[str]
    model_file: str


class BaseAPIResponse(TypedDict, total=False):
    """Базовый тип для ответов API"""
    status: str
    timestamp: str


class ErrorResponse(BaseAPIResponse):
    """Тип для ответа с ошибкой"""
    error: str


class SuccessPredictionResponse(TypedDict):
    """Тип для успешного ответа прогнозирования (все поля обязательны)"""
    status: str
    timestamp: str
    category_store_id: str
    forecast_days: int
    predictions: List[PredictionPoint]
    model_used: str
    statistics: PredictionStatistics
    metadata: Optional[Dict[str, Any]]
    report: Optional[Dict[str, Any]]


# Объединенный тип для ответа прогноза (может быть успешным или с ошибкой)
PredictionResponse = Union[SuccessPredictionResponse, ErrorResponse]


class HealthCheckResponse(TypedDict, total=False):
    """Тип для ответа проверки работоспособности"""
    status: str
    timestamp: str
    components: Dict[str, str]
    metrics: Dict[str, Union[int, float]]
    error: str


class ModelInfoResponse(TypedDict, total=False):
    """Тип для информации о модели"""
    status: str
    timestamp: str
    category_store_id: str
    model_type: str
    metadata: Dict[str, Any]
    is_loaded_in_cache: bool
    error: str


class BatchPredictionSummary(TypedDict):
    """Сводка по пакетному прогнозированию"""
    total: int
    successful: int
    failed: int
    success_rate: float


class BatchPredictionResultItem(TypedDict, total=False):
    """Элемент результата пакетного прогнозирования"""
    category_store_id: str
    forecast_days: int
    total_predicted: float
    error: str


class BatchPredictionResponse(TypedDict):
    """Тип для форматированного ответа пакетного прогнозирования"""
    summary: BatchPredictionSummary
    successful_predictions: List[BatchPredictionResultItem]
    failed_predictions: List[BatchPredictionResultItem]


class BatchPredictionAPIResponse(TypedDict, total=False):
    """Тип для ответа API пакетного прогнозирования"""
    status: str
    timestamp: str
    results: BatchPredictionResponse
    raw_results: Dict[str, Any]
