import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
from datetime import datetime, timedelta

from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso

from app.preprocessing.prepare import DataPreprocessor
from app.model.loader import ModelLoader
from app.postprocessing.format import ResultFormatter
from app.schemas import HistoricalDataPoint, DataForPredictionItem
from app.types import (
    PredictionResponse, ErrorResponse, HealthCheckResponse,
    ModelInfoResponse, BatchPredictionAPIResponse, SuccessPredictionResponse, PredictionPoint,
)

logger = logging.getLogger(__name__)


class SalesPredictor:
    """Основной класс для прогнозирования"""

    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.loader = ModelLoader()

    def predict(self, category_store_id: str,
                historical_data: List[HistoricalDataPoint],
                forecast_days: int = 14) -> PredictionResponse:
        """Прогнозирование продаж"""
        logger.info(f"Прогнозирование для {category_store_id}")

        model, metadata = self.loader.load_model(category_store_id)
        if model is None:
            error_response: ErrorResponse = {
                'status': 'error',
                'error': f"Модель не найдена: {category_store_id}",
                'timestamp': datetime.now().isoformat()
            }
            return error_response

        model_features = self.preprocessor.set_features_from_model(model)

        if not model_features:
            logger.warning(f"Не удалось получить признаки из модели {category_store_id}")
            if metadata and 'feature_columns' in metadata:
                model_features = metadata['feature_columns']
                self.preprocessor.feature_columns = model_features
                self.preprocessor.feature_order = model_features
                logger.info(f"Используем признаки из метаданных: {len(model_features)}")

        historical_df, future_df = self.preprocessor.prepare_for_prediction(
            historical_data, forecast_days
        )

        X_historical, X_future = self.preprocessor.get_features_for_prediction(
            historical_df, future_df
        )
        logger.info(f"Используется {len(X_future.columns)} признаков для прогноза")

        if hasattr(model, 'n_features_in_'):
            expected_features = model.n_features_in_
            actual_features = len(X_future.columns)
            if expected_features != actual_features:
                logger.warning(f'Количество признаков не совпадает: '
                               f'ожидалось {expected_features}, '
                               f'получено {actual_features}')

        try:
            if isinstance(model, lgb.LGBMRegressor):
                predictions = model.predict(X_future, predict_disable_shape_check=True)
                logger.info('Использована модель LightGBM с отключенной проверкой формы')

            elif isinstance(model, XGBRegressor):
                if hasattr(model, 'feature_names_in_'):
                    model_order = list(model.feature_names_in_)
                    data_order = list(X_future.columns)

                    if model_order != data_order:
                        logger.warning('Порядок признаков не совпадает, пытаемся исправить...')
                        try:
                            X_future = X_future[model_order]
                            logger.info('Порядок признаков успешно исправлен')
                        except KeyError as e:
                            logger.error(f'Не удалось привести порядок признаков: {e}')
                            missing = set(model_order) - set(data_order)
                            for feature in missing:
                                X_future[feature] = 0.0
                            X_future = X_future[model_order]
                            logger.warning(f'Добавлены отсутствующие признаки: {missing}')

                predictions = model.predict(X_future)
                logger.info('Использована модель XGBoost')

            elif isinstance(model, RandomForestRegressor):
                predictions = model.predict(X_future)
                logger.info('Использована модель RandomForest')

            elif isinstance(model, GradientBoostingRegressor):
                predictions = model.predict(X_future)
                logger.info('Использована модель GradientBoosting')

            elif isinstance(model, Ridge):
                predictions = model.predict(X_future)
                logger.info('Использована модель Ridge')

            elif isinstance(model, Lasso):
                predictions = model.predict(X_future)
                logger.info('Использована модель Lasso')

            else:
                predictions = model.predict(X_future)
                model_type = type(model).__name__
                logger.info(f'Использована модель неизвестного типа: {model_type}')

            predictions = np.maximum(predictions, 0)
            result = self._format_predictions(
                category_store_id, predictions, forecast_days, model, metadata
            )

            logger.info(f"Прогноз завершен для {category_store_id}")
            return result

        except Exception as e:
            logger.error(f"Ошибка прогнозирования для {category_store_id}: {e}")
            error_response: ErrorResponse = {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return error_response

    def evaluate_model(self, category_store_id: str,
                       test_data: pd.DataFrame) -> Dict[str, float]:
        """Оценка модели на тестовых данных"""

        model, metadata = self.loader.load_model(category_store_id)
        if model is None:
            return {}

        self.preprocessor.set_features_from_model(model)

        data = self.preprocessor.create_features(test_data, target_col='sales')

        feature_cols = [col for col in data.columns if col not in ['date', 'sales']]
        X = data[feature_cols]
        y = data['sales']

        if isinstance(model, lgb.LGBMRegressor):
            y_pred = model.predict(X, predict_disable_shape_check=True)
        else:
            y_pred = model.predict(X)

        metrics = {
            'MAE': float(mean_absolute_error(y, y_pred)),
            'RMSE': float(np.sqrt(np.mean((y - y_pred) ** 2))),
            'MAPE': float(np.mean(np.abs((y - y_pred) / np.maximum(y, 1))) * 100)
        }

        return metrics

    def _format_predictions(self, category_store_id: str,
                            predictions: np.ndarray,
                            forecast_days: int,
                            model: Any,
                            metadata: Dict) -> SuccessPredictionResponse:
        """Форматирование результатов прогноза"""

        predictions = np.maximum(predictions, 0)
        today = datetime.now()

        predictions_list: List[PredictionPoint] = []
        for i, pred in enumerate(predictions):
            date = (today + timedelta(days=i + 1)).strftime('%Y-%m-%d')
            predictions_list.append({
                'date': date,
                'predicted_sales': float(np.float64(pred))
            })

        pred_values = [prediction['predicted_sales'] for prediction in predictions_list]

        result_dict: Dict[str, Any] = {
            'status': 'success',
            'timestamp': today.isoformat(),
            'category_store_id': category_store_id,
            'forecast_days': forecast_days,
            'predictions': predictions_list,
            'model_used': type(model).__name__,
            'statistics': {
                'mean': float(np.mean(pred_values)),
                'min': float(np.min(pred_values)),
                'max': float(np.max(pred_values)),
                'total': float(np.sum(pred_values))
            },
            'metadata': None,
            'report': None
        }

        if metadata:
            result_dict['metadata'] = {
                'model_type': metadata.get('model_type', type(model).__name__),
                'trained_date': metadata.get('saved_at', 'unknown'),
                'performance_metrics': metadata.get('performance_metrics', {})
            }

        result: SuccessPredictionResponse = {
            'status': result_dict['status'],
            'timestamp': result_dict['timestamp'],
            'category_store_id': result_dict['category_store_id'],
            'forecast_days': result_dict['forecast_days'],
            'predictions': result_dict['predictions'],
            'model_used': result_dict['model_used'],
            'statistics': result_dict['statistics'],
            'metadata': result_dict['metadata'],
            'report': result_dict['report']
        }

        return result

class PredictionAPI:
    """API сервис для прогнозирования"""

    NUMBER_OF_HISTORICAL_DATA = 28
    STATUS_OF_SUCCESS = 'success'

    def     __init__(self):
        self.predictor = SalesPredictor()
        self.loader = ModelLoader()
        self.formatter = ResultFormatter()


    def get_prediction(self, category_store_id: str,
                       historical_data: List[HistoricalDataPoint],
                       forecast_days: int = 7) -> PredictionResponse:
        """Получение прогноза для одной категории"""

        logger.info(f"API: Получение прогноза для {category_store_id}")

        if not historical_data:
            error_response: ErrorResponse = {
                'status': 'error',
                'error': "Исторические данные не предоставлены",
                'timestamp': datetime.now().isoformat()
            }
            return error_response

        if len(historical_data) < self.NUMBER_OF_HISTORICAL_DATA:
            logger.warning(f"Мало исторических данных: {len(historical_data)} дней")

        result = self.predictor.predict(
            category_store_id=category_store_id,
            historical_data=historical_data,
            forecast_days=forecast_days
        )

        if result.get('status') == self.STATUS_OF_SUCCESS:
            model_info = self.loader.load_model(category_store_id)[1]
            report = self.formatter.create_report(
                result['predictions'],
                model_info
            )
            result['report'] = report

        return result

    def get_batch_predictions(self, requests: list[DataForPredictionItem]) -> BatchPredictionAPIResponse:
        """Пакетное прогнозирование"""
        logger.info(f"API: Пакетное прогнозирование для {len(requests)} запросов")

        results = {}
        for req in requests:
            cat_id = req.get('category_store_id')

            if not cat_id:
                results[f"request_{len(results)}"] = {
                    'status': 'error',
                    'error': "Отсутствует category_store_id",
                    'timestamp': datetime.now().isoformat()
                }
                continue

            hist_data = req.get('historical_data', [])
            days = req.get('forecast_days', 14)

            try:
                result = self.predictor.predict(cat_id, hist_data, days)
                results[cat_id] = result

            except Exception as e:
                results[req.get('category_store_id', f"request_{len(results)}")] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }

        formatted_results = self.formatter.format_batch_results(results)

        response: BatchPredictionAPIResponse = {
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'results': formatted_results,
            'raw_results': results
        }

        return response

    def get_model_info(self, category_store_id: str) -> ModelInfoResponse:
        """Получение информации о модели"""
        model, metadata = self.loader.load_model(category_store_id)

        if model is None:
            error_response: ModelInfoResponse = {
                'status': 'error',
                'error': f"Модель не найдена: {category_store_id}",
                'timestamp': datetime.now().isoformat(),
                'category_store_id': category_store_id,
                'model_type': 'unknown',
                'metadata': {},
                'is_loaded_in_cache': False
            }
            return error_response

        response: ModelInfoResponse = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'category_store_id': category_store_id,
            'model_type': type(model).__name__,
            'metadata': metadata,
            'is_loaded_in_cache': category_store_id in self.predictor.loader.models_cache
        }

        return response

    def health_check(self) -> HealthCheckResponse:
        """Проверка работоспособности, возвращает статус в виде словаря"""
        try:
            models_count = len(self.loader.list_models())

            response: HealthCheckResponse = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'model_loader': 'operational',
                    'predictor': 'operational',
                    'formatter': 'operational'
                },
                'metrics': {
                    'available_models': models_count,
                    'cache_size': len(self.loader.models_cache)
                }
            }
            return response

        except Exception as e:
            error_response: HealthCheckResponse = {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'components': {},
                'metrics': {},
                'error': f"Health check failed: {e}"
            }
            return error_response
