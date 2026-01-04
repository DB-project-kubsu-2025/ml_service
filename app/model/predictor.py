import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
from datetime import datetime, timedelta

from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

from app.preprocessing.prepare import DataPreprocessor
from app.model.loader import ModelLoader

logger = logging.getLogger(__name__)


class SalesPredictor:
    """Основной класс для прогнозирования"""

    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.loader = ModelLoader()


    def predict(self, category_store_id: str,
                historical_data: List[Dict[str, float]],
                forecast_days: int = 14) -> Dict[str, Any]:
        """Прогнозирование продаж"""
        try:
            logger.info(f"Прогнозирование для {category_store_id}")

            model, metadata = self.loader.load_model(category_store_id)
            if model is None:
                return self._create_error_response(f"Модель не найдена: {category_store_id}")

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
                    logger.warning(f"Количество признаков не совпадает: "
                                   f"ожидалось {expected_features}, "
                                   f"получено {actual_features}")

            if isinstance(model, lgb.LGBMRegressor):
                predictions = model.predict(X_future, predict_disable_shape_check=True)
            elif isinstance(model, XGBRegressor):
                if hasattr(model, 'feature_names_in_'):
                    model_order = list(model.feature_names_in_)
                    data_order = list(X_future.columns)

                    if model_order != data_order:
                        logger.warning("Порядок признаков не совпадает, пытаемся исправить...")
                        try:
                            X_future = X_future[model_order]
                        except KeyError as e:
                            logger.error(f"Не удалось привести порядок признаков: {e}")
                            missing = set(model_order) - set(data_order)
                            for feature in missing:
                                X_future[feature] = 0.0
                            X_future = X_future[model_order]

                predictions = model.predict(X_future)
            else:
                predictions = model.predict(X_future)

            result = self._format_predictions(
                category_store_id, predictions, forecast_days, model, metadata
            )

            logger.info(f"Прогноз завершен для {category_store_id}")
            return result

        except Exception as e:
            logger.error(f"Ошибка прогнозирования для {category_store_id}: {e}")
            return self._create_error_response(str(e))


    def evaluate_model(self, category_store_id: str,
                       test_data: pd.DataFrame) -> Dict[str, float]:
        """Оценка модели на тестовых данных"""
        try:
            model, metadata = self.loader.load_model(category_store_id)
            if model is None:
                return {}

            self.preprocessor.set_features_from_model(model)

            df = self.preprocessor.create_features(test_data, target_col='sales')

            feature_cols = [col for col in df.columns if col not in ['date', 'sales']]
            X = df[feature_cols]
            y = df['sales']

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

        except Exception as e:
            logger.error(f"Ошибка оценки модели: {e}")
            return {}


    def _format_predictions(self, category_store_id: str,
                            predictions: np.ndarray,
                            forecast_days: int,
                            model: Any,
                            metadata: Dict) -> Dict[str, Any]:
        """Форматирование результатов прогноза"""
        today = datetime.now()

        predictions_list = []
        for i, pred in enumerate(predictions):
            date = (today + timedelta(days=i + 1)).strftime('%Y-%m-%d')
            predictions_list.append({
                'date': date,
                'predicted_sales': float(pred)
            })

        pred_values = [p['predicted_sales'] for p in predictions_list]

        result = {
            'category_store_id': category_store_id,
            'forecast_days': forecast_days,
            'predictions': predictions_list,
            'model_used': type(model).__name__,
            'generated_at': today.isoformat(),
            'statistics': {
                'mean': float(np.mean(pred_values)),
                'min': float(np.min(pred_values)),
                'max': float(np.max(pred_values)),
                'total': float(np.sum(pred_values))
            },
            'status': 'success'
        }

        if metadata:
            result['metadata'] = {
                'model_type': metadata.get('model_type', type(model).__name__),
                'trained_date': metadata.get('saved_at', 'unknown'),
                'performance_metrics': metadata.get('performance_metrics', {})
            }

        return result


    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Создание ответа с ошибкой"""
        return {
            'status': 'error',
            'error': error_msg,
            'generated_at': datetime.now().isoformat()
        }