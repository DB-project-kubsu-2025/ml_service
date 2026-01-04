from typing import Dict, List, Any
import logging
from datetime import datetime

from ..model.predictor import SalesPredictor
from ..model.loader import ModelLoader
from ..postprocessing.format import ResultFormatter

logger = logging.getLogger(__name__)


class PredictionAPI:
    """API сервис для прогнозирования"""

    def __init__(self):
        self.predictor = SalesPredictor()
        self.loader = ModelLoader()
        self.formatter = ResultFormatter()


    def get_prediction(self, category_store_id: str,
                       historical_data: List[Dict[str, float]],
                       forecast_days: int = 7) -> Dict[str, Any]:
        """Получение прогноза для одной категории"""
        logger.info(f"API: Получение прогноза для {category_store_id}")

        if not historical_data:
            return self._error_response("Исторические данные не предоставлены")

        if len(historical_data) < 28:
            logger.warning(f"Мало исторических данных: {len(historical_data)} дней")

        result = self.predictor.predict(
            category_store_id=category_store_id,
            historical_data=historical_data,
            forecast_days=forecast_days
        )

        if result.get('status') == 'success':
            model_info = self.loader.load_model(category_store_id)[1]
            report = self.formatter.create_report(
                result['predictions'],
                model_info
            )
            result['report'] = report

        return result


    def get_batch_predictions(self, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Пакетное прогнозирование"""
        logger.info(f"API: Пакетное прогнозирование для {len(requests)} запросов")

        results = {}
        for req in requests:
            try:
                cat_id = req.get('category_store_id')
                hist_data = req.get('historical_data', [])
                days = req.get('forecast_days', 14)

                if not cat_id:
                    results[f"request_{len(results)}"] = self._error_response("Отсутствует category_store_id")
                    continue

                result = self.predictor.predict(cat_id, hist_data, days)
                results[cat_id] = result

            except Exception as e:
                results[req.get('category_store_id', f"request_{len(results)}")] = self._error_response(str(e))

        formatted_results = self.formatter.format_batch_results(results)

        return {
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'results': formatted_results,
            'raw_results': results
        }


    def get_available_models(self) -> Dict[str, Any]:
        """Получение списка доступных моделей"""
        models = self.loader.list_models()

        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'total_models': len(models),
            'models': models
        }


    def get_model_info(self, category_store_id: str) -> Dict[str, Any]:
        """Получение информации о модели"""
        model, metadata = self.loader.load_model(category_store_id)

        if model is None:
            return self._error_response(f"Модель не найдена: {category_store_id}")

        return {
            'status': 'success',
            'category_store_id': category_store_id,
            'model_type': type(model).__name__,
            'metadata': metadata,
            'is_loaded_in_cache': category_store_id in self.predictor.loader.models_cache
        }


    def health_check(self) -> Dict[str, Any]:
        """Проверка работоспособности"""
        try:
            models_count = len(self.loader.list_models())

            return {
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
        except Exception as e:
            return self._error_response(f"Health check failed: {e}")


    def _error_response(self, error_msg: str) -> Dict[str, Any]:
        """Создание ответа об ошибке"""
        return {
            'status': 'error',
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }