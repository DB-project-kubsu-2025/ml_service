import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

from app.types import (
    BatchPredictionResponse, ModelMetadata, BatchPredictionSummary,
    BatchPredictionResultItem
)

logger = logging.getLogger(__name__)


class ResultFormatter:
    """Форматирование результатов"""

    STATUS_OF_SUCCESS = 'success'

    def __init__(self):
        pass


    def format_batch_results(self, results: Dict[str, Dict]) -> BatchPredictionResponse:
        """Форматирование пакетных результатов"""
        successful: List[BatchPredictionResultItem] = []
        failed: List[BatchPredictionResultItem] = []

        for cat_id, result in results.items():
            if result.get('status') == self.STATUS_OF_SUCCESS:
                # Создание успешного результата с нужными полями
                successful_item: BatchPredictionResultItem = {
                    'category_store_id': cat_id,
                    'forecast_days': result.get('forecast_days', 0),
                    'total_predicted': sum(p['predicted_sales'] for p in result.get('predictions', []))
                }
                successful.append(successful_item)
            else:
                # Создание результата с ошибкой
                failed_item: BatchPredictionResultItem = {
                    'category_store_id': cat_id,
                    'error': result.get('error', 'Unknown error')
                }
                failed.append(failed_item)

        summary: BatchPredictionSummary = {
            'total': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(results) * 100 if results else 0
        }

        response: BatchPredictionResponse = {
            'summary': summary,
            'successful_predictions': successful,
            'failed_predictions': failed
        }

        return response


    def create_report(self, predictions: List[Dict],
                      model_info: ModelMetadata) -> Dict[str, Any]:
        """Создание отчета по прогнозу"""
        if not predictions:
            return {}

        sales_values = [p['predicted_sales'] for p in predictions]

        report = {
            'forecast_period': {
                'start_date': predictions[0]['date'],
                'end_date': predictions[-1]['date'],
                'days': len(predictions)
            },
            'statistics': {
                'total_sales': float(np.sum(sales_values)),
                'average_daily': float(np.mean(sales_values)),
                'min_daily': float(np.min(sales_values)),
                'max_daily': float(np.max(sales_values)),
                'std_daily': float(np.std(sales_values))
            },
            'model_info': {
                'type': model_info.get('model_type', 'unknown'),
                'trained_date': model_info.get('saved_at', 'unknown')
            }
        }

        if len(sales_values) >= 7:
            weekly_totals = []
            for i in range(0, len(sales_values), 7):
                week = sales_values[i:i + 7]
                if len(week) == 7:
                    weekly_totals.append(sum(week))

            if len(weekly_totals) > 1:
                weekly_growth = []
                for i in range(1, len(weekly_totals)):
                    growth = ((weekly_totals[i] - weekly_totals[i - 1]) / weekly_totals[i - 1]) * 100
                    weekly_growth.append(float(growth))

                report['weekly_trends'] = {
                    'weekly_totals': weekly_totals,
                    'weekly_growth_rates': weekly_growth
                }

        return report


    def to_dataframe(self, predictions: List[Dict]) -> pd.DataFrame:
        """Конвертация прогнозов в DataFrame"""
        if not predictions:
            return pd.DataFrame()

        data = []
        for pred in predictions:
            row = {
                'date': pred['date'],
                'predicted_sales': pred['predicted_sales']
            }
            data.append(row)

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df