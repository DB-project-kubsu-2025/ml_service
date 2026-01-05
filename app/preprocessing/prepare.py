import pandas as pd
from datetime import timedelta
from typing import Optional, Tuple, List, Dict, Any, Callable
import logging

from app.types import HistoricalDataPoint

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Класс для предобработки временных рядов"""
    LAG_LIST = [1, 2, 3, 7, 14, 28]
    WINDOW_LIST = [7, 14, 28]

    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.feature_columns = feature_columns or []
        self.feature_order = feature_columns
        logger.info(f"Инициализирован препроцессор с {len(self.feature_columns)} признаками")

        self._feature_configs = self._get_feature_configs()

    def _get_feature_configs(self) -> Dict[str, Dict[str, Any]]:
        """Возвращает конфигурацию функций для создания признаков"""
        return {
            'lag': {
                'prefix': 'sales_lag_',
                'func': self._create_lag_feature,
                'requires_date': False
            },
            'rolling_mean': {
                'prefix': 'rolling_mean_',
                'func': self._create_rolling_mean_feature,
                'requires_date': False
            },
            'rolling_std': {
                'prefix': 'rolling_std_',
                'func': self._create_rolling_std_feature,
                'requires_date': False
            }
        }

    def _create_lag_feature(self, data: pd.DataFrame, feature_name: str,
                            target_col: str) -> pd.Series:
        """Создает лаговый признак"""
        try:
            lag = int(feature_name.split('_')[-1])
            return data[target_col].shift(lag)
        except (ValueError, IndexError) as e:
            logger.warning(f"Не удалось создать лаговый признак {feature_name}: {e}")
            return pd.Series(0.0, index=data.index)

    def _create_rolling_mean_feature(self, data: pd.DataFrame, feature_name: str,
                                     target_col: str) -> pd.Series:
        """Создает признак скользящего среднего"""
        try:
            window = int(feature_name.split('_')[-1])
            return data[target_col].rolling(window=window, min_periods=1).mean()
        except (ValueError, IndexError) as e:
            logger.warning(f"Не удалось создать rolling mean {feature_name}: {e}")
            return pd.Series(0.0, index=data.index)

    def _create_rolling_std_feature(self, data: pd.DataFrame, feature_name: str,
                                    target_col: str) -> pd.Series:
        """Создает признак скользящего стандартного отклонения"""
        try:
            window = int(feature_name.split('_')[-1])
            return data[target_col].rolling(window=window, min_periods=1).std()
        except (ValueError, IndexError) as e:
            logger.warning(f"Не удалось создать rolling std {feature_name}: {e}")
            return pd.Series(0.0, index=data.index)

    def _get_date_feature_func(self, feature_name: str) -> Optional[Callable]:
        """Возвращает функцию для создания признака на основе даты"""
        date_features = {
            'day_of_week': lambda dt: dt.dayofweek,
            'wday': lambda dt: dt.dayofweek,
            'day_of_month': lambda dt: dt.day,
            'month': lambda dt: dt.month,
            'year': lambda dt: dt.year,
            'is_weekend': lambda dt: dt.dayofweek >= 5,
            'is_month_start': lambda dt: dt.day == 1,
            'day_of_year': lambda dt: dt.dayofyear,
            'week_of_month': lambda dt: dt.day // 7
        }
        return date_features.get(feature_name)

    def set_features_from_model(self, model) -> List[str]:
        """Устанавливает признаки и их порядок из обученной модели"""
        features = []

        # Для XGBoost и scikit-learn >= 1.0
        if hasattr(model, 'feature_names_in_'):
            features = list(model.feature_names_in_)
            logger.info(f"Получено {len(features)} признаков из model.feature_names_in_")

        # Для LightGBM
        elif hasattr(model, 'feature_name_'):
            features = model.feature_name_
            logger.info(f"Получено {len(features)} признаков из model.feature_name_")

        # Для старых scikit-learn или если нет feature_names
        elif hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
            features = [f'feature_{i}' for i in range(n_features)]
            logger.info(f"Создано {len(features)} generic признаков из n_features_in_")

        self.feature_columns = features
        self.feature_order = features

        if features:
            logger.info(f"Установлен порядок признаков: {features[:5]}...")

        return features

    def _create_single_feature(self, data: pd.DataFrame, feature_name: str,
                               target_col: str = 'sales', date_col: str = 'date') -> pd.DataFrame:
        """Создает один признак по его имени"""

        # Проверка, является ли признак лаговым
        if feature_name.startswith('sales_lag_'):
            data[feature_name] = self._create_lag_feature(data, feature_name, target_col)

        # Проверка, является ли признак скользящим средним
        elif feature_name.startswith('rolling_mean_'):
            data[feature_name] = self._create_rolling_mean_feature(data, feature_name, target_col)

        # Проверка, является ли признак скользящим отклонением
        elif feature_name.startswith('rolling_std_'):
            data[feature_name] = self._create_rolling_std_feature(data, feature_name, target_col)

        # Проверка, является ли признак основанным на дате
        else:
            date_func = self._get_date_feature_func(feature_name)
            if date_func is not None and date_col in data.columns:
                try:
                    data[feature_name] = pd.to_datetime(data[date_col]).apply(date_func).astype(float)
                except Exception as e:
                    logger.warning(f"Не удалось создать признак даты {feature_name}: {e}")
                    data[feature_name] = 0.0
            else:
                data[feature_name] = 0.0
                if feature_name not in ['date', 'sales']:
                    logger.warning(f"Неизвестный тип признака: {feature_name}")

        return data

    def create_features(self, df: pd.DataFrame, target_col: str = 'sales') -> pd.DataFrame:
        """Создание признаков для временного ряда в порядке, указанном в feature_columns"""
        data = df.copy()

        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values('date').reset_index(drop=True)

        data[target_col] = data[target_col].astype(float)

        if not self.feature_columns:
            self.feature_columns = self._get_default_features()
            logger.info(f"Используется стандартный набор из {len(self.feature_columns)} признаков")

        for feature_name in self.feature_columns:
            data = self._create_single_feature(data, feature_name, target_col)

        data = data.fillna(0)

        for col in self.feature_columns:
            if col in data.columns:
                data[col] = data[col].astype(float)

        ordered_cols = ['date', target_col] + [f for f in self.feature_columns if f in data.columns]
        data = data.reindex(columns=ordered_cols)

        logger.info(f"Создано {len(self.feature_columns)} признаков в заданном порядке")
        return data

    def _get_default_features(self) -> List[str]:
        """Стандартный набор признаков (используется если не задан явно)"""
        features = []

        for lag in self.LAG_LIST:
            features.append(f'sales_lag_{lag}')

        for window in self.WINDOW_LIST:
            features.append(f'rolling_mean_{window}')
            features.append(f'rolling_std_{window}')

        features.extend([
            'day_of_week', 'day_of_month', 'month', 'year',
            'is_weekend', 'is_month_start', 'wday',
            'day_of_year', 'week_of_month'
        ])

        return features

    def prepare_for_prediction(self, historical_data: List[HistoricalDataPoint],
                               forecast_days: int = 14) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Подготовка исторических данных для прогнозирования"""
        df = pd.DataFrame(historical_data)

        if 'date' not in df.columns or 'sales' not in df.columns:
            raise ValueError("Данные должны содержать колонки 'date' и 'sales'")

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        historical_df = self.create_features(df, target_col='sales')

        last_date = historical_df['date'].max()
        hist_sales = historical_df['sales'].astype(float).values

        future_rows = []

        for i in range(forecast_days):
            future_date = last_date + timedelta(days=i + 1)
            row = {'date': future_date, 'sales': 0.0}

            for feature_name in self.feature_columns:
                # Получаем функцию для признака даты
                date_func = self._get_date_feature_func(feature_name)
                if date_func is not None:
                    try:
                        row[feature_name] = float(date_func(future_date))
                    except Exception:
                        row[feature_name] = 0.0

                # Обрабатка лаговых признаков
                elif feature_name.startswith('sales_lag_'):
                    try:
                        lag = int(feature_name.split('_')[-1])
                        if i + lag <= len(hist_sales):
                            row[feature_name] = float(hist_sales[-(i + lag)])
                        else:
                            row[feature_name] = 0.0
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Ошибка при создании лагового признака {feature_name}: {e}")
                        row[feature_name] = 0.0

                # Обработка скользящих статистик
                elif feature_name.startswith('rolling_'):
                    row[feature_name] = 0.0

                else:
                    row[feature_name] = 0.0

            future_rows.append(row)

        future_df = pd.DataFrame(future_rows).fillna(0.0)

        for col in future_df.columns:
            if col != 'date' and pd.api.types.is_numeric_dtype(future_df[col]):
                future_df[col] = future_df[col].astype(float)

        historical_cols = historical_df.columns.tolist()
        future_df = future_df.reindex(columns=historical_cols)

        logger.info(f"Подготовлено {len(future_df)} дней для прогноза")
        return historical_df, future_df

    def get_features_for_prediction(self, historical_df: pd.DataFrame,
                                    future_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Получает только признаки (без date и sales) в правильном порядке"""
        feature_cols = [
            col for col in historical_df.columns if col not in ['date', 'sales']
        ]

        if self.feature_order:
            ordered_features = [feature for feature in self.feature_order if feature in feature_cols]
            other_features = [feature for feature in feature_cols if feature not in ordered_features]
            feature_cols = ordered_features + other_features

        X_historical = historical_df[feature_cols]
        X_future = future_df[feature_cols]

        return X_historical, X_future