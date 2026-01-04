import pandas as pd
from datetime import timedelta
from typing import Optional, Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Класс для предобработки временных рядов"""

    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.feature_columns = feature_columns or []
        self.feature_order = feature_columns  # Сохраняем порядок признаков
        logger.info(f"Инициализирован препроцессор с {len(self.feature_columns)} признаками")


    def set_features_from_model(self, model) -> None:
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

        if feature_name.startswith('sales_lag_'):
            try:
                lag = int(feature_name.split('_')[-1])
                data[feature_name] = data[target_col].shift(lag)
            except:
                data[feature_name] = 0.0

        elif feature_name.startswith('rolling_mean_'):
            try:
                window = int(feature_name.split('_')[-1])
                data[feature_name] = data[target_col].rolling(window=window, min_periods=1).mean()
            except:
                data[feature_name] = 0.0

        elif feature_name.startswith('rolling_std_'):
            try:
                window = int(feature_name.split('_')[-1])
                data[feature_name] = data[target_col].rolling(window=window, min_periods=1).std()
            except:
                data[feature_name] = 0.0

        elif feature_name in ['day_of_week', 'wday']:
            if date_col in data.columns:
                data[feature_name] = pd.to_datetime(data[date_col]).dt.dayofweek.astype(int)
            else:
                data[feature_name] = 0

        elif feature_name == 'day_of_month':
            if date_col in data.columns:
                data[feature_name] = pd.to_datetime(data[date_col]).dt.day.astype(int)
            else:
                data[feature_name] = 0

        elif feature_name == 'month':
            if date_col in data.columns:
                data[feature_name] = pd.to_datetime(data[date_col]).dt.month.astype(int)
            else:
                data[feature_name] = 0

        elif feature_name == 'year':
            if date_col in data.columns:
                data[feature_name] = pd.to_datetime(data[date_col]).dt.year.astype(int)
            else:
                data[feature_name] = 0

        elif feature_name == 'is_weekend':
            if date_col in data.columns:
                data[feature_name] = (pd.to_datetime(data[date_col]).dt.dayofweek >= 5).astype(int)
            else:
                data[feature_name] = 0

        elif feature_name == 'is_month_start':
            if date_col in data.columns:
                data[feature_name] = (pd.to_datetime(data[date_col]).dt.day == 1).astype(int)
            else:
                data[feature_name] = 0

        elif feature_name == 'day_of_year':
            if date_col in data.columns:
                data[feature_name] = pd.to_datetime(data[date_col]).dt.dayofyear.astype(int)
            else:
                data[feature_name] = 0

        elif feature_name == 'week_of_month':
            if date_col in data.columns:
                data[feature_name] = (pd.to_datetime(data[date_col]).dt.day // 7).astype(int)
            else:
                data[feature_name] = 0

        else:
            data[feature_name] = 0.0
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

        for lag in [1, 2, 3, 7, 14, 28]:
            features.append(f'sales_lag_{lag}')

        for window in [7, 14, 28]:
            features.append(f'rolling_mean_{window}')
            features.append(f'rolling_std_{window}')

        features.extend([
            'day_of_week', 'day_of_month', 'month', 'year',
            'is_weekend', 'is_month_start', 'wday',
            'day_of_year', 'week_of_month'
        ])

        return features


    def prepare_for_prediction(self, historical_data: List[Dict[str, Any]],
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
                if feature_name in ['day_of_week', 'wday']:
                    row[feature_name] = float(future_date.dayofweek)
                elif feature_name == 'day_of_month':
                    row[feature_name] = float(future_date.day)
                elif feature_name == 'month':
                    row[feature_name] = float(future_date.month)
                elif feature_name == 'year':
                    row[feature_name] = float(future_date.year)
                elif feature_name == 'is_weekend':
                    row[feature_name] = float(future_date.dayofweek >= 5)
                elif feature_name == 'is_month_start':
                    row[feature_name] = float(future_date.day == 1)
                elif feature_name == 'day_of_year':
                    row[feature_name] = float(future_date.dayofyear)
                elif feature_name == 'week_of_month':
                    row[feature_name] = float(future_date.day // 7)
                elif feature_name.startswith('sales_lag_'):
                    try:
                        lag = int(feature_name.split('_')[-1])
                        if i + lag <= len(hist_sales):
                            row[feature_name] = float(hist_sales[-(i + lag)])
                        else:
                            row[feature_name] = 0.0
                    except:
                        row[feature_name] = 0.0
                elif feature_name.startswith('rolling_'):
                    row[feature_name] = 0.0
                else:
                    row[feature_name] = 0.0

            future_rows.append(row)

        future_df = pd.DataFrame(future_rows)

        future_df = future_df.fillna(0.0)

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
        feature_cols = [col for col in historical_df.columns
                        if col not in ['date', 'sales']]

        if self.feature_order:
            ordered_features = [f for f in self.feature_order if f in feature_cols]
            other_features = [f for f in feature_cols if f not in ordered_features]
            feature_cols = ordered_features + other_features

        X_historical = historical_df[feature_cols]
        X_future = future_df[feature_cols]

        return X_historical, X_future

    def split_data(self, data: pd.DataFrame, target_col: str = 'sales',
                   test_size: int = 28) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Разделение данных на обучающую и тестовую выборки"""
        if len(data) < test_size + 14:
            raise ValueError(f"Недостаточно данных. Всего: {len(data)}, нужно минимум: {test_size + 14}")

        split_idx = len(data) - test_size

        feature_cols = [col for col in data.columns if col not in ['date', target_col]]

        if self.feature_order:
            ordered_features = [f for f in self.feature_order if f in feature_cols]
            other_features = [f for f in feature_cols if f not in ordered_features]
            feature_cols = ordered_features + other_features

        X = data[feature_cols]
        y = data[target_col]

        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        logger.info(f"Данные разделены: train={len(X_train)}, test={len(X_test)}")

        return X_train, X_test, y_train, y_test