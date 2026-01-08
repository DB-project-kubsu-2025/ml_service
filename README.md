# ml_service

## Системные требования
- python 3.12+
- Docker

## Настройка тестового окружения для разработки
- Разархивировать metadata.pkl.gz в папку `app/model/artifacts` (перед этим ее надо создать)
- Создать виртуальное окружение python
```bash
python -m venv venv
```
- Активировать его 
```bash 
.\venv\Scripts\activate
```
- Установить poetry
```bash
python -m pip install poetry==2.1.3
```
- Установить зависимости
```bash
poetry install --no-root
```
- создать docker-network (если ранее не была создана)
```bash
docker network create db_project_network
```
- запустить приложение
```bash
docker compose up
```
- по адресу `<ip-адрес>:8071/swagger` можно проверить, что все работает как надо

## Настройка интерпретатора для запуска через PyCharm
- В настройках (Ctrl + Alt + S) найти `Project ml_service` и перейти в `Python interpreter`
- Нажать `Add interpreter` -> `on docker compose`
- Прокликать оставшиеся кнопки, не меняя никаких настроек, дождаться настройки интерпретатора
- Справа сверху PyCharm-а настроить конфигурацию запуска, выбрав `fastapi` и в `Python interpreter` указать созданный интерпретатор в списке

## Пример и описание возвращаемого словаря в format.py в методе format_batch_result
-  Метод агрегирует результаты пакетного прогнозирования, создавая сводную статистику выполнения всех запросов. 
Преобразует детальные прогнозы в компактный формат для быстрого анализа.
- Рассмотрим пример ответа, в котором все модели отработали без перебоев:
```json
{
"summary": {
      "total": 5,
      "successful": 5,
      "failed": 0,
      "success_rate": 100.0
    },
    "successful_predictions": [
      {
        "category_store_id": "FOODS/CA/1",
        "forecast_days": 14,
        "total_predicted": 26335.586303710938
      },
      {
        "category_store_id": "FOODS/CA/2",
        "forecast_days": 14,
        "total_predicted": 289.1878730806665
      },
      {
        "category_store_id": "FOODS/CA/3",
        "forecast_days": 14,
        "total_predicted": 35410.791015625
      },
      {
        "category_store_id": "HOBBIES/CA/1",
        "forecast_days": 10,
        "total_predicted": 2682.6097397581475
      },
      {
        "category_store_id": "HOUSEHOLD/CA/1",
        "forecast_days": 7,
        "total_predicted": 2878.4702129712396
      }
    ],
    "failed_predictions": []
}
```
- Содержимое `summary`:
   * `total` - общее количество использованных моделей для прогнозов
   * `successful` - количество моделей, процесс прогнозирования в которых прошёл успешно
   * `failed` - количество моделей, проваливших процесс прогнозирования
   * `success_rate` - процесс успешных прогнозов
- Содержимое `successful_predictions` (таких элементов может быть столько, 
сколько указано в `success` в `summary`, в нашем случае их 5):
   * `category_store_id` - идентефикатор категории магазина
   * `forecast_days"` - количество дней прогноза (указанный период)
   * `total_predicted` - суммарный прогноз за все указанные дни
- Содержимое `failed_predictions` (таких элементов может быть столько, 
сколько указано в `failed` в `summary`, в нашем случае 0):
   * `category_store_id` - идентефикатор категории магазина
   * `error` - сообщение об ошибке
- Также пример с ошибкой, если вдруг появится модель, которая некорректно отработает:
```json
{
    "summary": {
        "total": 5,
        "successful": 4,
        "failed": 1,
        "success_rate": 80.0
    },
    "successful_predictions": [
        {
            "category_store_id": "FOODS/CA/1",
            "forecast_days": 14,
            "total_predicted": 4231.5
        },
        {
            "category_store_id": "FOODS/CA/2", 
            "forecast_days": 14,
            "total_predicted": 3820.3
        },
        {
            "category_store_id": "HOBBIES/CA/1",
            "forecast_days": 7,
            "total_predicted": 2150.8
        },
        {
            "category_store_id": "HOUSEHOLD/CA/1",
            "forecast_days": 7,
            "total_predicted": 1876.4
        }
    ],
    "failed_predictions": [
        {
            "category_store_id": "FOODS/CA/3",
            "error": "Модель не найдена: FOODS/CA/3"
        }
    ]
} 
```

## Пример и описание возвращаемого словаря в format.py в методе create_report
- Пример возвращаемого ответа:
```json
{
  "forecast_period": {
    "start_date": "2026-01-05",
    "end_date": "2026-01-18",
    "days": 14
  },
  "statistics": {
    "total_sales": 1577.0325543286954,
    "average_daily": 112.64518245204967,
    "min_daily": 53.01962705402356,
    "max_daily": 259.32259636716304,
    "std_daily": 82.1765341328125
  },
  "model_info": {
    "type": "Ridge",
    "trained_date": "unknown"
  },
  "weekly_trends": {
    "weekly_totals": [857.2472663369189, 719.7852879917765],
    "weekly_growth_rates": [-16.03]
  }
}
```
- Приведём описание структуры его полей. json поделён на несколько секций:
  1) Секция `forecast_period` (информация о периоде прогноза):
       * `start_date` - дата первого дня прогноза 
       * `end_date` - дата последнего дня прогноза 
       * `days` - общее количество дней в прогнозе 
  2) Секция `statistics` (статистика прогноза):
       * `total_sales` - суммарный объем продаж за весь период
       * `average_daily` - средний дневной объем продаж 
       * `min_daily` - минимальный дневной объем продаж 
       * `max_daily` - максимальный дневной объем продаж 
       * `std_daily` - стандартное отклонение дневных продаж (показывает волатильность)
  3) Секция `model_info` (информация о модели):
       * `type` - тип использованной модели машинного обучения (XGBRegressor, Ridge, LGBMRegressor и т.д.)
       * `trained_date` - дата обучения модели
  4) Секция `weekly_trends` (недельные тренды, только если прогноз больше или равен 7 дням):
       * `weekly_totals` - список суммарных продаж по неделям
       * `weekly_growth_rates` - список процентных изменений между неделями (рассчитывается, если не менее двух недель)

## Итоговый json
- В PredictAPI эти два представленных выше формата json после объединяются в один: 
```json
{
  "status": "completed",
  "timestamp": "2026-01-04T19:33:43.203923",
  "results": {
    "... содержимое, полученное методом format_batch_results ...": "..."
  },
  "raw_results": {
    "... сырые результаты каждого отдельного прогноза ...": "..."
  }
}
```
- Пример полного такого ответа
```json
{
  "status": "completed",
  "timestamp": "2024-01-05T10:30:00",
  
  "results": {
    "summary": {
      "total": 3,
      "successful": 2,
      "failed": 1,
      "success_rate": 66.67
    },
    "successful_predictions": [
      {
        "category_store_id": "FOODS/CA/1",
        "forecast_days": 7,
        "total_predicted": 1121.4
      }
    ],
    "failed_predictions": [
      {
        "category_store_id": "FOODS/CA/2",
        "error": "Модель не найдена"
      }
    ]
  },
  
  "raw_results": {
    "FOODS/CA/1": {
      "status": "success",
      "timestamp": "2024-01-05T10:30:00",
      "category_store_id": "FOODS/CA/1",
      "forecast_days": 7,
      "predictions": [
        {
          "date": "2024-01-06",
          "predicted_sales": 158.3
        }, 
        {
          "date": "2024-01-07",
          "predicted_sales": 159.9
        },
        {"и т.д." : "и т.д."
        }
      ],
      "model_used": "XGBRegressor",
      "statistics": {
        "mean": 160.2,
        "min": 145.5,
        "max": 178.3,
        "total": 1121.4
      },
      "metadata": {
        "model_type": "XGBRegressor",
        "trained_date": "2024-01-04T15:45:00",
        "performance_metrics": {
          "MAE": 12.5,
          "RMSE": 18.3,
          "MAPE": 8.2
        }
      },
      "report": {
        "forecast_period": {
          "start_date": "2024-01-06",
          "end_date": "2024-01-12",
          "days": 7
        },
        "statistics": {
          "total_sales": 1121.4,
          "average_daily": 160.2,
          "min_daily": 145.5,
          "max_daily": 178.3,
          "std_daily": 12.8
        },
        "model_info": {
          "type": "XGBRegressor",
          "trained_date": "2024-01-04T15:45:00"
        }
      }
    },
    "FOODS/CA/2": {
      "status": "error",
      "timestamp": "2024-01-05T10:30:00",
      "error": "Модель не найдена"
    }
  }
}
```
