import logging
import logging.handlers
import sys
from app.config import LOG_CONFIG


def setup_logging():
    """Настройка логирования для всего проекта"""

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

    file_handler = logging.handlers.RotatingFileHandler(
        filename=LOG_CONFIG['log_file'],
        maxBytes=LOG_CONFIG['max_file_size'],
        backupCount=LOG_CONFIG['backup_count'],
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, LOG_CONFIG['log_level']))
    file_handler.setFormatter(formatter)

    error_handler = logging.handlers.RotatingFileHandler(
        filename=LOG_CONFIG['error_log_file'],
        maxBytes=LOG_CONFIG['max_file_size'],
        backupCount=LOG_CONFIG['backup_count'],
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)

    logging.getLogger('lightgbm').setLevel(logging.WARNING)
    logging.getLogger('xgboost').setLevel(logging.WARNING)

    root_logger.info(f"Логирование настроено. Основной файл: {LOG_CONFIG['log_file']}")
    root_logger.info(f"Файл ошибок: {LOG_CONFIG['error_log_file']}")


def get_logger(name):
    """Получение логгера с указанным именем"""
    return logging.getLogger(name)


setup_logging()