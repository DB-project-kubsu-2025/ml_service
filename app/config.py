from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODEL_ARTIFACTS_DIR = BASE_DIR / "app" / "model" / "artifacts"
LOG_DIR = BASE_DIR / "logs"

MODEL_SETTINGS = {
    "test_days": 28,
    "random_state": 42,
    "n_jobs": -1,
}

MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "ml_service.log"
ERROR_LOG_FILE = LOG_DIR / "ml_service_errors.log"

LOG_CONFIG = {
    "log_file": str(LOG_FILE),
    "error_log_file": str(ERROR_LOG_FILE),
    "log_level": "INFO",
    "max_file_size": 10485760,  # 10MB
    "backup_count": 5
}
