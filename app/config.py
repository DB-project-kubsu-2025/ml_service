from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODEL_ARTIFACTS_DIR = BASE_DIR / "app" / "model" / "artifacts"

MODEL_SETTINGS = {
    "test_days": 28,
    "random_state": 42,
    "n_jobs": -1,
}

MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)