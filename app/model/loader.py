import joblib
import json
from typing import Optional, Tuple, Dict, Any
import logging
from datetime import datetime
from app.config import MODEL_ARTIFACTS_DIR
from app.type import ModelMetadata

logger = logging.getLogger(__name__)


class ModelLoader:
    """Класс для работы с моделями"""

    def __init__(self):
        self.models_cache = {}
        self.model_dir = MODEL_ARTIFACTS_DIR

    def save_model(self, model: Any, category_store_id: str,
                   metadata: Optional[ModelMetadata] = None) -> bool:
        """Сохранение модели и метаданных"""

        safe_name = self._get_safe_filename(category_store_id)
        model_path = self.model_dir / f"{safe_name}_model.pkl"
        meta_path = self.model_dir / f"{safe_name}_metadata.json"

        joblib.dump(model, model_path)

        if metadata is None:
            metadata = {}

        metadata.update({
            'category_store_id': category_store_id,
            'saved_at': datetime.now().isoformat(),
            'model_type': type(model).__name__
        })

        try:
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"Модель сохранена: {category_store_id}")
            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения модели {category_store_id}: {e}")
            return False

    def load_model(self, category_store_id: str) -> Tuple[Optional[Any], ModelMetadata]:
        """Загрузка модели и метаданных"""
        if category_store_id in self.models_cache:
            return self.models_cache[category_store_id]

        try:
            safe_name = self._get_safe_filename(category_store_id)

            possible_model_files = [
                self.model_dir / f"best_model_{safe_name}.pkl",
                self.model_dir / f"{safe_name}_model.pkl",
                self.model_dir / f"model_{safe_name}.pkl"
            ]

            model_path = None
            for path in possible_model_files:
                if path.exists():
                    model_path = path
                    break

            if model_path is None:
                logger.warning(f"Модель не найдена: {category_store_id}")
                files = list(self.model_dir.glob("*.pkl"))
                logger.warning(f"Доступные файлы: {[f.name for f in files]}")
                return None, {}

            possible_meta_files = [
                self.model_dir / f"best_model_{safe_name}_metadata.json",
                self.model_dir / f"{safe_name}_metadata.json",
                self.model_dir / f"metadata_{safe_name}.json",
                self.model_dir / f"best_model_{safe_name}.json"
            ]

            meta_path = None
            for path in possible_meta_files:
                if path.exists():
                    meta_path = path
                    break

            model = joblib.load(model_path)

            metadata = {}
            if meta_path and meta_path.exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            else:
                metadata = {
                    'category_store_id': category_store_id,
                    'model_type': type(model).__name__,
                    'loaded_at': datetime.now().isoformat(),
                    'model_file': str(model_path.name)
                }

            self.models_cache[category_store_id] = (model, metadata)

            logger.info(f"Модель загружена: {category_store_id} из {model_path.name}")
            return model, metadata

        except Exception as e:
            logger.error(f"Ошибка загрузки модели {category_store_id}: {e}")
            return None, {}

    def list_models(self) -> list:
        """Список доступных моделей"""
        models = []
        for pattern in ["best_model_*.pkl", "*_model.pkl", "model_*.pkl"]:
            for file in self.model_dir.glob(pattern):
                if file.name.startswith('best_model_'):
                    model_name = file.stem.replace('best_model_', '')
                elif file.name.endswith('_model.pkl'):
                    model_name = file.stem.replace('_model', '')
                elif file.name.startswith('model_'):
                    model_name = file.stem.replace('model_', '')
                else:
                    continue

                category_id = model_name.replace('_', '/')
                if category_id not in models:
                    models.append(category_id)

        return sorted(models)

    def _get_safe_filename(self, category_store_id: str) -> str:
        """Создание безопасного имени файла"""
        return category_store_id.replace('/', '_').replace('\\', '_')
