"""
Центральный реестр моделей: строковое имя ↔ Python-класс.
Добавление новой модели = одна строка внизу файла.
"""
from typing import Dict, Type
from .models.base_model import BaseModel
from .models.cnn_model import CNNModel
from .models.transformer_model import TransformerModel
from .models.logreg_model import LogRegModel
from .models.atac_linear_model import ATACLinearModel 

_MODEL_REGISTRY = {
    "cnn"        : CNNModel,
    "transformer": TransformerModel,
    "logreg"     : LogRegModel,
    "atac_linear": ATACLinearModel,     # <-- зарегистрировали
}

def get_model_cls(name: str) -> Type[BaseModel]:
    try:
        return _MODEL_REGISTRY[name.lower()]
    except KeyError as e:
        raise ValueError(f"Неизвестная модель '{name}'") from e
