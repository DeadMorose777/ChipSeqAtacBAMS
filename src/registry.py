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
from .models.transformer_dna_only_model  import TransformerDNAOnlyModel
from .models.transformer_atac_only_model import TransformerATACOnlyModel

_MODEL_REGISTRY = {
    "cnn"        : CNNModel,
    "transformer": TransformerModel,
    "dna_only"   : TransformerDNAOnlyModel,   # ← новое
    "atac_only"  : TransformerATACOnlyModel,  # ← новое
    "atac_linear": ATACLinearModel,
}


def get_model_cls(name: str) -> Type[BaseModel]:
    try:
        return _MODEL_REGISTRY[name.lower()]
    except KeyError as e:
        raise ValueError(f"Неизвестная модель '{name}'") from e
