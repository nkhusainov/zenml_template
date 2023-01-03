from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class ModelHandler:
    func_train: Callable
    func_infer: Callable
    func_preprocess: Optional[Callable] = None
