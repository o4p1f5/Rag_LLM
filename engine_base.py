from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Optional

import numpy as np


class BaseEngine(ABC):
    def __init__(self, model: str) -> None:
        self.model = model

    @property
    def supports_embedding(self) -> bool:
        return True

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        context: Optional[str] = None,
    ) -> str:
        raise NotImplementedError
