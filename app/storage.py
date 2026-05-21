from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, TypeAdapter

T = TypeVar("T", bound=BaseModel)


class JsonStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load_many(self, model: type[T]) -> list[T]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text())
        adapter = TypeAdapter(list[model])
        return adapter.validate_python(payload)

    def save_many(self, items: list[BaseModel]) -> None:
        serializable = [item.model_dump(mode="json") for item in items]
        self.path.write_text(json.dumps(serializable, indent=2))
