"""Artifact storage abstractions for Stage 0."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.models import ArtifactHandle, CONTRACT_VERSION


class ArtifactRecord(BaseModel):
    """Stored artifact payload and metadata."""

    model_config = ConfigDict(validate_assignment=True)

    contract_version: str = Field(default=CONTRACT_VERSION)
    handle: ArtifactHandle
    value: Any
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactStore:
    """Deterministic in-memory store with optional file-backed persistence."""

    def __init__(self, base_path: str | Path | None = None) -> None:
        self._records: dict[str, ArtifactRecord] = {}
        self._base_path = Path(base_path) if base_path else None
        if self._base_path:
            self._base_path.mkdir(parents=True, exist_ok=True)

    def put(
        self,
        key: str,
        value: Any,
        *,
        artifact_type: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactHandle:
        handle = ArtifactHandle(key=key, artifact_type=artifact_type)
        record = ArtifactRecord(handle=handle, value=value, metadata=metadata or {})
        self._records[key] = record
        self._persist(record)
        return handle

    def get(self, key: str) -> ArtifactRecord:
        if key in self._records:
            return self._records[key]

        if self._base_path:
            persisted = self._read_persisted(key)
            if persisted is not None:
                self._records[key] = persisted
                return persisted

        raise KeyError(f"unknown artifact key: {key}")

    def list(self) -> list[ArtifactHandle]:
        keys = set(self._records)
        if self._base_path:
            keys.update(path.stem for path in self._base_path.glob("*.json"))
        return [self.get(key).handle for key in sorted(keys)]

    def delete(self, key: str) -> None:
        self._records.pop(key, None)
        if self._base_path:
            artifact_path = self._artifact_path(key)
            if artifact_path.exists():
                artifact_path.unlink()

    def clear(self) -> None:
        for key in list(self._records):
            self.delete(key)
        self._records.clear()

    def _persist(self, record: ArtifactRecord) -> None:
        if not self._base_path:
            return
        self._artifact_path(record.handle.key).write_text(
            json.dumps(record.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )

    def _read_persisted(self, key: str) -> ArtifactRecord | None:
        if not self._base_path:
            return None
        artifact_path = self._artifact_path(key)
        if not artifact_path.exists():
            return None
        return ArtifactRecord.model_validate_json(artifact_path.read_text(encoding="utf-8"))

    def _artifact_path(self, key: str) -> Path:
        safe_key = key.replace("/", "__").replace("\\", "__")
        return self._base_path / f"{safe_key}.json"  # type: ignore[operator]
