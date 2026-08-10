"""Model artefacts and the on-disk / in-database registry.

An artefact is a JSON document plus, for tree-based backends, a pickle
sidecar.  The JSON always carries the feature specification and the
calibrators, so even if the sidecar is missing (different machine, missing
LightGBM) the registry can report exactly what is wrong instead of silently
predicting garbage.

Only one version of a model is ``active`` at a time; activation is recorded in
the ``model_versions`` table so it is part of the audit trail.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from app.logger import get_logger
from app.ml.calibration import Calibrator, calibrator_from_dict
from app.ml.features import FeatureSpec

log = get_logger(__name__)


@dataclass(slots=True)
class ModelArtifact:
    name: str
    version: str
    algorithm: str
    trained_at: int
    rows: int
    feature_spec: FeatureSpec
    model_payload: dict[str, Any]
    long_calibrator: Calibrator | None = None
    short_calibrator: Calibrator | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    runtime: Any = field(default=None, compare=False, repr=False)

    @property
    def key(self) -> str:
        return f"{self.name}_{self.version}"

    @property
    def needs_sidecar(self) -> bool:
        return bool(self.model_payload.get("external"))

    def attach_runtime(self, model: Any) -> None:
        """Hold the live model object (not serialised into JSON)."""

        self.runtime = model

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "algorithm": self.algorithm,
            "trained_at": self.trained_at,
            "rows": self.rows,
            "feature_spec": self.feature_spec.as_dict(),
            "model": self.model_payload,
            "long_calibrator": (
                self.long_calibrator.as_dict() if self.long_calibrator else None
            ),
            "short_calibrator": (
                self.short_calibrator.as_dict() if self.short_calibrator else None
            ),
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelArtifact":
        return cls(
            name=str(data["name"]),
            version=str(data["version"]),
            algorithm=str(data.get("algorithm", "unknown")),
            trained_at=int(data.get("trained_at", 0)),
            rows=int(data.get("rows", 0)),
            feature_spec=FeatureSpec.from_dict(data.get("feature_spec", {})),
            model_payload=dict(data.get("model", {})),
            long_calibrator=calibrator_from_dict(data.get("long_calibrator")),
            short_calibrator=calibrator_from_dict(data.get("short_calibrator")),
            metrics=dict(data.get("metrics", {})),
        )


class ModelRegistry:
    def __init__(self, directory: Path, repository: Any = None) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.repository = repository

    # -- paths ------------------------------------------------------------

    def json_path(self, artifact_or_key: ModelArtifact | str) -> Path:
        key = (
            artifact_or_key.key
            if isinstance(artifact_or_key, ModelArtifact)
            else artifact_or_key
        )
        return self.directory / f"{key}.json"

    def sidecar_path(self, artifact_or_key: ModelArtifact | str) -> Path:
        key = (
            artifact_or_key.key
            if isinstance(artifact_or_key, ModelArtifact)
            else artifact_or_key
        )
        return self.directory / f"{key}.pkl"

    # -- write ------------------------------------------------------------

    def save(self, artifact: ModelArtifact, activate: bool = True) -> Path:
        path = self.json_path(artifact)
        path.write_text(
            json.dumps(artifact.as_dict(), indent=2, default=str), encoding="utf-8"
        )

        if artifact.needs_sidecar and artifact.runtime is not None:
            try:
                self.sidecar_path(artifact).write_bytes(pickle.dumps(artifact.runtime))
            except Exception as exc:  # noqa: BLE001 - JSON alone is still useful
                log.warning("could not write model sidecar for %s: %s", artifact.key, exc)

        if self.repository is not None:
            try:
                self.repository.register(
                    {
                        "name": artifact.name,
                        "version": artifact.version,
                        "algorithm": artifact.algorithm,
                        "trained_at": artifact.trained_at,
                        "rows": artifact.rows,
                        "metrics": artifact.metrics,
                        "feature_names": artifact.feature_spec.names,
                        "path": str(path),
                        "active": activate,
                    }
                )
                if activate:
                    self.repository.activate(artifact.name, artifact.version)
            except Exception as exc:  # noqa: BLE001 - disk copy is the source of truth
                log.warning("could not record model in the database: %s", exc)

        if activate:
            self._write_pointer(artifact.name, artifact.key)
        log.info("saved model %s (%s)", artifact.key, artifact.algorithm)
        return path

    def _write_pointer(self, name: str, key: str) -> None:
        pointer = self.directory / f"{name}.active"
        pointer.write_text(key, encoding="utf-8")

    # -- read -------------------------------------------------------------

    def load(self, key: str) -> ModelArtifact | None:
        path = self.json_path(key)
        if not path.is_file():
            return None
        try:
            artifact = ModelArtifact.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError, KeyError) as exc:
            log.error("could not read model %s: %s", key, exc)
            return None

        if artifact.needs_sidecar:
            sidecar = self.sidecar_path(key)
            if sidecar.is_file():
                try:
                    artifact.attach_runtime(pickle.loads(sidecar.read_bytes()))
                except Exception as exc:  # noqa: BLE001
                    log.error("could not load model sidecar %s: %s", key, exc)
                    return None
            else:
                log.error(
                    "model %s needs sidecar %s which is missing - ignoring the model",
                    key,
                    sidecar.name,
                )
                return None
        else:
            from app.ml.train import SoftmaxRegression

            artifact.attach_runtime(SoftmaxRegression.from_dict(artifact.model_payload))
        return artifact

    def active(self, name: str) -> ModelArtifact | None:
        if self.repository is not None:
            try:
                row = self.repository.active(name)
                if row and row.get("path"):
                    key = Path(str(row["path"])).stem
                    artifact = self.load(key)
                    if artifact is not None:
                        return artifact
            except Exception as exc:  # noqa: BLE001
                log.debug("database model lookup failed: %s", exc)

        pointer = self.directory / f"{name}.active"
        if pointer.is_file():
            return self.load(pointer.read_text(encoding="utf-8").strip())

        candidates = sorted(self.directory.glob(f"{name}_*.json"))
        if not candidates:
            return None
        return self.load(candidates[-1].stem)

    def list_versions(self, name: str | None = None) -> list[str]:
        pattern = f"{name}_*.json" if name else "*.json"
        return sorted(path.stem for path in self.directory.glob(pattern))

    def prune(self, name: str, keep: int = 5) -> int:
        versions = self.list_versions(name)
        removed = 0
        active_key = None
        pointer = self.directory / f"{name}.active"
        if pointer.is_file():
            active_key = pointer.read_text(encoding="utf-8").strip()
        for key in versions[:-keep] if len(versions) > keep else []:
            if key == active_key:
                continue
            self.json_path(key).unlink(missing_ok=True)
            self.sidecar_path(key).unlink(missing_ok=True)
            removed += 1
        return removed


__all__ = ["ModelArtifact", "ModelRegistry"]
