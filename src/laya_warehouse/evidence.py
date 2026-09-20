"""Portable model identity and training provenance for benchmark reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_evidence(model: str) -> dict[str, Any]:
    evidence: dict[str, Any] = {"identifier": model}
    model_path = Path(model)
    if not model_path.is_dir():
        return evidence

    weights_path = model_path / "model.safetensors"
    config_path = model_path / "rl_agent_config.json"
    metrics_path = model_path / "training_metrics.json"
    if weights_path.is_file():
        evidence["weights_sha256"] = sha256_file(weights_path)
    if config_path.is_file():
        evidence["config_sha256"] = sha256_file(config_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        evidence["fine_tuned"] = bool(config.get("fine_tuned"))
        evidence["trained_questions"] = config.get("trained_questions")
    if metrics_path.is_file():
        evidence["training_metrics_sha256"] = sha256_file(metrics_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        evidence["training"] = {
            key: metrics.get(key)
            for key in (
                "base_model",
                "base_model_requested_revision",
                "base_model_resolved_snapshot",
                "base_model_weights_sha256",
                "implementation_commit",
                "seed",
                "manifests",
                "best_epoch",
                "best_validation_selection",
                "training_augmentation",
                "training_loss_weighting",
            )
        }
    return evidence
