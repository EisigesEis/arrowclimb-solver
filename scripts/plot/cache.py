from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .config import PlotSpec, RunConfig, TOOL_VERSION


def file_fingerprint(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_output_fingerprint(
    input_fingerprint: str,
    config: RunConfig,
    plot_spec: PlotSpec,
) -> str:
    payload = {
        "tool_version": TOOL_VERSION,
        "input_fingerprint": input_fingerprint,
        "config": config.normalized_options(),
        "plot": plot_spec.fingerprint_payload(),
    }
    return stable_json_fingerprint(payload)


def load_manifest(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def should_skip_output(
    output_path: str,
    manifest: dict[str, Any],
    plot_key: str,
    expected_fingerprint: str,
    force: bool,
) -> bool:
    if force or not os.path.exists(output_path):
        return False
    outputs = manifest.get("outputs", {})
    output_entry = outputs.get(plot_key)
    if not isinstance(output_entry, dict):
        return False
    return output_entry.get("fingerprint") == expected_fingerprint


def build_manifest(
    input_path: str,
    input_fingerprint: str,
    config: RunConfig,
    outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "tool_version": TOOL_VERSION,
        "input": {
            "path": input_path,
            "fingerprint": input_fingerprint,
        },
        "options": config.normalized_options(),
        "outputs": outputs,
    }


def write_json(path: str, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

