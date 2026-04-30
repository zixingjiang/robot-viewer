from __future__ import annotations

from typing import Any, Callable

from .model_sources import ModelSource, StartupFileNotFoundError


def execute_model_load(
    *,
    state: Any,
    source: ModelSource,
    status_text: Any,
    load_meshes: bool,
    mount_loaded_robot: Callable[[Any, str], None],
    reload_connected_pages: Callable[[], None],
) -> None:
    status_text.value = f"Loading {source.loading_label}..."

    with state.load_lock:
        try:
            result = source.load(load_meshes=load_meshes, tmp_dir=state.tmp_dir)
            mount_loaded_robot(result.urdf, result.source_path)
            status_text.value = f"Loaded {result.status_label}."
        except Exception as exc:
            if isinstance(exc, StartupFileNotFoundError):
                status_text.value = str(exc)
            else:
                failure_label = getattr(source, "failure_label", source.loading_label)
                status_text.value = f"Failed to load {failure_label}: {exc!r}"
