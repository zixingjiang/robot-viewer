from __future__ import annotations

from importlib import import_module
import os

import yourdfpy  # type: ignore[import]


def get_robot_description_candidates() -> list[str]:
    try:
        from robot_descriptions import DESCRIPTIONS
    except Exception:
        return []

    return sorted(
        name
        for name, description in DESCRIPTIONS.items()
        if getattr(description, "has_urdf", False)
    )


def load_urdf(path: str, load_meshes: bool) -> yourdfpy.URDF:
    return yourdfpy.URDF.load(
        path,
        load_meshes=load_meshes,
        load_collision_meshes=False,
        mesh_dir=os.path.dirname(path),
    )


def _resolve_robot_description_urdf_path(description_name: str) -> tuple[str, str]:
    candidates = [description_name]
    if not description_name.endswith("_description"):
        candidates.append(f"{description_name}_description")

    module = None
    resolved_name = ""
    for candidate in candidates:
        module_name = f"robot_descriptions.{candidate}"
        try:
            module = import_module(module_name)
            resolved_name = candidate
            break
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise

    if module is None:
        raise ModuleNotFoundError(
            "Could not import robot description "
            f"'{description_name}' as a robot_descriptions submodule"
        )

    if hasattr(module, "URDF_PATH"):
        return resolved_name, str(module.URDF_PATH)
    if hasattr(module, "XACRO_PATH"):
        from robot_descriptions._xacro import get_urdf_path

        return resolved_name, str(get_urdf_path(module))

    raise RuntimeError("Selected robot description does not provide URDF or Xacro data")


def load_robot_description_urdf(
    description_name: str,
) -> tuple[str, yourdfpy.URDF, str]:
    try:
        from robot_descriptions.loaders.yourdfpy import (
            load_robot_description as load_rd,
        )
    except Exception as exc:
        raise RuntimeError(
            "robot_descriptions is not available; install the package to use this feature"
        ) from exc

    resolved_name, urdf_path = _resolve_robot_description_urdf_path(description_name)
    return resolved_name, load_rd(resolved_name), urdf_path
