from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import os
import xml.etree.ElementTree as ET
from typing import Any, Callable, Protocol

import yourdfpy  # type: ignore[import]
from viser._gui_handles import UploadedFile

from .utils import safe_write_file


class StartupFileNotFoundError(FileNotFoundError):
    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"Startup file not found: {path}")


@dataclass(frozen=True)
class LoadResult:
    urdf: yourdfpy.URDF
    source_path: str
    file_label: str
    status_label: str


class ModelSource(Protocol):
    @property
    def loading_label(self) -> str: ...

    @property
    def failure_label(self) -> str: ...

    def load(self, *, load_meshes: bool, tmp_dir: str) -> LoadResult: ...


@dataclass(frozen=True)
class PathModelSource:
    path: str

    @property
    def loading_label(self) -> str:
        return os.path.basename(os.path.abspath(self.path))

    @property
    def failure_label(self) -> str:
        return self.loading_label

    def load(self, *, load_meshes: bool, tmp_dir: str) -> LoadResult:
        del tmp_dir
        resolved_path = os.path.abspath(self.path)
        if not os.path.isfile(resolved_path):
            raise StartupFileNotFoundError(resolved_path)

        file_name = os.path.basename(resolved_path)
        return LoadResult(
            urdf=load_urdf(resolved_path, load_meshes),
            source_path=resolved_path,
            file_label=file_name,
            status_label=file_name,
        )


@dataclass(frozen=True)
class UploadedModelSource:
    uploaded_file: UploadedFile | Any

    @property
    def loading_label(self) -> str:
        return str(self.uploaded_file.name)

    @property
    def failure_label(self) -> str:
        return self.loading_label

    def load(self, *, load_meshes: bool, tmp_dir: str) -> LoadResult:
        source_path = safe_write_file(self.uploaded_file, tmp_dir)
        file_label = str(self.uploaded_file.name)
        return LoadResult(
            urdf=load_urdf(source_path, load_meshes),
            source_path=source_path,
            file_label=file_label,
            status_label=file_label,
        )


@dataclass(frozen=True)
class RobotDescriptionModelSource:
    description_name: str

    @property
    def loading_label(self) -> str:
        return self.description_name

    @property
    def failure_label(self) -> str:
        return f"robot_descriptions entry {self.description_name}"

    def load(self, *, load_meshes: bool, tmp_dir: str) -> LoadResult:
        del load_meshes
        del tmp_dir
        resolved_name, urdf, source_path = load_robot_description_urdf(
            self.description_name
        )
        return LoadResult(
            urdf=urdf,
            source_path=source_path,
            file_label=f"{resolved_name} (robot_descriptions)",
            status_label=resolved_name,
        )


def _detect_format(path: str) -> str:
    """Return 'urdf', 'mjcf', or 'unknown' by sniffing the XML root element.

    For non-XML files, falls back to extension-based detection.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mjb":
        return "mjcf"

    if ext not in (".urdf", ".xml"):
        return "unknown"

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception:
        return "unknown"

    tag = root.tag
    if tag == "robot":
        return "urdf"
    if tag == "mujoco":
        return "mjcf"
    return "unknown"


def load_mjcf(path: str) -> tuple[Any, Any]:
    """Load an MJCF file with MuJoCo, returning (MjModel, MjData)."""
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "MJCF support requires 'mujoco'. Install with: pip install mujoco mink trimesh"
        ) from exc

    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    return model, data


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


def get_robot_description_mjcf_candidates() -> list[str]:
    """Return robot_descriptions entries that have MJCF (but not URDF)."""
    try:
        from robot_descriptions import DESCRIPTIONS
    except Exception:
        return []

    return sorted(
        name
        for name, description in DESCRIPTIONS.items()
        if getattr(description, "has_mjcf", False)
        and not getattr(description, "has_urdf", False)
    )


def load_urdf(path: str, load_meshes: bool) -> yourdfpy.URDF:
    return yourdfpy.URDF.load(
        path,
        load_meshes=load_meshes,
        load_collision_meshes=False,
        mesh_dir=os.path.dirname(path),
    )


def _resolve_robot_description_module(description_name: str) -> tuple[Any, str]:
    """Resolve a robot_descriptions module, returning (module, resolved_name)."""
    candidates = [description_name]
    if not description_name.endswith("_description"):
        candidates.append(f"{description_name}_description")

    for candidate in candidates:
        module_name = f"robot_descriptions.{candidate}"
        try:
            return import_module(module_name), candidate
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise

    raise ModuleNotFoundError(
        "Could not import robot description "
        f"'{description_name}' as a robot_descriptions submodule"
    )


def _detect_robot_description_format(description_name: str) -> str:
    """Return 'urdf' or 'mjcf' for a robot_descriptions entry."""
    module, _ = _resolve_robot_description_module(description_name)
    if hasattr(module, "URDF_PATH") or hasattr(module, "XACRO_PATH"):
        return "urdf"
    if hasattr(module, "MJCF_PATH"):
        return "mjcf"
    raise RuntimeError(
        f"Robot description '{description_name}' has no URDF, Xacro, or MJCF data"
    )


def _resolve_robot_description_urdf_path(description_name: str) -> tuple[str, str]:
    module, resolved_name = _resolve_robot_description_module(description_name)
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


def _resolve_robot_description_mjcf_path(description_name: str) -> tuple[str, str]:
    """Resolve an MJCF path from a robot_descriptions entry."""
    module, resolved_name = _resolve_robot_description_module(description_name)
    if hasattr(module, "MJCF_PATH"):
        return resolved_name, str(module.MJCF_PATH)

    raise RuntimeError("Selected robot description does not provide MJCF data")


def load_robot_description_mjcf(description_name: str) -> tuple[str, Any, Any, str]:
    """Load an MJCF from a robot_descriptions entry.

    Returns (resolved_name, MjModel, MjData, source_path).
    """
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "MJCF support requires 'mujoco'. Install with: pip install mujoco mink trimesh"
        ) from exc

    resolved_name, mjcf_path = _resolve_robot_description_mjcf_path(description_name)
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    return resolved_name, model, data, mjcf_path


def execute_model_load(
    *,
    state: Any,
    source: ModelSource,
    status_text: Any,
    load_meshes: bool,
    mount_loaded_robot: Callable[[Any, str], None],
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
