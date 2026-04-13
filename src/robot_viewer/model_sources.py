from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Protocol

import yourdfpy  # type: ignore[import]
from viser._gui_handles import UploadedFile

from .loader import load_robot_description_urdf, load_urdf
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
