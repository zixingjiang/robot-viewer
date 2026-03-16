from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin  # type: ignore[import]
from viser._gui_handles import UploadedFile


def rotation_matrix_to_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    quat_xyzw = pin.Quaternion(rotation).coeffs()  # type: ignore[attr-defined]
    return (
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    )


def wxyz_to_rotation_matrix(wxyz: tuple[float, float, float, float]) -> np.ndarray:
    quat = pin.Quaternion(  # type: ignore[attr-defined]
        float(wxyz[0]), float(wxyz[1]), float(wxyz[2]), float(wxyz[3])
    )
    return quat.matrix()


def safe_write_file(uploaded_file: UploadedFile, tmp_dir: str) -> str:
    """Write the uploaded file to a temporary directory and return the path."""

    os.makedirs(tmp_dir, exist_ok=True)
    out_path = os.path.join(tmp_dir, os.path.basename(uploaded_file.name))
    with open(out_path, "wb") as file_handle:
        file_handle.write(uploaded_file.content)
    return out_path


def sanitize_urdf_for_pinocchio(path: str, tmp_dir: str) -> tuple[str, int]:
    """Return a URDF path safe for Pinocchio, removing duplicate global materials."""

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception:
        return path, 0

    if root.tag != "robot":
        return path, 0

    seen_materials: set[str] = set()
    removed_count = 0

    for child in list(root):
        if child.tag != "material":
            continue

        material_name = child.attrib.get("name")
        if not material_name:
            continue

        if material_name in seen_materials:
            root.remove(child)
            removed_count += 1
            continue

        seen_materials.add(material_name)

    if removed_count == 0:
        return path, 0

    out_name = f"pinocchio_sanitized_{os.path.basename(path)}"
    out_path = os.path.join(tmp_dir, out_name)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path, removed_count
