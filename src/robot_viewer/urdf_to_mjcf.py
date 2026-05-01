from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from .utils import rotation_matrix_to_wxyz

_JOINT_TYPE_MAP: dict[str, str] = {
    "revolute": "hinge",
    "continuous": "hinge",
    "prismatic": "slide",
    "fixed": "",
    "floating": "free",
    "planar": "slide",
}

_RESERVED = frozenset({"world"})
_MOVING_JOINT_TYPES = frozenset({"revolute", "continuous", "prismatic", "floating", "planar"})
_MIN_INERTIA = 1e-6


def _urdf_rpy_to_mjcf_quat(rpy: str) -> str:
    """Convert URDF rpy (Rz(y)@Ry(p)@Rx(r)) to an MJCF body quat attribute."""
    r, p, y = (float(v) for v in rpy.split())
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])
    q = rotation_matrix_to_wxyz(R)
    return f"{q[0]} {q[1]} {q[2]} {q[3]}"


def build_ik_mjcf(urdf_path: str) -> str:
    """Build a minimal MJCF XML string from a URDF file's kinematic skeleton.

    The generated MJCF contains only the body hierarchy, joints, and inertias —
    no visual or collision geometry. This is sufficient for Mink IK.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    mjcf = ET.Element("mujoco", model=root.get("name", "robot"))
    ET.SubElement(mjcf, "compiler", angle="radian")
    worldbody = ET.SubElement(mjcf, "worldbody")

    links: dict[str, Any] = {}
    for child in root:
        if child.tag == "link":
            links[child.get("name")] = child

    joint_map: dict[str, Any] = {}
    children: dict[str, list[str]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        joint_map[child] = joint
        children.setdefault(parent, []).append(child)

    all_children = set(joint_map.keys())
    all_parents = set(children.keys())
    root_link = next(iter(all_parents - all_children), None)
    if root_link is None:
        root_link = "world"

    moving_bodies: set[str] = {
        child
        for child, joint in joint_map.items()
        if joint.get("type") in _MOVING_JOINT_TYPES
    }

    def add_body(parent_elem: ET.Element, link_name: str) -> None:
        link = links.get(link_name)
        is_world = link_name.lower() in _RESERVED

        if link_name != root_link and link_name in joint_map:
            joint = joint_map[link_name]
            origin = joint.find("origin")
            body_pos = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
            body_quat = (
                _urdf_rpy_to_mjcf_quat(origin.get("rpy", "0 0 0"))
                if origin is not None
                else "1 0 0 0"
            )
        else:
            body_pos = "0 0 0"
            body_quat = "1 0 0 0"

        if is_world and link_name == root_link:
            body = parent_elem
        elif is_world:
            body = ET.SubElement(
                parent_elem,
                "body",
                name=f"{link_name}_link",
                pos=body_pos,
                quat=body_quat,
            )
        else:
            body = ET.SubElement(
                parent_elem,
                "body",
                name=link_name,
                pos=body_pos,
                quat=body_quat,
            )

        if link_name != root_link and link_name in joint_map:
            joint = joint_map[link_name]
            joint_type = joint.get("type", "")

            axis_elem = joint.find("axis")
            axis_xyz = (
                axis_elem.get("xyz", "0 0 1") if axis_elem is not None else "0 0 1"
            )

            mj_type = _JOINT_TYPE_MAP.get(joint_type, "hinge")

            if mj_type:
                limit = joint.find("limit")
                attrs: dict[str, str] = {
                    "name": joint.get("name"),
                    "type": mj_type,
                    "pos": "0 0 0",
                }
                if mj_type in ("hinge", "slide"):
                    attrs["axis"] = axis_xyz
                if limit is not None:
                    lower = limit.get("lower", str(-np.pi))
                    upper = limit.get("upper", str(np.pi))
                    attrs["range"] = f"{lower} {upper}"
                ET.SubElement(body, "joint", **attrs)

        is_moving = link_name in moving_bodies
        urdf_inertial = link.find("inertial") if link is not None else None
        needs_inertial = (is_moving or urdf_inertial is not None) and not (
            is_world and link_name == root_link
        )
        if needs_inertial:
            if urdf_inertial is not None:
                mass = urdf_inertial.find("mass")
                mass_val = (
                    mass.get("value", "0.001") if mass is not None else "0.001"
                )
                inert = urdf_inertial.find("inertia")
                if inert is not None:
                    ixx = str(
                        max(float(inert.get("ixx", str(_MIN_INERTIA))), _MIN_INERTIA)
                    )
                    iyy = str(
                        max(float(inert.get("iyy", str(_MIN_INERTIA))), _MIN_INERTIA)
                    )
                    izz = str(
                        max(float(inert.get("izz", str(_MIN_INERTIA))), _MIN_INERTIA)
                    )
                else:
                    ixx = iyy = izz = str(_MIN_INERTIA)
            else:
                mass_val = "0.001"
                ixx = iyy = izz = str(_MIN_INERTIA)
            ET.SubElement(
                body,
                "inertial",
                pos="0 0 0",
                mass=mass_val,
                diaginertia=f"{ixx} {iyy} {izz}",
            )

        for child_name in children.get(link_name, []):
            add_body(body, child_name)

    add_body(worldbody, root_link)

    return ET.tostring(mjcf, encoding="unicode")
