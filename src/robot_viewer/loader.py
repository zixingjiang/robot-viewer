from __future__ import annotations

from importlib import import_module
import os
import re
from typing import Any

import viser
import yourdfpy  # type: ignore[import]

from .state import ViewerState


_ROBOT_ROOT_RE = re.compile(r"^/robot(?:_[0-9a-f]+)?(?:/|$)")


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


def _discover_robot_roots(server: viser.ViserServer) -> set[str]:
    scene_handles = getattr(server.scene, "_handle_from_node_name", None)
    if not isinstance(scene_handles, dict):
        return set()

    roots: set[str] = set()
    for node_name in scene_handles.keys():
        if not _ROBOT_ROOT_RE.match(node_name):
            continue
        parts = node_name.split("/", 2)
        if len(parts) >= 2 and parts[1]:
            roots.add(f"/{parts[1]}")
    return roots


def _purge_robot_replay_messages(server: viser.ViserServer) -> None:
    websock_server = getattr(server, "_websock_server", None)
    if websock_server is None:
        return

    broadcast_buffer = getattr(websock_server, "_broadcast_buffer", None)
    if broadcast_buffer is None:
        return

    remove_from_buffer = getattr(broadcast_buffer, "remove_from_buffer", None)
    if remove_from_buffer is None:
        return

    def _is_robot_message(message: object) -> bool:
        name = getattr(message, "name", None)
        return isinstance(name, str) and _ROBOT_ROOT_RE.match(name) is not None

    try:
        remove_from_buffer(_is_robot_message)
    except Exception:
        return


def prune_stale_robot_roots(server: viser.ViserServer, state: ViewerState) -> None:
    discovered_roots = _discover_robot_roots(server)
    keep_root = state.current_root_name

    for root_name in discovered_roots:
        if keep_root is not None and root_name == keep_root:
            continue
        try:
            server.scene.remove_by_name(root_name)
        except Exception:
            continue

    state.robot_root_names = {keep_root} if keep_root is not None else set()


def clear_previous_robot(server: viser.ViserServer, state: ViewerState) -> None:
    state.ik_enabled = False

    previous_urdf = state.current_urdf
    did_scene_reset = False

    try:
        server.scene.reset()
        server.flush()
        did_scene_reset = True
    except Exception:
        pass

    if not did_scene_reset:
        if state.ground_plane_handle is not None:
            try:
                state.ground_plane_handle.remove()
            except Exception:
                try:
                    server.scene.remove_by_name("/grid")
                except Exception:
                    pass

        if state.cartesian_target_handle is not None:
            try:
                state.cartesian_target_handle.remove()
            except Exception:
                pass

        robot_roots = set(state.robot_root_names)
        if state.current_root_name is not None:
            robot_roots.add(state.current_root_name)
        robot_roots.update(_discover_robot_roots(server))

        removed_any_root = False
        for root_name in robot_roots:
            try:
                server.scene.remove_by_name(root_name)
                removed_any_root = True
            except Exception:
                pass

        if not removed_any_root and previous_urdf is not None:
            try:
                previous_urdf.remove()
            except Exception:
                pass

    state.ground_plane_handle = None
    state.ground_plane_size = (0, 0)
    state.cartesian_target_handle = None
    state.current_urdf = None
    state.current_root_name = None
    state.robot_root_names.clear()
    state.link_frame_handles.clear()
    state.frame_name_handles.clear()

    for attr in (
        "control_folder_handle",
        "visibility_folder_handle",
        "cartesian_folder_handle",
        "transform_folder_handle",
    ):
        folder = getattr(state, attr)
        if folder is not None:
            try:
                folder.remove()
            except Exception:
                pass
        setattr(state, attr, None)

    state.slider_handles = None
    state.joint_names = None
    state.initial_config = None
    state.joint_limits = None
    state.randomize_button = None
    state.reset_button = None
    state.cartesian_mode_checkbox = None
    state.visibility_visual_checkbox = None
    state.visibility_frames_checkbox = None
    state.visibility_frame_names_checkbox = None
    state.visibility_ground_checkbox = None
    state.cartesian_frame_dropdown = None
    state.transform_from_dropdown = None
    state.transform_to_dropdown = None
    state.transform_translation_text = None
    state.transform_rotation_text = None
    state.ik_configuration = None
    state.ik_tasks = None
    state.ik_frame_task = None
    state.ik_posture_task = None
    state.ik_damping_task = None
    state.ik_joint_name_to_q_index = {}
    state.ik_solver = None


def prepare_scene_for_reload(server: viser.ViserServer, state: ViewerState) -> None:
    prune_stale_robot_roots(server, state)
    clear_previous_robot(server, state)
    _purge_robot_replay_messages(server)
