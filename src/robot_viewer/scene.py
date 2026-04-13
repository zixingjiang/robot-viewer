from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import viser

from .state import ViewerState
from .utils import rotation_matrix_to_wxyz


_MAX_GROUND_PLANE_SAMPLE_LINKS = 128
_ROBOT_ROOT_RE = re.compile(r"^/robot(?:_[0-9a-f]+)?(?:/|$)")


def update_transform_display(state: ViewerState) -> None:
    if (
        state.current_urdf is None
        or state.transform_from_dropdown is None
        or state.transform_to_dropdown is None
        or state.transform_translation_text is None
        or state.transform_rotation_text is None
    ):
        return

    from_frame = state.transform_from_dropdown.value
    to_frame = state.transform_to_dropdown.value

    try:
        urdf = state.current_urdf._urdf
        world_from = urdf.get_transform(from_frame)
        world_to = urdf.get_transform(to_frame)
        from_to = np.linalg.inv(world_from) @ world_to

        translation = from_to[:3, 3]
        rotation_wxyz = rotation_matrix_to_wxyz(from_to[:3, :3])

        state.suppress_transform_text_callbacks = True
        try:
            state.transform_translation_text.value = (
                f"{translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}"
            )
            state.transform_rotation_text.value = (
                f"{rotation_wxyz[0]:.4f}, {rotation_wxyz[1]:.4f}, "
                f"{rotation_wxyz[2]:.4f}, {rotation_wxyz[3]:.4f}"
            )
        finally:
            state.suppress_transform_text_callbacks = False
    except Exception:
        state.suppress_transform_text_callbacks = True
        try:
            state.transform_translation_text.value = "N/A"
            state.transform_rotation_text.value = "N/A"
        finally:
            state.suppress_transform_text_callbacks = False


def update_link_frame_visuals(state: ViewerState) -> None:
    if state.current_urdf is None:
        return

    urdf = state.current_urdf._urdf
    for link_name, frame_handle in state.link_frame_handles.items():
        try:
            transform = urdf.get_transform(link_name)
        except Exception:
            continue

        frame_handle.wxyz = rotation_matrix_to_wxyz(transform[:3, :3])
        frame_handle.position = (
            float(transform[0, 3]),
            float(transform[1, 3]),
            float(transform[2, 3]),
        )
        frame_handle.visible = state.show_link_frames

        name_handle = state.frame_name_handles.get(link_name)
        if name_handle is not None:
            name_handle.position = (
                float(transform[0, 3]),
                float(transform[1, 3]),
                float(transform[2, 3]),
            )
            name_handle.visible = state.show_frame_names

    update_transform_display(state)


def remove_link_frame_visuals(state: ViewerState) -> None:
    for handle in state.link_frame_handles.values():
        try:
            handle.remove()
        except Exception:
            pass
    state.link_frame_handles.clear()

    for handle in state.frame_name_handles.values():
        try:
            handle.remove()
        except Exception:
            pass
    state.frame_name_handles.clear()


def create_link_frame_visuals(server: viser.ViserServer, state: ViewerState) -> None:
    if state.current_urdf is None or state.current_root_name is None:
        return

    remove_link_frame_visuals(state)

    urdf = state.current_urdf._urdf
    for link_name in urdf.link_map.keys():
        safe_link_name = link_name.replace("/", "_")

        frame_handle = server.scene.add_frame(
            f"{state.current_root_name}/frames/{safe_link_name}",
            axes_length=0.12,
            axes_radius=0.006,
            origin_radius=0.01,
            visible=state.show_link_frames,
        )
        name_handle = server.scene.add_3d_gui_container(
            f"{state.current_root_name}/frame_names/{safe_link_name}",
            visible=state.show_frame_names,
        )
        with name_handle:
            server.gui.add_html(
                (
                    '<div style="'
                    "display: inline-block; width: fit-content; "
                    "background: transparent; color: #111; "
                    "font-family: ui-monospace, SFMono-Regular, Menlo, monospace; "
                    "font-size: 12px; font-weight: 500; "
                    "line-height: 1; padding: 0; margin: 0; white-space: nowrap;"
                    '">'
                    f"{link_name}"
                    "</div>"
                )
            )

        state.link_frame_handles[link_name] = frame_handle
        state.frame_name_handles[link_name] = name_handle

    update_link_frame_visuals(state)


def ensure_link_frame_visuals(server: viser.ViserServer, state: ViewerState) -> None:
    if state.current_urdf is None or state.current_root_name is None:
        return
    if state.link_frame_handles and state.frame_name_handles:
        return
    create_link_frame_visuals(server, state)


def compute_ground_plane_size(state: ViewerState) -> tuple[int, int]:
    if state.current_urdf is None:
        return (2, 2)

    xs: list[float] = []
    ys: list[float] = []
    urdf = state.current_urdf._urdf

    link_names = list(urdf.link_map.keys())
    if len(link_names) > _MAX_GROUND_PLANE_SAMPLE_LINKS:
        step = max(1, len(link_names) // _MAX_GROUND_PLANE_SAMPLE_LINKS)
        link_names = link_names[::step]

    for link_name in link_names:
        try:
            transform = urdf.get_transform(link_name)
        except Exception:
            continue
        xs.append(float(transform[0, 3]))
        ys.append(float(transform[1, 3]))

    if not xs or not ys:
        return (2, 2)

    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)

    margin = 1.0
    width = max(2, int(math.ceil((span_x + margin) / 2.0)) * 2)
    height = max(2, int(math.ceil((span_y + margin) / 2.0)) * 2)

    return (width, height)


def set_ground_plane_visible(
    server: viser.ViserServer,
    state: ViewerState,
    visible: bool,
) -> None:
    if not visible:
        if state.ground_plane_handle is not None:
            state.ground_plane_handle.visible = False
        return

    width, height = compute_ground_plane_size(state)

    if state.ground_plane_handle is not None and state.ground_plane_size == (
        width,
        height,
    ):
        state.ground_plane_handle.visible = True
        return

    if state.ground_plane_handle is not None:
        try:
            state.ground_plane_handle.remove()
        except Exception:
            try:
                server.scene.remove_by_name("/grid")
            except Exception:
                pass
        state.ground_plane_handle = None

    state.ground_plane_handle = server.scene.add_grid(
        "/grid",
        width=width,
        height=height,
        position=(0.0, 0.0, 0.0),
        infinite_grid=False,
    )
    state.ground_plane_handle.visible = True
    state.ground_plane_size = (width, height)


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
