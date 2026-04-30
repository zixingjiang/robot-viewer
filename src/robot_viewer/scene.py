from __future__ import annotations

import re
from typing import Any

import numpy as np
import viser

from .state import RobotInstance, ViewerState
from .utils import rotation_matrix_to_wxyz


_ROBOT_ROOT_RE = re.compile(r"^/robot(?:_[0-9a-f]+)?(?:/|$)")


def update_transform_display(robot: RobotInstance) -> None:
    if (
        robot.transform_from_dropdown is None
        or robot.transform_to_dropdown is None
        or robot.transform_translation_text is None
        or robot.transform_rotation_text is None
    ):
        return

    from_frame = robot.transform_from_dropdown.value
    to_frame = robot.transform_to_dropdown.value

    try:
        urdf = robot.urdf._urdf
        world_from = urdf.get_transform(from_frame)
        world_to = urdf.get_transform(to_frame)
        from_to = np.linalg.inv(world_from) @ world_to

        translation = from_to[:3, 3]
        rotation_wxyz = rotation_matrix_to_wxyz(from_to[:3, :3])

        robot.suppress_transform_text_callbacks = True
        try:
            robot.transform_translation_text.value = (
                f"{translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}"
            )
            robot.transform_rotation_text.value = (
                f"{rotation_wxyz[0]:.4f}, {rotation_wxyz[1]:.4f}, "
                f"{rotation_wxyz[2]:.4f}, {rotation_wxyz[3]:.4f}"
            )
        finally:
            robot.suppress_transform_text_callbacks = False
    except Exception:
        robot.suppress_transform_text_callbacks = True
        try:
            robot.transform_translation_text.value = "N/A"
            robot.transform_rotation_text.value = "N/A"
        finally:
            robot.suppress_transform_text_callbacks = False


def update_link_frame_visuals(robot: RobotInstance) -> None:
    urdf = robot.urdf._urdf
    for link_name, frame_handle in robot.link_frame_handles.items():
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
        frame_handle.visible = robot.show_link_frames

        name_handle = robot.frame_name_handles.get(link_name)
        if name_handle is not None:
            name_handle.position = (
                float(transform[0, 3]),
                float(transform[1, 3]),
                float(transform[2, 3]),
            )
            name_handle.visible = robot.show_frame_names

    update_transform_display(robot)


def remove_link_frame_visuals(robot: RobotInstance) -> None:
    for handle in robot.link_frame_handles.values():
        try:
            handle.remove()
        except Exception:
            pass
    robot.link_frame_handles.clear()

    for handle in robot.frame_name_handles.values():
        try:
            handle.remove()
        except Exception:
            pass
    robot.frame_name_handles.clear()


def create_link_frame_visuals(
    server: viser.ViserServer, robot: RobotInstance
) -> None:
    if robot.urdf is None or robot.root_name is None:
        return

    remove_link_frame_visuals(robot)

    urdf = robot.urdf._urdf
    for link_name in urdf.link_map.keys():
        safe_link_name = link_name.replace("/", "_")

        frame_handle = server.scene.add_frame(
            f"{robot.root_name}/frames/{safe_link_name}",
            axes_length=0.12,
            axes_radius=0.006,
            origin_radius=0.01,
            visible=robot.show_link_frames,
        )
        name_handle = server.scene.add_3d_gui_container(
            f"{robot.root_name}/frame_names/{safe_link_name}",
            visible=robot.show_frame_names,
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

        robot.link_frame_handles[link_name] = frame_handle
        robot.frame_name_handles[link_name] = name_handle

    update_link_frame_visuals(robot)


def ensure_link_frame_visuals(
    server: viser.ViserServer, robot: RobotInstance
) -> None:
    if robot.link_frame_handles and robot.frame_name_handles:
        return
    create_link_frame_visuals(server, robot)


def set_ground_plane_visible(
    server: viser.ViserServer,
    state: ViewerState,
    visible: bool,
) -> None:
    if not visible:
        if state.ground_plane_handle is not None:
            state.ground_plane_handle.visible = False
        return

    if state.ground_plane_handle is not None:
        state.ground_plane_handle.visible = True
        return

    width, height = state.ground_plane_size
    state.ground_plane_handle = server.scene.add_grid(
        "/grid",
        width=width,
        height=height,
        position=(0.0, 0.0, 0.0),
        infinite_grid=False,
    )
    state.ground_plane_handle.visible = True


def set_world_frame_visible(
    server: viser.ViserServer,
    state: ViewerState,
    visible: bool,
) -> None:
    server.scene.world_axes.visible = visible


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
    active_roots = {robot.root_name for robot in state.robots.values()}

    for root_name in discovered_roots:
        if root_name in active_roots:
            continue
        try:
            server.scene.remove_by_name(root_name)
        except Exception:
            continue


def remove_robot(server: viser.ViserServer, state: ViewerState, name: str) -> None:
    robot = state.robots.get(name)
    if robot is None:
        return

    robot.ik_enabled = False

    if robot.tab_handle is not None:
        try:
            tab_parent = robot.tab_handle._parent
            robot.tab_handle.remove()
            tab_parent._tab_container_ids = tuple(
                h._id for h in tab_parent._tab_handles
            )
        except Exception:
            pass

    if robot.remove_button_handle is not None:
        try:
            robot.remove_button_handle.remove()
        except Exception:
            pass

    if robot.root_control_handle is not None:
        try:
            robot.root_control_handle.remove()
        except Exception:
            pass

    try:
        server.scene.remove_by_name(robot.root_name)
        removed_root = True
    except Exception:
        removed_root = False

    if not removed_root:
        if robot.cartesian_target_handle is not None:
            try:
                robot.cartesian_target_handle.remove()
            except Exception:
                pass
        try:
            robot.urdf.remove()
        except Exception:
            pass

    remove_link_frame_visuals(robot)
    del state.robots[name]


def clear_all_robots(server: viser.ViserServer, state: ViewerState) -> None:
    for name in list(state.robots.keys()):
        remove_robot(server, state, name)
