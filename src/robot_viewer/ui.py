from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import viser
import yourdfpy  # type: ignore[import]
from viser.extras import ViserUrdf

from .ik import setup_cartesian_controls
from .state import ViewerState
from .utils import rotation_matrix_to_wxyz


def _remove_link_frame_visuals(state: ViewerState) -> None:
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


def _create_link_frame_visuals(server: viser.ViserServer, state: ViewerState) -> None:
    if state.current_urdf is None or state.current_root_name is None:
        return

    _remove_link_frame_visuals(state)

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


def create_robot_control_sliders(
    server: viser.ViserServer, viser_urdf: ViserUrdf, state: ViewerState
) -> tuple[
    list[viser.GuiInputHandle[float]],
    list[str],
    list[float],
    list[tuple[float, float]],
]:
    """Create sliders to control joints for a loaded URDF."""

    slider_handles: list[viser.GuiInputHandle[float]] = []
    joint_names: list[str] = []
    initial_config: list[float] = []
    joint_limits: list[tuple[float, float]] = []

    for joint_name, (lower, upper) in viser_urdf.get_actuated_joint_limits().items():
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi

        initial_pos = 0.0
        if initial_pos < lower:
            initial_pos = lower
        elif initial_pos > upper:
            initial_pos = upper

        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=initial_pos,
        )

        def _on_slider_update(_event: object, handles=slider_handles):
            if state.ik_enabled and not state.suppress_slider_callbacks:
                return
            viser_urdf.update_cfg(np.array([h.value for h in handles]))
            update_link_frame_visuals(state)

        slider.on_update(_on_slider_update)
        slider_handles.append(slider)
        joint_names.append(joint_name)
        initial_config.append(initial_pos)
        joint_limits.append((lower, upper))

    return slider_handles, joint_names, initial_config, joint_limits


def clear_previous_robot(state: ViewerState) -> None:
    state.ik_enabled = False

    if state.cartesian_target_handle is not None:
        try:
            state.cartesian_target_handle.remove()
        except Exception:
            pass
        state.cartesian_target_handle = None

    if state.current_urdf is not None:
        try:
            state.current_urdf.remove()
        except Exception:
            pass
        state.current_urdf = None

    _remove_link_frame_visuals(state)

    if state.control_folder_handle is not None:
        try:
            state.control_folder_handle.remove()
        except Exception:
            pass
        state.control_folder_handle = None

    if state.visibility_folder_handle is not None:
        try:
            state.visibility_folder_handle.remove()
        except Exception:
            pass
        state.visibility_folder_handle = None

    if state.cartesian_folder_handle is not None:
        try:
            state.cartesian_folder_handle.remove()
        except Exception:
            pass
        state.cartesian_folder_handle = None

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
    state.cartesian_frame_dropdown = None
    state.ik_configuration = None
    state.ik_tasks = None
    state.ik_frame_task = None
    state.ik_posture_task = None
    state.ik_damping_task = None
    state.ik_joint_name_to_q_index = {}
    state.ik_solver = None


def load_urdf_file(
    server: viser.ViserServer,
    state: ViewerState,
    path: str,
    status_text: Any,
    load_meshes: bool,
    show_grid: bool,
) -> None:
    clear_previous_robot(state)

    root_node_name = f"/robot_{int(time.time() * 1000)}"
    state.current_root_name = root_node_name

    urdf = yourdfpy.URDF.load(
        path,
        load_meshes=load_meshes,
        load_collision_meshes=False,
        mesh_dir=os.path.dirname(path),
    )

    viser_urdf = ViserUrdf(
        server,
        urdf,
        root_node_name=root_node_name,
        load_meshes=load_meshes,
        load_collision_meshes=False,
    )
    state.current_urdf = viser_urdf
    state.show_visual_meshes = state.show_visual_meshes and load_meshes
    if load_meshes:
        viser_urdf.show_visual = state.show_visual_meshes

    _create_link_frame_visuals(server, state)

    state.visibility_folder_handle = server.gui.add_folder("Visibility")
    with state.visibility_folder_handle:
        show_meshes_cb = server.gui.add_checkbox(
            "Show meshes",
            initial_value=state.show_visual_meshes,
        )
        show_frames_cb = server.gui.add_checkbox(
            "Show frames",
            initial_value=state.show_link_frames,
        )
        show_frame_names_cb = server.gui.add_checkbox(
            "Show frame names",
            initial_value=state.show_frame_names,
        )

    state.visibility_visual_checkbox = show_meshes_cb
    state.visibility_frames_checkbox = show_frames_cb
    state.visibility_frame_names_checkbox = show_frame_names_cb

    @show_meshes_cb.on_update
    def _show_meshes(_: object) -> None:
        state.show_visual_meshes = show_meshes_cb.value
        if load_meshes:
            viser_urdf.show_visual = state.show_visual_meshes

    @show_frames_cb.on_update
    def _show_frames(_: object) -> None:
        state.show_link_frames = show_frames_cb.value
        update_link_frame_visuals(state)

    @show_frame_names_cb.on_update
    def _show_frame_names(_: object) -> None:
        state.show_frame_names = show_frame_names_cb.value
        update_link_frame_visuals(state)

    show_meshes_cb.visible = load_meshes

    state.control_folder_handle = server.gui.add_folder("Robot joint controls")
    with state.control_folder_handle:
        slider_handles, joint_names, initial_config, joint_limits = (
            create_robot_control_sliders(server, viser_urdf, state)
        )
        state.slider_handles = slider_handles
        state.joint_names = joint_names
        state.initial_config = initial_config
        state.joint_limits = joint_limits

        if slider_handles:
            viser_urdf.update_cfg(np.zeros(len(slider_handles)))

        randomize_button = server.gui.add_button("Randomize joints")
        state.randomize_button = randomize_button

        def _on_randomize(_: object) -> None:
            if state.slider_handles is None or state.joint_limits is None:
                return

            for slider, (lower, upper) in zip(state.slider_handles, state.joint_limits):
                slider.value = float(np.random.uniform(lower, upper))

        randomize_button.on_click(_on_randomize)

        reset_button = server.gui.add_button("Reset joints")
        state.reset_button = reset_button

        def _on_reset(_: object) -> None:
            if state.slider_handles is None or state.initial_config is None:
                return

            for slider, init in zip(state.slider_handles, state.initial_config):
                slider.value = init

        reset_button.on_click(_on_reset)

    state.cartesian_folder_handle = server.gui.add_folder("Cartesian controls")
    with state.cartesian_folder_handle:
        cartesian_mode_checkbox = server.gui.add_checkbox(
            "Cartesian control mode", initial_value=False
        )
        state.cartesian_mode_checkbox = cartesian_mode_checkbox

        setup_cartesian_controls(
            server, state, path, status_text, cartesian_mode_checkbox
        )

    if show_grid:
        try:
            server.scene.remove_by_name("/grid")
        except Exception:
            pass

        trimesh_scene = viser_urdf._urdf.scene or viser_urdf._urdf.collision_scene
        server.scene.add_grid(
            "/grid",
            width=2,
            height=2,
            position=(
                0.0,
                0.0,
                trimesh_scene.bounds[0, 2] if trimesh_scene is not None else 0.0,
            ),
        )
