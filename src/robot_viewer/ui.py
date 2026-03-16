from __future__ import annotations

import os
import time

import numpy as np
import viser
import yourdfpy  # type: ignore[import]
from viser.extras import ViserUrdf

from .ik import setup_cartesian_controls
from .state import ViewerState


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

    if state.control_folder_handle is not None:
        try:
            state.control_folder_handle.remove()
        except Exception:
            pass
        state.control_folder_handle = None

    state.slider_handles = None
    state.joint_names = None
    state.initial_config = None
    state.joint_limits = None
    state.randomize_button = None
    state.reset_button = None
    state.cartesian_mode_checkbox = None
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
    status_text: object,
    load_meshes: bool,
    load_collision_meshes: bool,
    show_grid: bool,
) -> None:
    clear_previous_robot(state)

    root_node_name = f"/robot_{int(time.time() * 1000)}"
    state.current_root_name = root_node_name

    urdf = yourdfpy.URDF.load(
        path,
        load_meshes=load_meshes,
        load_collision_meshes=load_collision_meshes,
        mesh_dir=os.path.dirname(path),
    )

    viser_urdf = ViserUrdf(
        server,
        urdf,
        root_node_name=root_node_name,
        load_meshes=load_meshes,
        load_collision_meshes=load_collision_meshes,
    )
    state.current_urdf = viser_urdf

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

        show_meshes_cb = server.gui.add_checkbox("Show meshes", viser_urdf.show_visual)
        show_collision_cb = server.gui.add_checkbox(
            "Show collision meshes", viser_urdf.show_collision
        )

        @show_meshes_cb.on_update
        def _show_meshes(_: object) -> None:
            viser_urdf.show_visual = show_meshes_cb.value

        @show_collision_cb.on_update
        def _show_collision(_: object) -> None:
            viser_urdf.show_collision = show_collision_cb.value

        show_meshes_cb.visible = load_meshes
        show_collision_cb.visible = load_collision_meshes

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
