from __future__ import annotations

import math
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

    _update_transform_display(state)


def _update_transform_display(state: ViewerState) -> None:
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


def _compute_ground_plane_size(state: ViewerState) -> tuple[int, int]:
    if state.current_urdf is None:
        return (2, 2)

    xs: list[float] = []
    ys: list[float] = []
    urdf = state.current_urdf._urdf
    for link_name in urdf.link_map.keys():
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

    # Add a border around the robot and quantize so tiny motions do not cause
    # repeated grid re-creation.
    margin = 1.0
    width = max(2, int(math.ceil(span_x + margin)))
    height = max(2, int(math.ceil(span_y + margin)))

    return (width, height)


def _set_ground_plane_visible(
    server: viser.ViserServer,
    state: ViewerState,
    visible: bool,
) -> None:
    if not visible:
        if state.ground_plane_handle is not None:
            state.ground_plane_handle.visible = False
        return

    width, height = _compute_ground_plane_size(state)

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


def _apply_joint_configuration(
    state: ViewerState,
    values: np.ndarray,
    *,
    update_sliders: bool,
) -> None:
    """Apply all joint values in one batched update."""
    if state.current_urdf is None:
        return

    if update_sliders and state.slider_handles is not None:
        state.suppress_slider_callbacks = True
        try:
            for slider, value in zip(state.slider_handles, values):
                slider.value = float(value)
        finally:
            state.suppress_slider_callbacks = False

    state.current_urdf.update_cfg(values)
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
            if state.suppress_slider_callbacks:
                return
            if state.ik_enabled:
                return
            viser_urdf.update_cfg(np.array([h.value for h in handles]))
            update_link_frame_visuals(state)

        slider.on_update(_on_slider_update)
        slider_handles.append(slider)
        joint_names.append(joint_name)
        initial_config.append(initial_pos)
        joint_limits.append((lower, upper))

    return slider_handles, joint_names, initial_config, joint_limits


def clear_previous_robot(server: viser.ViserServer, state: ViewerState) -> None:
    state.ik_enabled = False

    if state.ground_plane_handle is not None:
        try:
            state.ground_plane_handle.remove()
        except Exception:
            try:
                server.scene.remove_by_name("/grid")
            except Exception:
                pass
        state.ground_plane_handle = None
        state.ground_plane_size = (0, 0)

    if state.cartesian_target_handle is not None:
        try:
            state.cartesian_target_handle.remove()
        except Exception:
            pass
        state.cartesian_target_handle = None

    if state.current_root_name is not None:
        try:
            # Remove the whole robot subtree in one call to avoid duplicate
            # child removals that trigger viser warnings.
            server.scene.remove_by_name(state.current_root_name)
        except Exception:
            pass

    state.current_urdf = None
    state.current_root_name = None
    state.link_frame_handles.clear()
    state.frame_name_handles.clear()

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

    if state.transform_folder_handle is not None:
        try:
            state.transform_folder_handle.remove()
        except Exception:
            pass
        state.transform_folder_handle = None

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
    state.ground_plane_size = (0, 0)


def load_urdf_file(
    server: viser.ViserServer,
    state: ViewerState,
    path: str,
    status_text: Any,
    load_meshes: bool,
) -> None:
    clear_previous_robot(server, state)

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
            "Meshes",
            initial_value=state.show_visual_meshes,
        )
        show_frames_cb = server.gui.add_checkbox(
            "Frames",
            initial_value=state.show_link_frames,
        )
        show_frame_names_cb = server.gui.add_checkbox(
            "Frame names",
            initial_value=state.show_frame_names,
        )
        show_ground_cb = server.gui.add_checkbox(
            "Ground plane",
            initial_value=state.show_ground_plane,
        )

    state.visibility_visual_checkbox = show_meshes_cb
    state.visibility_frames_checkbox = show_frames_cb
    state.visibility_frame_names_checkbox = show_frame_names_cb
    state.visibility_ground_checkbox = show_ground_cb

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

    @show_ground_cb.on_update
    def _show_ground(_: object) -> None:
        state.show_ground_plane = show_ground_cb.value
        _set_ground_plane_visible(server, state, state.show_ground_plane)

    show_meshes_cb.visible = load_meshes

    state.control_folder_handle = server.gui.add_folder("Joint Control")
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

        randomize_button = server.gui.add_button("Randomize")
        state.randomize_button = randomize_button

        def _on_randomize(_: object) -> None:
            if state.slider_handles is None or state.joint_limits is None:
                return

            randomized = np.array(
                [
                    np.random.uniform(lower, upper)
                    for lower, upper in state.joint_limits
                ],
                dtype=float,
            )
            _apply_joint_configuration(
                state,
                randomized,
                update_sliders=True,
            )

        randomize_button.on_click(_on_randomize)

        reset_button = server.gui.add_button("Reset")
        state.reset_button = reset_button

        def _on_reset(_: object) -> None:
            if state.slider_handles is None or state.initial_config is None:
                return

            reset_cfg = np.array(state.initial_config, dtype=float)
            _apply_joint_configuration(
                state,
                reset_cfg,
                update_sliders=True,
            )

        reset_button.on_click(_on_reset)

    state.cartesian_folder_handle = server.gui.add_folder("Cartesian Control")
    with state.cartesian_folder_handle:
        cartesian_mode_checkbox = server.gui.add_checkbox("Enable", initial_value=False)
        state.cartesian_mode_checkbox = cartesian_mode_checkbox

        setup_cartesian_controls(
            server, state, path, status_text, cartesian_mode_checkbox
        )

    state.transform_folder_handle = server.gui.add_folder("Get Transform")
    with state.transform_folder_handle:
        frame_options = list(urdf.link_map.keys())
        initial_frame = frame_options[0]
        initial_to_frame = initial_frame
        if state.cartesian_frame_dropdown is not None:
            cartesian_target_frame = state.cartesian_frame_dropdown.value
            if cartesian_target_frame in frame_options:
                initial_to_frame = cartesian_target_frame

        from_dropdown = server.gui.add_dropdown(
            "From",
            options=frame_options,
            initial_value=initial_frame,
        )
        to_dropdown = server.gui.add_dropdown(
            "To",
            options=frame_options,
            initial_value=initial_to_frame,
        )
        translation_text = server.gui.add_text("Translation (x,y,z)", "")
        rotation_text = server.gui.add_text("Rotation (w,x,y,z)", "")

        state.transform_from_dropdown = from_dropdown
        state.transform_to_dropdown = to_dropdown
        state.transform_translation_text = translation_text
        state.transform_rotation_text = rotation_text

        @from_dropdown.on_update
        def _on_transform_frame_change(_event: object) -> None:
            _update_transform_display(state)

        @to_dropdown.on_update
        def _on_transform_target_change(_event: object) -> None:
            _update_transform_display(state)

        @translation_text.on_update
        def _on_translation_text_edit(_event: object) -> None:
            if state.suppress_transform_text_callbacks:
                return
            _update_transform_display(state)

        @rotation_text.on_update
        def _on_rotation_text_edit(_event: object) -> None:
            if state.suppress_transform_text_callbacks:
                return
            _update_transform_display(state)

    _update_transform_display(state)

    _set_ground_plane_visible(server, state, state.show_ground_plane)
