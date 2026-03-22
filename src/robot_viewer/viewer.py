from __future__ import annotations

import io
import math
import time
from typing import Any, Callable

import imageio.v3 as iio
import numpy as np
import viser
import yourdfpy  # type: ignore[import]
from viser._gui_handles import (
    GuiButtonHandle,
    GuiDropdownHandle,
    GuiEvent,
    GuiUploadButtonHandle,
)
from viser.extras import ViserUrdf

from .ik import setup_cartesian_controls
from .loader import get_robot_description_candidates, prepare_scene_for_reload
from .state import ViewerState
from .utils import rotation_matrix_to_wxyz


_MAX_GROUND_PLANE_SAMPLE_LINKS = 128


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


def _ensure_link_frame_visuals(server: viser.ViserServer, state: ViewerState) -> None:
    if state.current_urdf is None or state.current_root_name is None:
        return
    if state.link_frame_handles and state.frame_name_handles:
        return
    _create_link_frame_visuals(server, state)


def _compute_ground_plane_size(state: ViewerState) -> tuple[int, int]:
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
    slider_handles: list[viser.GuiInputHandle[float]] = []
    joint_names: list[str] = []
    initial_config: list[float] = []
    joint_limits: list[tuple[float, float]] = []

    for joint_name, (lower, upper) in viser_urdf.get_actuated_joint_limits().items():
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi

        initial_pos = min(max(0.0, lower), upper)

        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=initial_pos,
        )

        def _on_slider_update(_event: object, handles=slider_handles):
            if state.suppress_slider_callbacks or state.ik_enabled:
                return
            viser_urdf.update_cfg(np.array([h.value for h in handles]))
            update_link_frame_visuals(state)

        slider.on_update(_on_slider_update)
        slider_handles.append(slider)
        joint_names.append(joint_name)
        initial_config.append(initial_pos)
        joint_limits.append((lower, upper))

    return slider_handles, joint_names, initial_config, joint_limits


def load_robot_into_viewer(
    server: viser.ViserServer,
    state: ViewerState,
    urdf: yourdfpy.URDF,
    source_path: str,
    status_text: Any,
    load_meshes: bool,
) -> None:
    prepare_scene_for_reload(server, state)

    state.current_root_name = "/robot"
    state.robot_root_names.add(state.current_root_name)

    viser_urdf = ViserUrdf(
        server,
        urdf,
        root_node_name=state.current_root_name,
        load_meshes=load_meshes,
        load_collision_meshes=False,
    )
    state.current_urdf = viser_urdf
    state.show_visual_meshes = state.show_visual_meshes and load_meshes
    if load_meshes:
        viser_urdf.show_visual = state.show_visual_meshes

    if state.show_link_frames or state.show_frame_names:
        _create_link_frame_visuals(server, state)

    state.visibility_folder_handle = server.gui.add_folder("Visibility")
    with state.visibility_folder_handle:
        show_meshes_cb = server.gui.add_checkbox(
            "Meshes", initial_value=state.show_visual_meshes
        )
        show_frames_cb = server.gui.add_checkbox(
            "Frames", initial_value=state.show_link_frames
        )
        show_frame_names_cb = server.gui.add_checkbox(
            "Frame names", initial_value=state.show_frame_names
        )
        show_ground_cb = server.gui.add_checkbox(
            "Ground plane", initial_value=state.show_ground_plane
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
        if state.show_link_frames:
            _ensure_link_frame_visuals(server, state)
        update_link_frame_visuals(state)

    @show_frame_names_cb.on_update
    def _show_frame_names(_: object) -> None:
        state.show_frame_names = show_frame_names_cb.value
        if state.show_frame_names:
            _ensure_link_frame_visuals(server, state)
        update_link_frame_visuals(state)

    @show_ground_cb.on_update
    def _show_ground(_: object) -> None:
        state.show_ground_plane = show_ground_cb.value
        _set_ground_plane_visible(server, state, state.show_ground_plane)

    show_meshes_cb.visible = load_meshes

    state.control_folder_handle = server.gui.add_folder("Joint Control")
    with state.control_folder_handle:
        sliders, names, initial_cfg, limits = create_robot_control_sliders(
            server, viser_urdf, state
        )
        state.slider_handles = sliders
        state.joint_names = names
        state.initial_config = initial_cfg
        state.joint_limits = limits

        if sliders:
            viser_urdf.update_cfg(np.zeros(len(sliders)))

        randomize_button = server.gui.add_button("Randomize")
        state.randomize_button = randomize_button

        @randomize_button.on_click
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
            _apply_joint_configuration(state, randomized, update_sliders=True)

        reset_button = server.gui.add_button("Reset")
        state.reset_button = reset_button

        @reset_button.on_click
        def _on_reset(_: object) -> None:
            if state.slider_handles is None or state.initial_config is None:
                return
            _apply_joint_configuration(
                state,
                np.array(state.initial_config, dtype=float),
                update_sliders=True,
            )

    state.cartesian_folder_handle = server.gui.add_folder("Cartesian Control")
    with state.cartesian_folder_handle:
        cartesian_mode_checkbox = server.gui.add_checkbox("Enable", initial_value=False)
        state.cartesian_mode_checkbox = cartesian_mode_checkbox
        setup_cartesian_controls(
            server, state, source_path, status_text, cartesian_mode_checkbox
        )

    state.transform_folder_handle = server.gui.add_folder("Get Transform")
    with state.transform_folder_handle:
        frame_options = list(urdf.link_map.keys())
        initial_frame = frame_options[0]
        initial_to_frame = initial_frame
        if state.cartesian_frame_dropdown is not None:
            target_frame = state.cartesian_frame_dropdown.value
            if target_frame in frame_options:
                initial_to_frame = target_frame

        from_dropdown = server.gui.add_dropdown(
            "From", options=frame_options, initial_value=initial_frame
        )
        to_dropdown = server.gui.add_dropdown(
            "To", options=frame_options, initial_value=initial_to_frame
        )
        translation_text = server.gui.add_text("Translation (x,y,z)", "")
        rotation_text = server.gui.add_text("Rotation (w,x,y,z)", "")

        state.transform_from_dropdown = from_dropdown
        state.transform_to_dropdown = to_dropdown
        state.transform_translation_text = translation_text
        state.transform_rotation_text = rotation_text

        @from_dropdown.on_update
        def _on_transform_frame_change(_event: object) -> None:
            update_transform_display(state)

        @to_dropdown.on_update
        def _on_transform_target_change(_event: object) -> None:
            update_transform_display(state)

        @translation_text.on_update
        def _on_translation_text_edit(_event: object) -> None:
            if not state.suppress_transform_text_callbacks:
                update_transform_display(state)

        @rotation_text.on_update
        def _on_rotation_text_edit(_event: object) -> None:
            if not state.suppress_transform_text_callbacks:
                update_transform_display(state)

    update_transform_display(state)
    _set_ground_plane_visible(server, state, state.show_ground_plane)
    try:
        server.flush()
    except Exception:
        pass


def setup_viewer_actions(
    server: viser.ViserServer,
    status_text_getter: Callable[[], Any | None] | None = None,
) -> Any:
    with server.gui.add_folder("Viewer"):
        status_text = server.gui.add_text("Status", "Open a URDF file to begin.")
        reset_view_button = server.gui.add_button("Reset View")
        save_canvas_button = server.gui.add_button("Save Canvas")

    def _set_status(message: str) -> None:
        status_handle = (
            status_text_getter() if status_text_getter is not None else status_text
        )
        if status_handle is not None:
            status_handle.value = message

    @reset_view_button.on_click
    def _on_reset_view(event: GuiEvent[GuiButtonHandle]) -> None:
        if event.client is None:
            _set_status("No active client to reset view.")
            return

        initial_camera = server.initial_camera
        client = event.client
        client.camera.position = initial_camera.position
        client.camera.look_at = initial_camera.look_at
        if initial_camera.up is not None:
            client.camera.up_direction = initial_camera.up
        client.camera.fov = float(initial_camera.fov)
        client.camera.near = float(initial_camera.near)
        client.camera.far = float(initial_camera.far)
        _set_status("View reset.")

    @save_canvas_button.on_click
    def _on_save_canvas(event: GuiEvent[GuiButtonHandle]) -> None:
        client = event.client
        if client is None:
            _set_status("No active client to save canvas.")
            return

        width = int(client.camera.image_width) or 1280
        height = int(client.camera.image_height) or 720

        try:
            image = client.get_render(
                height=height, width=width, transport_format="png"
            )
            buffer = io.BytesIO()
            iio.imwrite(buffer, image, extension=".png")
            filename = f"robot_viewer_canvas_{time.strftime('%Y%m%d_%H%M%S')}.png"
            client.send_file_download(
                filename=filename,
                content=buffer.getvalue(),
                save_immediately=True,
            )
            _set_status(f"Saved canvas as {filename}.")
        except Exception as exc:
            _set_status(f"Failed to save canvas: {exc!r}")

    return status_text


def setup_file_actions(
    server: viser.ViserServer,
) -> tuple[
    Any,
    GuiUploadButtonHandle,
    GuiDropdownHandle[str] | None,
    GuiButtonHandle | None,
]:
    with server.gui.add_folder("Files"):
        file_text = server.gui.add_text("Filename", "No file loaded.")

        upload_button = server.gui.add_upload_button(
            "Open URDF",
            mime_type="*/*",
            hint="Select a URDF file (.urdf, .xml).",
        )

        available_descriptions = get_robot_description_candidates()
        description_dropdown: GuiDropdownHandle[str] | None = None
        load_description_button: GuiButtonHandle | None = None
        if available_descriptions:
            description_dropdown = server.gui.add_dropdown(
                "robot_descriptions",
                options=available_descriptions,
                initial_value=available_descriptions[0],
            )
            load_description_button = server.gui.add_button(
                "Load robot_descriptions entry"
            )
        else:
            server.gui.add_markdown(
                "Install `robot_descriptions` to browse robots from that catalog."
            )

    return file_text, upload_button, description_dropdown, load_description_button
