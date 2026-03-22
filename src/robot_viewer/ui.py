from __future__ import annotations

import io
import math
import re
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
from .robot_data import (
    get_robot_description_candidates,
    load_robot_description_urdf,
    load_urdf,
)
from .scene_sync import update_link_frame_visuals, update_transform_display
from .state import ViewerState


_MAX_GROUND_PLANE_SAMPLE_LINKS = 128
_ROBOT_ROOT_RE = re.compile(r"^/robot(?:_[0-9a-f]+)?(?:/|$)")


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
    """Drop persisted robot messages so reconnect replay keeps only fresh state."""
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
        pass


def prune_stale_robot_roots(server: viser.ViserServer, state: ViewerState) -> None:
    """Remove any robot roots except the currently active one."""
    discovered_roots = _discover_robot_roots(server)
    keep_root = state.current_root_name

    for root_name in discovered_roots:
        if keep_root is not None and root_name == keep_root:
            continue
        try:
            server.scene.remove_by_name(root_name)
        except Exception:
            pass

    if keep_root is None:
        state.robot_root_names.clear()
    else:
        state.robot_root_names = {keep_root}


def _load_robot_into_viewer(
    server: viser.ViserServer,
    state: ViewerState,
    urdf: yourdfpy.URDF,
    source_path: str,
    status_text: Any,
    load_meshes: bool,
) -> None:
    prune_stale_robot_roots(server, state)
    clear_previous_robot(server, state)
    _purge_robot_replay_messages(server)

    # Keep a single stable root so every load replaces the same subtree.
    root_node_name = "/robot"
    state.current_root_name = root_node_name
    state.robot_root_names.add(root_node_name)

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

    # Creating one frame + one label per link can be expensive on large robots.
    # Defer this until the user enables frame overlays.
    if state.show_link_frames or state.show_frame_names:
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
            server, state, source_path, status_text, cartesian_mode_checkbox
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
            update_transform_display(state)

        @to_dropdown.on_update
        def _on_transform_target_change(_event: object) -> None:
            update_transform_display(state)

        @translation_text.on_update
        def _on_translation_text_edit(_event: object) -> None:
            if state.suppress_transform_text_callbacks:
                return
            update_transform_display(state)

        @rotation_text.on_update
        def _on_rotation_text_edit(_event: object) -> None:
            if state.suppress_transform_text_callbacks:
                return
            update_transform_display(state)

    update_transform_display(state)

    _set_ground_plane_visible(server, state, state.show_ground_plane)
    try:
        server.flush()
    except Exception:
        pass


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
    """Create frame visuals on demand when any frame overlay is enabled."""
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

    previous_urdf = state.current_urdf
    did_scene_reset = False

    # Force a full live-scene cleanup for already connected tabs so stale
    # robot nodes cannot remain visible client-side.
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

        # Remove all known robot roots so reconnecting clients cannot replay stale
        # robots that were previously loaded.
        robot_roots = set(state.robot_root_names)
        if state.current_root_name is not None:
            robot_roots.add(state.current_root_name)

        # Viser keeps authoritative scene state server-side. Discover any leftover
        # robot roots from that registry so reconnecting tabs cannot replay stale
        # robots even if local tracking missed an earlier root.
        robot_roots.update(_discover_robot_roots(server))

        removed_any_root = False
        for root_name in robot_roots:
            try:
                # Remove the whole robot subtree in one call to avoid duplicate
                # child removals that trigger viser warnings.
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


def setup_viewer_actions(
    server: viser.ViserServer,
    status_text_getter: Callable[[], Any | None] | None = None,
) -> Any:
    with server.gui.add_folder("Viewer"):
        status_text = server.gui.add_text("Status", "Open a URDF file to begin.")
        reset_view_button = server.gui.add_button("Reset View")
        save_canvas_button = server.gui.add_button("Save Canvas")

    def _set_status(message: str) -> None:
        status_handle = status_text
        if status_text_getter is not None:
            maybe_handle = status_text_getter()
            if maybe_handle is not None:
                status_handle = maybe_handle
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

        width = int(client.camera.image_width)
        height = int(client.camera.image_height)
        if width <= 0 or height <= 0:
            width = 1280
            height = 720

        try:
            image = client.get_render(
                height=height,
                width=width,
                transport_format="png",
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


def load_urdf_file(
    server: viser.ViserServer,
    state: ViewerState,
    path: str,
    status_text: Any,
    load_meshes: bool,
) -> None:
    urdf = load_urdf(path, load_meshes)

    _load_robot_into_viewer(
        server,
        state,
        urdf,
        path,
        status_text,
        load_meshes,
    )


def load_robot_description(
    server: viser.ViserServer,
    state: ViewerState,
    description_name: str,
    status_text: Any,
    load_meshes: bool,
) -> str:
    resolved_name, urdf, urdf_path = load_robot_description_urdf(description_name)

    _load_robot_into_viewer(
        server,
        state,
        urdf,
        urdf_path,
        status_text,
        load_meshes,
    )

    return resolved_name
