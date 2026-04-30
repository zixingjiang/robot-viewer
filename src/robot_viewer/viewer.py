from __future__ import annotations

import io
import os
import time
from typing import Any, Callable

import imageio.v3 as iio
import numpy as np
import viser
import yourdfpy  # type: ignore[import]
from viser import _messages
from viser._gui_handles import (
    GuiButtonHandle,
    GuiDropdownHandle,
    GuiEvent,
    GuiUploadButtonHandle,
    UploadedFile,
)
from viser.extras import ViserUrdf

from .ik import setup_cartesian_controls
from .load_pipeline import execute_model_load
from .loader import get_robot_description_candidates
from .model_sources import (
    PathModelSource,
    RobotDescriptionModelSource,
    UploadedModelSource,
)
from .scene import (
    create_link_frame_visuals,
    ensure_link_frame_visuals,
    remove_robot,
    set_ground_plane_visible,
    set_world_frame_visible,
    update_link_frame_visuals,
    update_transform_display,
)
from .state import RobotInstance, ViewerState


def _apply_joint_configuration(
    robot: RobotInstance,
    values: np.ndarray,
    *,
    update_sliders: bool,
) -> None:
    if update_sliders and robot.slider_handles is not None:
        robot.suppress_slider_callbacks = True
        try:
            for slider, value in zip(robot.slider_handles, values):
                slider.value = float(value)
        finally:
            robot.suppress_slider_callbacks = False

    robot.urdf.update_cfg(values)
    update_link_frame_visuals(robot)


def create_robot_control_sliders(
    server: viser.ViserServer,
    viser_urdf: ViserUrdf,
    robot: RobotInstance,
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
            if robot.suppress_slider_callbacks or robot.ik_enabled:
                return
            viser_urdf.update_cfg(np.array([h.value for h in handles]))
            update_link_frame_visuals(robot)

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
    robot_name = os.path.splitext(os.path.basename(source_path))[0]
    base_name = robot_name
    index = 1
    while robot_name in state.robots:
        robot_name = f"{base_name}_{index}"
        index += 1

    root_name = f"/robot_{robot_name.lower().replace(' ', '_')}"

    root_frame = server.scene.add_frame(root_name, show_axes=False)
    root_frame.visible = True

    viser_urdf = ViserUrdf(
        server,
        urdf,
        root_node_name=root_name,
        load_meshes=load_meshes,
        load_collision_meshes=False,
    )

    robot = RobotInstance(
        name=robot_name,
        urdf=viser_urdf,
        root_name=root_name,
        root_frame_handle=root_frame,
        show_visual_meshes=load_meshes,
    )

    if robot.show_link_frames or robot.show_frame_names:
        create_link_frame_visuals(server, robot)

    robot.tab_handle = state.tab_group_handle.add_tab(robot_name, icon=viser.Icon.CUBE)
    with robot.tab_handle:
        robot.visibility_folder_handle = server.gui.add_folder("Visibility")
        with robot.visibility_folder_handle:
            show_meshes_cb = server.gui.add_checkbox(
                "Meshes", initial_value=robot.show_visual_meshes
            )
            show_frames_cb = server.gui.add_checkbox(
                "Frames", initial_value=robot.show_link_frames
            )
            show_frame_names_cb = server.gui.add_checkbox(
                "Frame names", initial_value=robot.show_frame_names
            )
            root_control_cb = server.gui.add_checkbox(
                "Gizmo", initial_value=robot.show_root_control
            )

        robot.visibility_visual_checkbox = show_meshes_cb
        robot.visibility_frames_checkbox = show_frames_cb
        robot.visibility_frame_names_checkbox = show_frame_names_cb
        robot.visibility_root_control_checkbox = root_control_cb

        @show_meshes_cb.on_update
        def _show_meshes(_: object) -> None:
            robot.show_visual_meshes = show_meshes_cb.value
            if load_meshes:
                viser_urdf.show_visual = robot.show_visual_meshes

        @show_frames_cb.on_update
        def _show_frames(_: object) -> None:
            robot.show_link_frames = show_frames_cb.value
            if robot.show_link_frames:
                ensure_link_frame_visuals(server, robot)
            update_link_frame_visuals(robot)

        @show_frame_names_cb.on_update
        def _show_frame_names(_: object) -> None:
            robot.show_frame_names = show_frame_names_cb.value
            if robot.show_frame_names:
                ensure_link_frame_visuals(server, robot)
            update_link_frame_visuals(robot)

        @root_control_cb.on_update
        def _show_root_control(_: object) -> None:
            robot.show_root_control = root_control_cb.value
            if robot.root_control_handle is not None:
                robot.root_control_handle.visible = robot.show_root_control

        show_meshes_cb.visible = load_meshes

        robot.control_folder_handle = server.gui.add_folder("Joint Control")
        with robot.control_folder_handle:
            sliders, names, initial_cfg, limits = create_robot_control_sliders(
                server, viser_urdf, robot
            )
            robot.slider_handles = sliders
            robot.joint_names = names
            robot.initial_config = initial_cfg
            robot.joint_limits = limits

            if sliders:
                viser_urdf.update_cfg(np.zeros(len(sliders)))

            randomize_button = server.gui.add_button("Randomize")
            robot.randomize_button = randomize_button

            @randomize_button.on_click
            def _on_randomize(_: object) -> None:
                if robot.slider_handles is None or robot.joint_limits is None:
                    return
                randomized = np.array(
                    [
                        np.random.uniform(lower, upper)
                        for lower, upper in robot.joint_limits
                    ],
                    dtype=float,
                )
                _apply_joint_configuration(robot, randomized, update_sliders=True)

            reset_button = server.gui.add_button("Reset")
            robot.reset_button = reset_button

            @reset_button.on_click
            def _on_reset(_: object) -> None:
                if robot.slider_handles is None or robot.initial_config is None:
                    return
                _apply_joint_configuration(
                    robot,
                    np.array(robot.initial_config, dtype=float),
                    update_sliders=True,
                )

        robot.cartesian_folder_handle = server.gui.add_folder("Cartesian Control")
        with robot.cartesian_folder_handle:
            cartesian_mode_checkbox = server.gui.add_checkbox("Enable", initial_value=False)
            robot.cartesian_mode_checkbox = cartesian_mode_checkbox
            setup_cartesian_controls(
                server, robot, source_path, state.tmp_dir, status_text, cartesian_mode_checkbox
            )

        robot.transform_folder_handle = server.gui.add_folder("Get Transform")
        with robot.transform_folder_handle:
            frame_options = list(urdf.link_map.keys())
            initial_frame = frame_options[0]
            initial_to_frame = initial_frame
            if robot.cartesian_frame_dropdown is not None:
                target_frame = robot.cartesian_frame_dropdown.value
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

            robot.transform_from_dropdown = from_dropdown
            robot.transform_to_dropdown = to_dropdown
            robot.transform_translation_text = translation_text
            robot.transform_rotation_text = rotation_text

            @from_dropdown.on_update
            def _on_transform_frame_change(_event: object) -> None:
                update_transform_display(robot)

            @to_dropdown.on_update
            def _on_transform_target_change(_event: object) -> None:
                update_transform_display(robot)

            @translation_text.on_update
            def _on_translation_text_edit(_event: object) -> None:
                if not robot.suppress_transform_text_callbacks:
                    update_transform_display(robot)

            @rotation_text.on_update
            def _on_rotation_text_edit(_event: object) -> None:
                if not robot.suppress_transform_text_callbacks:
                    update_transform_display(robot)

    robot.root_control_handle = server.scene.add_transform_controls(
        f"/gizmo_{robot_name.lower().replace(' ', '_')}",
        scale=0.25,
        line_width=3.0,
        visible=robot.show_root_control,
    )

    # Initialize the control handle at the robot's root position
    if robot.root_frame_handle is not None:
        robot.root_control_handle.position = robot.root_frame_handle.position
        robot.root_control_handle.wxyz = robot.root_frame_handle.wxyz

    @robot.root_control_handle.on_update
    def _on_root_control_update(_: object) -> None:
        if robot.root_control_handle is None or robot.root_frame_handle is None:
            return
        robot.root_frame_handle.position = robot.root_control_handle.position
        robot.root_frame_handle.wxyz = robot.root_control_handle.wxyz

    state.robots[robot_name] = robot
    update_transform_display(robot)

    if state.remove_robots_folder is not None:
        with state.remove_robots_folder:
            robot.remove_button_handle = server.gui.add_button(
                f"\u00a0\u00a0Remove {robot_name}", color="red", icon=viser.Icon.TRASH
            )
            @robot.remove_button_handle.on_click
            def _on_remove(_: object, _name: str = robot_name) -> None:
                remove_robot(server, state, _name)

    status_text.value = f"Loaded {robot_name}."
    try:
        server.flush()
    except Exception:
        pass


def setup_global_gui(
    server: viser.ViserServer,
    state: ViewerState,
) -> tuple[Any, GuiUploadButtonHandle, GuiDropdownHandle[str] | None, GuiButtonHandle | None]:
    tabs = server.gui.add_tab_group()
    state.tab_group_handle = tabs

    with tabs.add_tab("Controls", icon=viser.Icon.SETTINGS):
        with server.gui.add_folder("Viewer"):
            status_text = server.gui.add_text("Status", "Open a URDF file to begin.")
            reset_view_button = server.gui.add_button("\u00a0\u00a0Reset View", icon=viser.Icon.HOME_MOVE)
            save_canvas_button = server.gui.add_button("\u00a0\u00a0Save Canvas", icon=viser.Icon.PHOTO)
        ground_plane_folder = server.gui.add_folder("Ground Plane")
        with ground_plane_folder:
            ground_plane_cb = server.gui.add_checkbox(
                "Show Grid", initial_value=state.show_ground_plane
            )
            world_frame_cb = server.gui.add_checkbox(
                "Show Frame", initial_value=state.show_world_frame
            )
            width_input = server.gui.add_number(
                "Grid X",
                initial_value=state.ground_plane_size[0],
                min=1,
                max=100,
                step=1,
            )
            height_input = server.gui.add_number(
                "Grid Y",
                initial_value=state.ground_plane_size[1],
                min=1,
                max=100,
                step=1,
            )

        state.remove_robots_folder = server.gui.add_folder("Robots")
        with state.remove_robots_folder:
            upload_button = server.gui.add_upload_button(
                "\u00a0\u00a0Add from File",
                icon=viser.Icon.UPLOAD,
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
                    "\u00a0\u00a0Add from Robot Descriptions", icon=viser.Icon.UPLOAD
                )
            else:
                server.gui.add_markdown(
                    "Install `robot_descriptions` to browse robots from that catalog."
                )

    @reset_view_button.on_click
    def _on_reset_view(event: GuiEvent[GuiButtonHandle]) -> None:
        if event.client is None:
            status_text.value = "No active client to reset view."
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
        status_text.value = "View reset."

    @save_canvas_button.on_click
    def _on_save_canvas(event: GuiEvent[GuiButtonHandle]) -> None:
        client = event.client
        if client is None:
            status_text.value = "No active client to save canvas."
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
            status_text.value = f"Saved canvas as {filename}."
        except Exception as exc:
            status_text.value = f"Failed to save canvas: {exc!r}"

    @ground_plane_cb.on_update
    def _on_ground_plane(_: object) -> None:
        state.show_ground_plane = ground_plane_cb.value
        set_ground_plane_visible(server, state, state.show_ground_plane)

    def _recreate_ground_plane() -> None:
        if state.ground_plane_handle is not None:
            try:
                state.ground_plane_handle.remove()
            except Exception:
                pass
            state.ground_plane_handle = None
        set_ground_plane_visible(server, state, state.show_ground_plane)

    @width_input.on_update
    def _on_width_change(_: object) -> None:
        state.ground_plane_size = (width_input.value, state.ground_plane_size[1])
        _recreate_ground_plane()

    @height_input.on_update
    def _on_height_change(_: object) -> None:
        state.ground_plane_size = (state.ground_plane_size[0], height_input.value)
        _recreate_ground_plane()

    @world_frame_cb.on_update
    def _on_world_frame(_: object) -> None:
        state.show_world_frame = world_frame_cb.value
        set_world_frame_visible(server, state, state.show_world_frame)

    return (
        status_text,
        upload_button,
        description_dropdown,
        load_description_button,
    )


def _reload_connected_pages(server: viser.ViserServer) -> None:
    for client in server.get_clients().values():
        try:
            client._websock_connection.queue_message(
                _messages.RunJavascriptMessage(source="window.location.reload();")
            )
            client.flush()
        except Exception:
            pass


def _mount_loaded_robot(
    server: viser.ViserServer,
    state: ViewerState,
    status_text: Any,
    load_meshes: bool,
) -> Callable[[yourdfpy.URDF, str], None]:
    def _mount(urdf: yourdfpy.URDF, source_path: str) -> None:
        load_robot_into_viewer(server, state, urdf, source_path, status_text, load_meshes)

    return _mount


def load_startup_target(
    server: viser.ViserServer,
    state: ViewerState,
    path: str,
    rd: bool,
    status_text: Any,
    load_meshes: bool,
) -> None:
    source = RobotDescriptionModelSource(path) if rd else PathModelSource(path)
    execute_model_load(
        state=state,
        source=source,
        status_text=status_text,
        load_meshes=load_meshes,
        mount_loaded_robot=_mount_loaded_robot(
            server=server,
            state=state,
            status_text=status_text,
            load_meshes=load_meshes,
        ),
        reload_connected_pages=lambda: _reload_connected_pages(server),
    )


def register_file_event_handlers(
    server: viser.ViserServer,
    state: ViewerState,
    status_text: Any,
    upload_button: GuiUploadButtonHandle,
    description_dropdown: GuiDropdownHandle[str] | None,
    load_description_button: GuiButtonHandle | None,
    load_meshes: bool,
) -> None:
    @upload_button.on_upload
    def _on_upload(event: GuiEvent[GuiUploadButtonHandle]) -> None:
        uploaded: UploadedFile = event.target.value
        if uploaded is None:
            return

        execute_model_load(
            state=state,
            source=UploadedModelSource(uploaded),
            status_text=status_text,
            load_meshes=load_meshes,
            mount_loaded_robot=_mount_loaded_robot(
                server=server,
                state=state,
                status_text=status_text,
                load_meshes=load_meshes,
            ),
            reload_connected_pages=lambda: _reload_connected_pages(server),
        )

    if description_dropdown is None or load_description_button is None:
        return

    @load_description_button.on_click
    def _on_load_robot_description(_event: GuiEvent[GuiButtonHandle]) -> None:
        execute_model_load(
            state=state,
            source=RobotDescriptionModelSource(description_dropdown.value),
            status_text=status_text,
            load_meshes=load_meshes,
            mount_loaded_robot=_mount_loaded_robot(
                server=server,
                state=state,
                status_text=status_text,
                load_meshes=load_meshes,
            ),
            reload_connected_pages=lambda: _reload_connected_pages(server),
        )
