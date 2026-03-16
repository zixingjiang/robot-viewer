from __future__ import annotations

import os
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional
from typing import Any

import numpy as np
import pinocchio as pin  # type: ignore[import]
import qpsolvers
import pink
import tyro
import viser
import yourdfpy  # type: ignore[import]
from pink import solve_ik
from pink.tasks import DampingTask, FrameTask, PostureTask
from viser.extras import ViserUrdf
from viser._gui_handles import (
    GuiEvent,
    GuiButtonHandle,
    GuiCheckboxHandle,
    GuiDropdownHandle,
    GuiFolderHandle,
    GuiUploadButtonHandle,
    UploadedFile,
)
from viser._scene_handles import TransformControlsHandle


@dataclass
class _ViewerState:
    """Mutable state used while the server is running."""

    current_urdf: Optional[ViserUrdf] = None
    current_root_name: Optional[str] = None
    control_folder_handle: Optional[GuiFolderHandle] = None
    slider_handles: Optional[list[viser.GuiInputHandle[float]]] = None
    joint_names: Optional[list[str]] = None
    initial_config: Optional[list[float]] = None
    joint_limits: Optional[list[tuple[float, float]]] = None
    randomize_button: Optional[GuiButtonHandle] = None
    reset_button: Optional[GuiButtonHandle] = None
    cartesian_mode_checkbox: Optional[GuiCheckboxHandle] = None
    cartesian_frame_dropdown: Optional[GuiDropdownHandle[str]] = None
    cartesian_target_handle: Optional[TransformControlsHandle] = None
    ik_configuration: Optional[pink.Configuration] = None
    ik_tasks: Optional[list[Any]] = None
    ik_frame_task: Optional[FrameTask] = None
    ik_posture_task: Optional[PostureTask] = None
    ik_damping_task: Optional[DampingTask] = None
    ik_joint_name_to_q_index: dict[str, int] = field(default_factory=dict)
    ik_solver: Optional[str] = None
    ik_frame_position_cost: float = 1.0
    ik_frame_orientation_cost: float = 1.0
    ik_posture_cost: float = 1e-3
    ik_damping_cost: float = 1e-3
    ik_enabled: bool = False
    ik_running: bool = True
    ik_dt: float = 1.0 / 120.0
    suppress_slider_callbacks: bool = False
    ik_lock: threading.Lock = field(default_factory=threading.Lock)
    tmp_dir: str = ""


def _create_robot_control_sliders(
    server: viser.ViserServer, viser_urdf: ViserUrdf, state: _ViewerState
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

        # Aim for a zero joint configuration on load.
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


def _rotation_matrix_to_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    quat_xyzw = pin.Quaternion(rotation).coeffs()  # type: ignore[attr-defined]
    return (
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    )


def _wxyz_to_rotation_matrix(wxyz: tuple[float, float, float, float]) -> np.ndarray:
    quat = pin.Quaternion(  # type: ignore[attr-defined]
        float(wxyz[0]), float(wxyz[1]), float(wxyz[2]), float(wxyz[3])
    )
    return quat.matrix()


def _build_joint_name_to_q_index(model: Any) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for joint_id, joint_name in enumerate(model.names):
        if joint_id == 0:
            continue
        joint = model.joints[joint_id]
        if joint.nq != 1:
            continue
        mapping[joint_name] = int(joint.idx_q)
    return mapping


def _pick_qp_solver() -> str:
    if "daqp" in qpsolvers.available_solvers:
        return "daqp"
    if not qpsolvers.available_solvers:
        raise RuntimeError("No QP solver is available for Pink IK")
    return qpsolvers.available_solvers[0]


def _safe_write_file(uploaded_file: UploadedFile, tmp_dir: str) -> str:
    """Write the uploaded file to a temporary directory and return the path."""

    os.makedirs(tmp_dir, exist_ok=True)
    out_path = os.path.join(tmp_dir, os.path.basename(uploaded_file.name))
    with open(out_path, "wb") as f:
        f.write(uploaded_file.content)
    return out_path


def _sanitize_urdf_for_pinocchio(path: str, tmp_dir: str) -> tuple[str, int]:
    """Return a URDF path safe for Pinocchio, removing duplicate global materials."""

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception:
        return path, 0

    if root.tag != "robot":
        return path, 0

    seen_materials: set[str] = set()
    removed_count = 0

    for child in list(root):
        if child.tag != "material":
            continue

        material_name = child.attrib.get("name")
        if not material_name:
            continue

        if material_name in seen_materials:
            root.remove(child)
            removed_count += 1
            continue

        seen_materials.add(material_name)

    if removed_count == 0:
        return path, 0

    out_name = f"pinocchio_sanitized_{os.path.basename(path)}"
    out_path = os.path.join(tmp_dir, out_name)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path, removed_count


def main(
    host: str = "0.0.0.0",
    port: int = 8080,
    label: Optional[str] = "Robot Viewer",
    load_meshes: bool = True,
    load_collision_meshes: bool = False,
    show_grid: bool = True,
) -> None:
    """Start a robot viewer server.

    The viewer will be served at http://{host}:{port}. Use the "Upload URDF" button
    in the GUI to select a local URDF file, and it will be visualized in the 3D view.

    The visualization is powered by the `viser` library.
    """

    state = _ViewerState(tmp_dir=tempfile.mkdtemp(prefix="robot_viewer_"))

    server = viser.ViserServer(host=host, port=port, label=label)

    status_text = server.gui.add_text("Status", "Open a URDF file to begin.")

    upload_button = server.gui.add_upload_button(
        "Open URDF",
        mime_type="*/*",
        hint="Select a URDF file (.urdf, .xml).",
    )

    def _clear_previous_robot() -> None:
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

        # Clear any cached UI state so controls aren't referenced after removal.
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

    def _load_urdf_file(path: str) -> None:
        _clear_previous_robot()

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

        # Build robot controls.
        state.control_folder_handle = server.gui.add_folder("Robot joint controls")
        with state.control_folder_handle:
            slider_handles, joint_names, initial_config, joint_limits = (
                _create_robot_control_sliders(server, viser_urdf, state)
            )
            state.slider_handles = slider_handles
            state.joint_names = joint_names
            state.initial_config = initial_config
            state.joint_limits = joint_limits

            # Ensure the robot starts in the zero joint configuration by default.
            if slider_handles:
                viser_urdf.update_cfg(np.zeros(len(slider_handles)))

            randomize_button = server.gui.add_button("Randomize joints")
            state.randomize_button = randomize_button

            def _on_randomize(_):
                if state.slider_handles is None or state.joint_limits is None:
                    return

                for s, (lower, upper) in zip(state.slider_handles, state.joint_limits):
                    s.value = float(np.random.uniform(lower, upper))

            randomize_button.on_click(_on_randomize)

            reset_button = server.gui.add_button("Reset joints")
            state.reset_button = reset_button

            def _on_reset(_):
                if state.slider_handles is None or state.initial_config is None:
                    return

                for s, init in zip(state.slider_handles, state.initial_config):
                    s.value = init

            reset_button.on_click(_on_reset)

            show_meshes_cb = server.gui.add_checkbox(
                "Show meshes", viser_urdf.show_visual
            )
            show_collision_cb = server.gui.add_checkbox(
                "Show collision meshes", viser_urdf.show_collision
            )

            @show_meshes_cb.on_update
            def _show_meshes(_):
                viser_urdf.show_visual = show_meshes_cb.value

            @show_collision_cb.on_update
            def _show_collision(_):
                viser_urdf.show_collision = show_collision_cb.value

            show_meshes_cb.visible = load_meshes
            show_collision_cb.visible = load_collision_meshes

            cartesian_mode_checkbox = server.gui.add_checkbox(
                "Cartesian control mode", initial_value=False
            )
            state.cartesian_mode_checkbox = cartesian_mode_checkbox

            try:
                pin_urdf_path, removed_materials = _sanitize_urdf_for_pinocchio(
                    path, state.tmp_dir
                )
                pin_model = pin.buildModelFromUrdf(  # type: ignore[attr-defined]
                    pin_urdf_path
                )
                pin_data = pin_model.createData()
                q0 = pin.neutral(pin_model)  # type: ignore[attr-defined]
                configuration = pink.Configuration(pin_model, pin_data, q0)
                configuration.update(q0)

                if removed_materials > 0:
                    status_text.value = (
                        f"Loaded with Cartesian IK (ignored {removed_materials} "
                        "duplicate global material definitions for Pinocchio)."
                    )

                state.ik_configuration = configuration
                state.ik_joint_name_to_q_index = _build_joint_name_to_q_index(pin_model)
                state.ik_solver = _pick_qp_solver()

                frame_options = [
                    frame.name
                    for frame in pin_model.frames
                    if frame.name != "universe" and frame.name
                ]
                if not frame_options:
                    raise RuntimeError("No valid frames found for Cartesian target")

                cartesian_frame_dropdown = server.gui.add_dropdown(
                    "Cartesian target frame",
                    options=frame_options,
                    initial_value=frame_options[-1],
                )
                state.cartesian_frame_dropdown = cartesian_frame_dropdown

                cartesian_target_handle = server.scene.add_transform_controls(
                    "/cartesian_target",
                    scale=0.25,
                    line_width=3.0,
                    visible=False,
                )
                state.cartesian_target_handle = cartesian_target_handle

                frame_task_position_cost_slider = server.gui.add_slider(
                    "Frame Task Position Cost",
                    min=0.0,
                    max=10.0,
                    initial_value=state.ik_frame_position_cost,
                    step=0.1,
                )
                frame_task_orientation_cost_slider = server.gui.add_slider(
                    "Frame Task Orientation Cost",
                    min=0.0,
                    max=10.0,
                    initial_value=state.ik_frame_orientation_cost,
                    step=0.1,
                )
                damping_cost_slider = server.gui.add_slider(
                    "Damping Cost",
                    min=0.0,
                    max=1.0,
                    initial_value=state.ik_damping_cost,
                    step=0.001,
                )
                posture_cost_slider = server.gui.add_slider(
                    "Posture Cost",
                    min=0.0,
                    max=1.0,
                    initial_value=state.ik_posture_cost,
                    step=0.001,
                )

                def _set_joint_controls_enabled(enabled: bool) -> None:
                    if state.slider_handles is not None:
                        for slider in state.slider_handles:
                            slider.disabled = not enabled
                    if state.randomize_button is not None:
                        state.randomize_button.disabled = not enabled
                    if state.reset_button is not None:
                        state.reset_button.disabled = not enabled

                def _sync_configuration_from_sliders() -> None:
                    if (
                        state.ik_configuration is None
                        or state.joint_names is None
                        or state.slider_handles is None
                    ):
                        return

                    with state.ik_lock:
                        q = np.array(state.ik_configuration.q, copy=True)
                        for joint_name, slider in zip(
                            state.joint_names, state.slider_handles
                        ):
                            q_index = state.ik_joint_name_to_q_index.get(joint_name)
                            if q_index is not None:
                                q[q_index] = slider.value
                        state.ik_configuration.update(q)

                def _set_ik_tasks_from_current_configuration(frame_name: str) -> None:
                    if state.ik_configuration is None:
                        return

                    with state.ik_lock:
                        frame_task = FrameTask(
                            frame_name,
                            position_cost=state.ik_frame_position_cost,
                            orientation_cost=state.ik_frame_orientation_cost,
                            lm_damping=1.0,
                        )
                        posture_task = PostureTask(cost=state.ik_posture_cost)
                        damping_task = DampingTask(cost=state.ik_damping_cost)

                        frame_task.set_target_from_configuration(state.ik_configuration)
                        posture_task.set_target_from_configuration(
                            state.ik_configuration
                        )

                        state.ik_frame_task = frame_task
                        state.ik_posture_task = posture_task
                        state.ik_damping_task = damping_task
                        state.ik_tasks = [frame_task, posture_task, damping_task]

                        frame_pose = (
                            state.ik_configuration.get_transform_frame_to_world(
                                frame_name
                            ).np
                        )

                    if state.cartesian_target_handle is not None:
                        state.cartesian_target_handle.position = (
                            float(frame_pose[0, 3]),
                            float(frame_pose[1, 3]),
                            float(frame_pose[2, 3]),
                        )
                        state.cartesian_target_handle.wxyz = _rotation_matrix_to_wxyz(
                            frame_pose[:3, :3]
                        )

                @frame_task_position_cost_slider.on_update
                def _on_frame_task_position_cost_update(_event: object) -> None:
                    state.ik_frame_position_cost = frame_task_position_cost_slider.value
                    if state.ik_frame_task is not None:
                        state.ik_frame_task.set_position_cost(
                            state.ik_frame_position_cost
                        )

                @frame_task_orientation_cost_slider.on_update
                def _on_frame_task_orientation_cost_update(_event: object) -> None:
                    state.ik_frame_orientation_cost = (
                        frame_task_orientation_cost_slider.value
                    )
                    if state.ik_frame_task is not None:
                        state.ik_frame_task.set_orientation_cost(
                            state.ik_frame_orientation_cost
                        )

                @damping_cost_slider.on_update
                def _on_damping_cost_update(_event: object) -> None:
                    state.ik_damping_cost = damping_cost_slider.value
                    if state.ik_damping_task is not None:
                        state.ik_damping_task.cost = state.ik_damping_cost

                @posture_cost_slider.on_update
                def _on_posture_cost_update(_event: object) -> None:
                    state.ik_posture_cost = posture_cost_slider.value
                    if state.ik_posture_task is not None:
                        state.ik_posture_task.cost = state.ik_posture_cost

                @cartesian_target_handle.on_update
                def _on_target_update(event: viser.TransformControlsEvent) -> None:
                    if state.ik_frame_task is None:
                        return
                    target = state.ik_frame_task.transform_target_to_world
                    if target is None:
                        return
                    target.translation = np.array(event.target.position)
                    target.rotation = _wxyz_to_rotation_matrix(
                        (
                            float(event.target.wxyz[0]),
                            float(event.target.wxyz[1]),
                            float(event.target.wxyz[2]),
                            float(event.target.wxyz[3]),
                        )
                    )

                @cartesian_frame_dropdown.on_update
                def _on_frame_change(_event: object) -> None:
                    if state.cartesian_frame_dropdown is None:
                        return
                    _sync_configuration_from_sliders()
                    _set_ik_tasks_from_current_configuration(
                        state.cartesian_frame_dropdown.value
                    )

                @cartesian_mode_checkbox.on_update
                def _on_cartesian_mode_change(_event: object) -> None:
                    enabled = cartesian_mode_checkbox.value

                    if not enabled:
                        state.ik_enabled = False
                        _set_joint_controls_enabled(True)
                        if state.cartesian_target_handle is not None:
                            state.cartesian_target_handle.visible = False
                        return

                    _set_joint_controls_enabled(False)
                    if state.cartesian_target_handle is not None:
                        state.cartesian_target_handle.visible = True

                    if state.cartesian_frame_dropdown is not None:
                        _sync_configuration_from_sliders()
                        # Reinitialize the target every time Cartesian mode is enabled.
                        _set_ik_tasks_from_current_configuration(
                            state.cartesian_frame_dropdown.value
                        )

                    state.ik_enabled = True

            except Exception as ik_exc:
                cartesian_mode_checkbox.disabled = True
                server.gui.add_text("Cartesian IK", f"Unavailable: {ik_exc!r}")

        if show_grid:
            # Remove any previous grid to avoid stacking multiple grids on top of one another.
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

    @upload_button.on_upload
    def _on_upload(event: GuiEvent[GuiUploadButtonHandle]) -> None:
        uploaded: UploadedFile = event.target.value
        if uploaded is None:
            return

        status_text.value = f"Loading {uploaded.name}..."

        try:
            path = _safe_write_file(uploaded, state.tmp_dir)
            _load_urdf_file(path)
            status_text.value = f"Loaded {uploaded.name}."
        except Exception as e:
            status_text.value = f"Failed to load {uploaded.name}: {e!r}"

    def _ik_worker() -> None:
        limit_tol = 1e-9

        while state.ik_running:
            time.sleep(state.ik_dt)

            if not state.ik_enabled:
                continue
            if state.ik_configuration is None or state.ik_tasks is None:
                continue
            if (
                state.slider_handles is None
                or state.joint_names is None
                or state.joint_limits is None
            ):
                continue
            if state.ik_solver is None:
                continue

            try:
                with state.ik_lock:
                    velocity = solve_ik(
                        state.ik_configuration,
                        state.ik_tasks,
                        state.ik_dt,
                        solver=state.ik_solver,
                    )
                    state.ik_configuration.integrate_inplace(velocity, state.ik_dt)
                    q = np.array(state.ik_configuration.q, copy=True)
            except Exception as ik_error:
                # Recover from occasional tiny limit violations by resyncing
                # the IK configuration from current slider values.
                if "NotWithinConfigurationLimits" in repr(ik_error):
                    with state.ik_lock:
                        q_recovered = np.array(state.ik_configuration.q, copy=True)
                        for joint_name, slider in zip(
                            state.joint_names, state.slider_handles
                        ):
                            q_index = state.ik_joint_name_to_q_index.get(joint_name)
                            if q_index is not None:
                                q_recovered[q_index] = float(slider.value)
                        state.ik_configuration.update(q_recovered)
                    status_text.value = (
                        "Cartesian IK hit a joint limit; clamped to limits and "
                        "continuing."
                    )
                    continue

                state.ik_enabled = False
                if state.cartesian_mode_checkbox is not None:
                    state.cartesian_mode_checkbox.value = False
                status_text.value = f"Cartesian IK error: {ik_error!r}"
                continue

            cfg: list[float] = []
            clamped_any = False
            for slider, joint_name, (lower, upper) in zip(
                state.slider_handles, state.joint_names, state.joint_limits
            ):
                q_index = state.ik_joint_name_to_q_index.get(joint_name)
                if q_index is None:
                    cfg.append(float(slider.value))
                    continue

                value = float(q[q_index])
                clamped = float(np.clip(value, lower + limit_tol, upper - limit_tol))
                if abs(clamped - value) > 0.0:
                    clamped_any = True
                    q[q_index] = clamped
                cfg.append(clamped)

            if clamped_any:
                with state.ik_lock:
                    state.ik_configuration.update(q)

            state.suppress_slider_callbacks = True
            try:
                for slider, value in zip(state.slider_handles, cfg):
                    slider.value = value
            finally:
                state.suppress_slider_callbacks = False

            if state.current_urdf is not None:
                state.current_urdf.update_cfg(np.array(cfg, dtype=float))

    ik_thread = threading.Thread(target=_ik_worker, daemon=True)
    ik_thread.start()

    try:
        server.sleep_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.ik_running = False
        ik_thread.join(timeout=1.0)
        server.stop()


if __name__ == "__main__":
    tyro.cli(main)
