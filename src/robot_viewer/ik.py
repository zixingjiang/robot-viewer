from __future__ import annotations

import time
from typing import Any

import numpy as np
import pinocchio as pin  # type: ignore[import]
import pink
import qpsolvers
import viser
from pink import solve_ik
from pink.tasks import DampingTask, FrameTask, PostureTask

from .state import ViewerState
from .utils import (
    rotation_matrix_to_wxyz,
    sanitize_urdf_for_pinocchio,
    wxyz_to_rotation_matrix,
)


def build_joint_name_to_q_index(model: Any) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for joint_id, joint_name in enumerate(model.names):
        if joint_id == 0:
            continue
        joint = model.joints[joint_id]
        if joint.nq != 1:
            continue
        mapping[joint_name] = int(joint.idx_q)
    return mapping


def pick_qp_solver() -> str:
    if "daqp" in qpsolvers.available_solvers:
        return "daqp"
    if not qpsolvers.available_solvers:
        raise RuntimeError("No QP solver is available for Pink IK")
    return qpsolvers.available_solvers[0]


def _get_frame_pose_from_viewer(
    state: ViewerState, frame_name: str
) -> np.ndarray | None:
    """Get frame pose from yourdfpy visualization state when available."""
    if state.current_urdf is None:
        return None

    try:
        return np.array(state.current_urdf._urdf.get_transform(frame_name), copy=True)
    except Exception:
        return None


def setup_cartesian_controls(
    server: viser.ViserServer,
    state: ViewerState,
    path: str,
    status_text: Any,
    cartesian_mode_checkbox: Any,
) -> None:
    try:
        pin_urdf_path, removed_materials = sanitize_urdf_for_pinocchio(
            path, state.tmp_dir
        )
        pin_model = pin.buildModelFromUrdf(pin_urdf_path)  # type: ignore[attr-defined]
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
        state.ik_joint_name_to_q_index = build_joint_name_to_q_index(pin_model)
        state.ik_solver = pick_qp_solver()

        pin_frame_names = {
            frame.name
            for frame in pin_model.frames
            if frame.name != "universe" and frame.name
        }
        frame_options = []
        if state.current_urdf is not None:
            for link_name in state.current_urdf._urdf.link_map.keys():
                if link_name in pin_frame_names:
                    frame_options.append(link_name)

        if not frame_options:
            frame_options = sorted(pin_frame_names)

        if not frame_options:
            raise RuntimeError("No valid frames found for Cartesian target")

        cartesian_frame_dropdown = server.gui.add_dropdown(
            "Target frame",
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

        advanced_folder = server.gui.add_folder("Advanced", expand_by_default=False)
        with advanced_folder:
            frame_task_position_cost_slider = server.gui.add_slider(
                "Position cost",
                min=0.0,
                max=10.0,
                initial_value=state.ik_frame_position_cost,
                step=0.1,
            )
            frame_task_orientation_cost_slider = server.gui.add_slider(
                "Orientation cost",
                min=0.0,
                max=10.0,
                initial_value=state.ik_frame_orientation_cost,
                step=0.1,
            )
            damping_cost_slider = server.gui.add_slider(
                "Damping cost",
                min=0.0,
                max=1.0,
                initial_value=state.ik_damping_cost,
                step=0.001,
            )
            posture_cost_slider = server.gui.add_slider(
                "Posture cost",
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
                for joint_name, slider in zip(state.joint_names, state.slider_handles):
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
                posture_task.set_target_from_configuration(state.ik_configuration)

                state.ik_frame_task = frame_task
                state.ik_posture_task = posture_task
                state.ik_damping_task = damping_task
                state.ik_tasks = [frame_task, posture_task, damping_task]

                frame_pose = state.ik_configuration.get_transform_frame_to_world(
                    frame_name
                ).np

                viewer_frame_pose = _get_frame_pose_from_viewer(state, frame_name)
                if viewer_frame_pose is not None:
                    frame_pose = viewer_frame_pose

                target = frame_task.transform_target_to_world
                if target is not None:
                    target.translation = frame_pose[:3, 3].copy()
                    target.rotation = frame_pose[:3, :3].copy()

            if state.cartesian_target_handle is not None:
                state.cartesian_target_handle.position = (
                    float(frame_pose[0, 3]),
                    float(frame_pose[1, 3]),
                    float(frame_pose[2, 3]),
                )
                state.cartesian_target_handle.wxyz = rotation_matrix_to_wxyz(
                    frame_pose[:3, :3]
                )

        @frame_task_position_cost_slider.on_update
        def _on_frame_task_position_cost_update(_event: object) -> None:
            state.ik_frame_position_cost = frame_task_position_cost_slider.value
            if state.ik_frame_task is not None:
                state.ik_frame_task.set_position_cost(state.ik_frame_position_cost)

        @frame_task_orientation_cost_slider.on_update
        def _on_frame_task_orientation_cost_update(_event: object) -> None:
            state.ik_frame_orientation_cost = frame_task_orientation_cost_slider.value
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
            target.rotation = wxyz_to_rotation_matrix(
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
                _set_ik_tasks_from_current_configuration(
                    state.cartesian_frame_dropdown.value
                )

            state.ik_enabled = True

    except Exception as ik_exc:
        cartesian_mode_checkbox.disabled = True
        server.gui.add_text("Cartesian IK", f"Unavailable: {ik_exc!r}")


def ik_worker_loop(state: ViewerState, status_text: Any) -> None:
    from .scene import update_link_frame_visuals

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
            # Recover from occasional tiny limit violations by resyncing the IK
            # configuration from current slider values.
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
                    "Cartesian IK hit a joint limit; clamped to limits and continuing."
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
            update_link_frame_visuals(state)
