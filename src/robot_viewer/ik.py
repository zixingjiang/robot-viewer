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

from .state import RobotInstance, ViewerState
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
    robot: RobotInstance, frame_name: str
) -> np.ndarray | None:
    try:
        return np.array(robot.urdf._urdf.get_transform(frame_name), copy=True)
    except Exception:
        return None


def setup_cartesian_controls(
    server: viser.ViserServer,
    robot: RobotInstance,
    path: str,
    tmp_dir: str,
    status_text: Any,
    cartesian_mode_checkbox: Any,
) -> None:
    try:
        pin_urdf_path, removed_materials = sanitize_urdf_for_pinocchio(path, tmp_dir)
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

        robot.ik_configuration = configuration
        robot.ik_joint_name_to_q_index = build_joint_name_to_q_index(pin_model)
        robot.ik_solver = pick_qp_solver()

        pin_frame_names = {
            frame.name
            for frame in pin_model.frames
            if frame.name != "universe" and frame.name
        }
        frame_options = []
        for link_name in robot.urdf._urdf.link_map.keys():
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
        robot.cartesian_frame_dropdown = cartesian_frame_dropdown

        cartesian_target_handle = server.scene.add_transform_controls(
            f"{robot.root_name}/cartesian_target",
            scale=0.25,
            line_width=3.0,
            visible=False,
        )
        robot.cartesian_target_handle = cartesian_target_handle

        advanced_folder = server.gui.add_folder("Advanced", expand_by_default=False)
        with advanced_folder:
            frame_task_position_cost_slider = server.gui.add_slider(
                "Position cost",
                min=0.0,
                max=10.0,
                initial_value=robot.ik_frame_position_cost,
                step=0.1,
            )
            frame_task_orientation_cost_slider = server.gui.add_slider(
                "Orientation cost",
                min=0.0,
                max=10.0,
                initial_value=robot.ik_frame_orientation_cost,
                step=0.1,
            )
            damping_cost_slider = server.gui.add_slider(
                "Damping cost",
                min=0.0,
                max=1.0,
                initial_value=robot.ik_damping_cost,
                step=0.001,
            )
            posture_cost_slider = server.gui.add_slider(
                "Posture cost",
                min=0.0,
                max=1.0,
                initial_value=robot.ik_posture_cost,
                step=0.001,
            )

        def _set_joint_controls_enabled(enabled: bool) -> None:
            if robot.slider_handles is not None:
                for slider in robot.slider_handles:
                    slider.disabled = not enabled
            if robot.randomize_button is not None:
                robot.randomize_button.disabled = not enabled
            if robot.reset_button is not None:
                robot.reset_button.disabled = not enabled

        def _sync_configuration_from_sliders() -> None:
            if (
                robot.ik_configuration is None
                or robot.joint_names is None
                or robot.slider_handles is None
            ):
                return

            with robot.ik_lock:
                q = np.array(robot.ik_configuration.q, copy=True)
                for joint_name, slider in zip(robot.joint_names, robot.slider_handles):
                    q_index = robot.ik_joint_name_to_q_index.get(joint_name)
                    if q_index is not None:
                        q[q_index] = slider.value
                robot.ik_configuration.update(q)

        def _set_ik_tasks_from_current_configuration(frame_name: str) -> None:
            if robot.ik_configuration is None:
                return

            with robot.ik_lock:
                frame_task = FrameTask(
                    frame_name,
                    position_cost=robot.ik_frame_position_cost,
                    orientation_cost=robot.ik_frame_orientation_cost,
                    lm_damping=1.0,
                )
                posture_task = PostureTask(cost=robot.ik_posture_cost)
                damping_task = DampingTask(cost=robot.ik_damping_cost)

                frame_task.set_target_from_configuration(robot.ik_configuration)
                posture_task.set_target_from_configuration(robot.ik_configuration)

                robot.ik_frame_task = frame_task
                robot.ik_posture_task = posture_task
                robot.ik_damping_task = damping_task
                robot.ik_tasks = [frame_task, posture_task, damping_task]

                frame_pose = robot.ik_configuration.get_transform_frame_to_world(
                    frame_name
                ).np

                viewer_frame_pose = _get_frame_pose_from_viewer(robot, frame_name)
                if viewer_frame_pose is not None:
                    frame_pose = viewer_frame_pose

                target = frame_task.transform_target_to_world
                if target is not None:
                    target.translation = frame_pose[:3, 3].copy()
                    target.rotation = frame_pose[:3, :3].copy()

            if robot.cartesian_target_handle is not None:
                robot.cartesian_target_handle.position = (
                    float(frame_pose[0, 3]),
                    float(frame_pose[1, 3]),
                    float(frame_pose[2, 3]),
                )
                robot.cartesian_target_handle.wxyz = rotation_matrix_to_wxyz(
                    frame_pose[:3, :3]
                )

        @frame_task_position_cost_slider.on_update
        def _on_frame_task_position_cost_update(_event: object) -> None:
            robot.ik_frame_position_cost = frame_task_position_cost_slider.value
            if robot.ik_frame_task is not None:
                robot.ik_frame_task.set_position_cost(robot.ik_frame_position_cost)

        @frame_task_orientation_cost_slider.on_update
        def _on_frame_task_orientation_cost_update(_event: object) -> None:
            robot.ik_frame_orientation_cost = frame_task_orientation_cost_slider.value
            if robot.ik_frame_task is not None:
                robot.ik_frame_task.set_orientation_cost(
                    robot.ik_frame_orientation_cost
                )

        @damping_cost_slider.on_update
        def _on_damping_cost_update(_event: object) -> None:
            robot.ik_damping_cost = damping_cost_slider.value
            if robot.ik_damping_task is not None:
                robot.ik_damping_task.cost = robot.ik_damping_cost

        @posture_cost_slider.on_update
        def _on_posture_cost_update(_event: object) -> None:
            robot.ik_posture_cost = posture_cost_slider.value
            if robot.ik_posture_task is not None:
                robot.ik_posture_task.cost = robot.ik_posture_cost

        @cartesian_target_handle.on_update
        def _on_target_update(event: viser.TransformControlsEvent) -> None:
            if robot.ik_frame_task is None:
                return
            target = robot.ik_frame_task.transform_target_to_world
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
            if robot.cartesian_frame_dropdown is None:
                return
            _sync_configuration_from_sliders()
            _set_ik_tasks_from_current_configuration(
                robot.cartesian_frame_dropdown.value
            )

        @cartesian_mode_checkbox.on_update
        def _on_cartesian_mode_change(_event: object) -> None:
            enabled = cartesian_mode_checkbox.value

            if not enabled:
                robot.ik_enabled = False
                _set_joint_controls_enabled(True)
                if robot.cartesian_target_handle is not None:
                    robot.cartesian_target_handle.visible = False
                return

            _set_joint_controls_enabled(False)
            if robot.cartesian_target_handle is not None:
                robot.cartesian_target_handle.visible = True

            if robot.cartesian_frame_dropdown is not None:
                _sync_configuration_from_sliders()
                _set_ik_tasks_from_current_configuration(
                    robot.cartesian_frame_dropdown.value
                )

            robot.ik_enabled = True

    except Exception as ik_exc:
        cartesian_mode_checkbox.disabled = True
        server.gui.add_text("Cartesian IK", f"Unavailable: {ik_exc!r}")


def ik_worker_loop(state: ViewerState, status_text: Any) -> None:
    from .scene import update_link_frame_visuals

    limit_tol = 1e-9

    while state.ik_running:
        time.sleep(state.ik_dt)

        for robot in list(state.robots.values()):
            if not robot.ik_enabled:
                continue
            if robot.ik_configuration is None or not robot.ik_tasks:
                continue
            if (
                robot.slider_handles is None
                or robot.joint_names is None
                or robot.joint_limits is None
            ):
                continue
            if robot.ik_solver is None:
                continue

            try:
                with robot.ik_lock:
                    velocity = solve_ik(
                        robot.ik_configuration,
                        robot.ik_tasks,
                        state.ik_dt,
                        solver=robot.ik_solver,
                    )
                    robot.ik_configuration.integrate_inplace(velocity, state.ik_dt)
                    q = np.array(robot.ik_configuration.q, copy=True)
            except Exception as ik_error:
                if "NotWithinConfigurationLimits" in repr(ik_error):
                    with robot.ik_lock:
                        q_recovered = np.array(robot.ik_configuration.q, copy=True)
                        for joint_name, slider in zip(
                            robot.joint_names, robot.slider_handles
                        ):
                            q_index = robot.ik_joint_name_to_q_index.get(joint_name)
                            if q_index is not None:
                                q_recovered[q_index] = float(slider.value)
                        robot.ik_configuration.update(q_recovered)
                    status_text.value = (
                        "Cartesian IK hit a joint limit; clamped to limits and continuing."
                    )
                    continue

                robot.ik_enabled = False
                if robot.cartesian_mode_checkbox is not None:
                    robot.cartesian_mode_checkbox.value = False
                status_text.value = f"Cartesian IK error: {ik_error!r}"
                continue

            cfg: list[float] = []
            clamped_any = False
            for slider, joint_name, (lower, upper) in zip(
                robot.slider_handles, robot.joint_names, robot.joint_limits
            ):
                q_index = robot.ik_joint_name_to_q_index.get(joint_name)
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
                with robot.ik_lock:
                    robot.ik_configuration.update(q)

            robot.suppress_slider_callbacks = True
            try:
                for slider, value in zip(robot.slider_handles, cfg):
                    slider.value = value
            finally:
                robot.suppress_slider_callbacks = False

            robot.urdf.update_cfg(np.array(cfg, dtype=float))
            update_link_frame_visuals(robot)
