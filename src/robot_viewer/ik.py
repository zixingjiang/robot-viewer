from __future__ import annotations

import time
from typing import Any

import numpy as np
import qpsolvers
import viser

from .state import RobotInstance, ViewerState
from .utils import rotation_matrix_to_wxyz, wxyz_to_rotation_matrix
from .viewer import update_link_frame_visuals

# ---------------------------------------------------------------------------
# Mink IK availability
# ---------------------------------------------------------------------------

_MINK_AVAILABLE = True

try:
    import mujoco
    from mujoco import mj_id2name, mjtObj
except ImportError:
    pass

try:
    import mink
    from mink import SE3, SO3
except ImportError:
    _MINK_AVAILABLE = False


def _body_name(mj_model: Any, body_id: int) -> str:
    name = mj_id2name(mj_model, mjtObj.mjOBJ_BODY, body_id)
    return name if name else f"body_{body_id}"


# ---------------------------------------------------------------------------
# Mink IK setup and step
# ---------------------------------------------------------------------------


def setup_mink_ik(
    server: viser.ViserServer,
    robot: RobotInstance,
    mj_model: Any,
    status_text: Any,
    cartesian_mode_checkbox: viser.GuiInputHandle[bool],
) -> None:
    """Set up mink IK for a MuJoCo robot — creates GUI elements and callbacks."""
    if not _MINK_AVAILABLE:
        cartesian_mode_checkbox.disabled = True
        server.gui.add_text("Cartesian IK", "mink is not installed.")
        return

    try:
        configuration = mink.Configuration(mj_model)
        robot.ik_configuration = configuration
        robot.ik_solver = (
            "daqp" if "daqp" in qpsolvers.available_solvers
            else qpsolvers.available_solvers[0]
        )

        frame_options = sorted(
            _body_name(mj_model, i) for i in range(1, mj_model.nbody)
        )
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
                or robot.slider_handles is None
                or robot.qpos_adrs is None
            ):
                return
            q = np.array(robot.ik_configuration.q, copy=True)
            for adr, slider in zip(robot.qpos_adrs, robot.slider_handles):
                q[adr] = slider.value
            robot.ik_configuration.update(q)

        def _set_ik_tasks_from_current_configuration(frame_name: str) -> None:
            if robot.ik_configuration is None:
                return

            with robot.ik_lock:
                frame_task = mink.FrameTask(
                    frame_name,
                    frame_type="body",
                    position_cost=robot.ik_frame_position_cost,
                    orientation_cost=robot.ik_frame_orientation_cost,
                    lm_damping=1.0,
                )
                posture_task = mink.PostureTask(mj_model, cost=robot.ik_posture_cost)
                posture_task.set_target_from_configuration(
                    robot.ik_configuration
                )
                config_limit = mink.ConfigurationLimit(mj_model)

                T_world = robot.ik_configuration.get_transform_frame_to_world(
                    frame_name, frame_type="body"
                )

                if robot.urdf is not None:
                    try:
                        viewer_pose = robot.urdf._urdf.get_transform(
                            frame_name
                        )
                        if viewer_pose is not None:
                            T_world = SE3.from_matrix(
                                np.asarray(viewer_pose, dtype=float)
                            )
                    except Exception:
                        pass

                frame_task.set_target(T_world)

                robot.ik_tasks = [frame_task, posture_task]
                robot.ik_limits = [config_limit]

            if robot.cartesian_target_handle is not None:
                robot.cartesian_target_handle.position = tuple(
                    T_world.translation()
                )
                robot.cartesian_target_handle.wxyz = rotation_matrix_to_wxyz(
                    T_world.rotation().as_matrix()
                )

        @frame_task_position_cost_slider.on_update
        def _on_position_cost(_event: object) -> None:
            robot.ik_frame_position_cost = frame_task_position_cost_slider.value
            if robot.ik_tasks:
                robot.ik_tasks[0].set_position_cost(robot.ik_frame_position_cost)

        @frame_task_orientation_cost_slider.on_update
        def _on_orientation_cost(_event: object) -> None:
            robot.ik_frame_orientation_cost = frame_task_orientation_cost_slider.value
            if robot.ik_tasks:
                robot.ik_tasks[0].set_orientation_cost(
                    robot.ik_frame_orientation_cost
                )

        @posture_cost_slider.on_update
        def _on_posture_cost(_event: object) -> None:
            robot.ik_posture_cost = posture_cost_slider.value
            if len(robot.ik_tasks) > 1:
                robot.ik_tasks[1].set_cost(robot.ik_posture_cost)

        @cartesian_target_handle.on_update
        def _on_target_update(event: viser.TransformControlsEvent) -> None:
            frame_task = robot.ik_tasks[0] if robot.ik_tasks else None
            if frame_task is None:
                return
            new_target = SE3.from_rotation_and_translation(
                SO3.from_matrix(
                    wxyz_to_rotation_matrix(
                        (
                            float(event.target.wxyz[0]),
                            float(event.target.wxyz[1]),
                            float(event.target.wxyz[2]),
                            float(event.target.wxyz[3]),
                        )
                    )
                ),
                np.array(event.target.position),
            )
            frame_task.set_target(new_target)

        @cartesian_frame_dropdown.on_update
        def _on_frame_change(_event: object) -> None:
            if robot.cartesian_frame_dropdown is None:
                return
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


def mink_ik_step(robot: RobotInstance, dt: float) -> np.ndarray | None:
    """Run one mink IK step. Returns updated joint values or None on error."""
    if not _MINK_AVAILABLE:
        return None

    with robot.ik_lock:
        velocity = mink.solve_ik(
            robot.ik_configuration,
            robot.ik_tasks,
            dt,
            solver=robot.ik_solver,
            limits=robot.ik_limits,
        )
        robot.ik_configuration.integrate_inplace(velocity, dt)
        return np.array(robot.ik_configuration.q, copy=True)


def ik_worker_loop(state: ViewerState, status_text: Any) -> None:
    while state.ik_running:
        time.sleep(state.ik_dt)

        for robot in list(state.robots.values()):
            if not robot.ik_enabled:
                continue

            if not robot.ik_tasks or robot.ik_solver is None:
                continue
            if robot.slider_handles is None or robot.qpos_adrs is None:
                continue

            try:
                q = mink_ik_step(robot, state.ik_dt)
            except Exception as ik_error:
                robot.ik_enabled = False
                if robot.cartesian_mode_checkbox is not None:
                    robot.cartesian_mode_checkbox.value = False
                status_text.value = f"Cartesian IK error: {ik_error!r}"
                continue

            cfg = np.array([float(q[adr]) for adr in robot.qpos_adrs], dtype=float)

            robot.suppress_slider_callbacks = True
            try:
                for slider, value in zip(robot.slider_handles, cfg):
                    slider.value = value
            finally:
                robot.suppress_slider_callbacks = False

            if robot.mjcf_handle is not None:
                robot.mjcf_handle.set_joint_values(cfg, robot.qpos_adrs)
            elif robot.urdf is not None:
                robot.urdf.update_cfg(cfg)
            update_link_frame_visuals(robot)
