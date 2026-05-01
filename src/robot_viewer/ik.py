from __future__ import annotations

import time
from typing import Any

import numpy as np

from .mjcf import mink_ik_step
from .state import ViewerState
from .viewer import update_link_frame_visuals


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
                status_text.value = f"Cartesian IK error: {ik_error}"
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
