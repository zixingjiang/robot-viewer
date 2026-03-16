from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import pink
import viser
from pink.tasks import DampingTask, FrameTask, PostureTask
from viser._gui_handles import (
    GuiButtonHandle,
    GuiCheckboxHandle,
    GuiDropdownHandle,
    GuiFolderHandle,
)
from viser._scene_handles import TransformControlsHandle
from viser.extras import ViserUrdf


@dataclass
class ViewerState:
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
