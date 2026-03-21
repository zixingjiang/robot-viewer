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
from viser._scene_handles import FrameHandle, Gui3dContainerHandle
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
    cartesian_folder_handle: Optional[GuiFolderHandle] = None
    transform_folder_handle: Optional[GuiFolderHandle] = None
    visibility_folder_handle: Optional[GuiFolderHandle] = None
    visibility_visual_checkbox: Optional[GuiCheckboxHandle] = None
    visibility_frames_checkbox: Optional[GuiCheckboxHandle] = None
    visibility_frame_names_checkbox: Optional[GuiCheckboxHandle] = None
    visibility_ground_checkbox: Optional[GuiCheckboxHandle] = None
    show_visual_meshes: bool = True
    show_link_frames: bool = False
    show_frame_names: bool = False
    show_ground_plane: bool = True
    link_frame_handles: dict[str, FrameHandle] = field(default_factory=dict)
    frame_name_handles: dict[str, Gui3dContainerHandle] = field(default_factory=dict)
    cartesian_frame_dropdown: Optional[GuiDropdownHandle[str]] = None
    transform_from_dropdown: Optional[GuiDropdownHandle[str]] = None
    transform_to_dropdown: Optional[GuiDropdownHandle[str]] = None
    transform_translation_text: Optional[viser.GuiInputHandle[str]] = None
    transform_rotation_text: Optional[viser.GuiInputHandle[str]] = None
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
    suppress_transform_text_callbacks: bool = False
    ik_lock: threading.Lock = field(default_factory=threading.Lock)
    tmp_dir: str = ""
