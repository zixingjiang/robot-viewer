from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import pink
import viser
from pink.tasks import DampingTask, FrameTask, PostureTask
from viser._gui_handles import (
    GuiButtonHandle,
    GuiCheckboxHandle,
    GuiDropdownHandle,
    GuiFolderHandle,
)
from viser._scene_handles import FrameHandle, Gui3dContainerHandle, TransformControlsHandle
from viser.extras import ViserUrdf


@dataclass
class RobotInstance:
    name: str
    urdf: ViserUrdf
    root_name: str
    tab_handle: Any = None
    remove_button_handle: Any = None

    show_visual_meshes: bool = True
    show_link_frames: bool = False
    show_frame_names: bool = False
    show_root_control: bool = False

    visibility_folder_handle: GuiFolderHandle | None = None
    visibility_visual_checkbox: GuiCheckboxHandle | None = None
    visibility_frames_checkbox: GuiCheckboxHandle | None = None
    visibility_frame_names_checkbox: GuiCheckboxHandle | None = None
    visibility_root_control_checkbox: GuiCheckboxHandle | None = None

    control_folder_handle: GuiFolderHandle | None = None
    slider_handles: list[viser.GuiInputHandle[float]] | None = None
    joint_names: list[str] | None = None
    initial_config: list[float] | None = None
    joint_limits: list[tuple[float, float]] | None = None
    randomize_button: GuiButtonHandle | None = None
    reset_button: GuiButtonHandle | None = None
    suppress_slider_callbacks: bool = False

    cartesian_folder_handle: GuiFolderHandle | None = None
    cartesian_mode_checkbox: GuiCheckboxHandle | None = None
    cartesian_frame_dropdown: GuiDropdownHandle[str] | None = None
    cartesian_target_handle: TransformControlsHandle | None = None

    ik_enabled: bool = False
    ik_configuration: pink.Configuration | None = None
    ik_tasks: list[Any] = field(default_factory=list)
    ik_frame_task: FrameTask | None = None
    ik_posture_task: PostureTask | None = None
    ik_damping_task: DampingTask | None = None
    ik_joint_name_to_q_index: dict[str, int] = field(default_factory=dict)
    ik_solver: str | None = None
    ik_frame_position_cost: float = 1.0
    ik_frame_orientation_cost: float = 1.0
    ik_posture_cost: float = 1e-3
    ik_damping_cost: float = 1e-3
    ik_lock: threading.Lock = field(default_factory=threading.Lock)

    transform_folder_handle: GuiFolderHandle | None = None
    transform_from_dropdown: GuiDropdownHandle[str] | None = None
    transform_to_dropdown: GuiDropdownHandle[str] | None = None
    transform_translation_text: viser.GuiInputHandle[str] | None = None
    transform_rotation_text: viser.GuiInputHandle[str] | None = None
    suppress_transform_text_callbacks: bool = False

    root_frame_handle: FrameHandle | None = None
    root_control_handle: TransformControlsHandle | None = None

    link_frame_handles: dict[str, FrameHandle] = field(default_factory=dict)
    frame_name_handles: dict[str, Gui3dContainerHandle] = field(default_factory=dict)


@dataclass
class ViewerState:
    tab_group_handle: Any = None
    robots: dict[str, RobotInstance] = field(default_factory=dict)
    show_ground_plane: bool = True
    ground_plane_handle: Any = None
    ground_plane_size: tuple[float, float] = (2.0, 2.0)
    remove_robots_folder: Any = None
    show_world_frame: bool = False
    ik_running: bool = True
    ik_dt: float = 1.0 / 30.0
    load_lock: threading.Lock = field(default_factory=threading.Lock)
    tmp_dir: str = ""
