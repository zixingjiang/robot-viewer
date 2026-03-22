from __future__ import annotations

import numpy as np

from .state import ViewerState
from .utils import rotation_matrix_to_wxyz


def update_transform_display(state: ViewerState) -> None:
    if (
        state.current_urdf is None
        or state.transform_from_dropdown is None
        or state.transform_to_dropdown is None
        or state.transform_translation_text is None
        or state.transform_rotation_text is None
    ):
        return

    from_frame = state.transform_from_dropdown.value
    to_frame = state.transform_to_dropdown.value

    try:
        urdf = state.current_urdf._urdf
        world_from = urdf.get_transform(from_frame)
        world_to = urdf.get_transform(to_frame)
        from_to = np.linalg.inv(world_from) @ world_to

        translation = from_to[:3, 3]
        rotation_wxyz = rotation_matrix_to_wxyz(from_to[:3, :3])

        state.suppress_transform_text_callbacks = True
        try:
            state.transform_translation_text.value = (
                f"{translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}"
            )
            state.transform_rotation_text.value = (
                f"{rotation_wxyz[0]:.4f}, {rotation_wxyz[1]:.4f}, "
                f"{rotation_wxyz[2]:.4f}, {rotation_wxyz[3]:.4f}"
            )
        finally:
            state.suppress_transform_text_callbacks = False
    except Exception:
        state.suppress_transform_text_callbacks = True
        try:
            state.transform_translation_text.value = "N/A"
            state.transform_rotation_text.value = "N/A"
        finally:
            state.suppress_transform_text_callbacks = False


def update_link_frame_visuals(state: ViewerState) -> None:
    if state.current_urdf is None:
        return

    urdf = state.current_urdf._urdf
    for link_name, frame_handle in state.link_frame_handles.items():
        try:
            transform = urdf.get_transform(link_name)
        except Exception:
            continue

        frame_handle.wxyz = rotation_matrix_to_wxyz(transform[:3, :3])
        frame_handle.position = (
            float(transform[0, 3]),
            float(transform[1, 3]),
            float(transform[2, 3]),
        )
        frame_handle.visible = state.show_link_frames

        name_handle = state.frame_name_handles.get(link_name)
        if name_handle is not None:
            name_handle.position = (
                float(transform[0, 3]),
                float(transform[1, 3]),
                float(transform[2, 3]),
            )
            name_handle.visible = state.show_frame_names

    update_transform_display(state)
