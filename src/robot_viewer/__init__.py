from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tyro
import viser
import yourdfpy  # type: ignore[import]
from viser.extras import ViserUrdf
from viser._gui_handles import (
    GuiEvent,
    GuiFolderHandle,
    GuiUploadButtonHandle,
    UploadedFile,
)


@dataclass
class _ViewerState:
    """Mutable state used while the server is running."""

    current_urdf: Optional[ViserUrdf] = None
    current_root_name: Optional[str] = None
    control_folder_handle: Optional[GuiFolderHandle] = None
    slider_handles: Optional[list[viser.GuiInputHandle[float]]] = None
    initial_config: Optional[list[float]] = None
    tmp_dir: str = ""


def _create_robot_control_sliders(
    server: viser.ViserServer, viser_urdf: ViserUrdf
) -> tuple[list[viser.GuiInputHandle[float]], list[float]]:
    """Create sliders to control joints for a loaded URDF."""

    slider_handles: list[viser.GuiInputHandle[float]] = []
    initial_config: list[float] = []

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
            viser_urdf.update_cfg(np.array([h.value for h in handles]))

        slider.on_update(_on_slider_update)
        slider_handles.append(slider)
        initial_config.append(initial_pos)

    return slider_handles, initial_config


def _safe_write_file(uploaded_file: UploadedFile, tmp_dir: str) -> str:
    """Write the uploaded file to a temporary directory and return the path."""

    os.makedirs(tmp_dir, exist_ok=True)
    out_path = os.path.join(tmp_dir, os.path.basename(uploaded_file.name))
    with open(out_path, "wb") as f:
        f.write(uploaded_file.content)
    return out_path


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

    status_text = server.gui.add_text("Status", "Upload a URDF file to begin.")

    upload_button = server.gui.add_upload_button(
        "Upload URDF",
        mime_type="*/*",
        hint="Select a URDF file (.urdf, .xml).",
    )

    def _clear_previous_robot() -> None:
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
        state.control_folder_handle = server.gui.add_folder("Robot controls")
        with state.control_folder_handle:
            slider_handles, initial_config = _create_robot_control_sliders(
                server, viser_urdf
            )
            state.slider_handles = slider_handles
            state.initial_config = initial_config

            # Ensure the robot starts in the zero joint configuration by default.
            if slider_handles:
                viser_urdf.update_cfg(np.zeros(len(slider_handles)))

            reset_button = server.gui.add_button("Reset pose")

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

    try:
        server.sleep_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    tyro.cli(main)
