from __future__ import annotations

import tempfile
import threading
from typing import Optional

import tyro
import viser
from viser._gui_handles import GuiEvent, GuiUploadButtonHandle, UploadedFile

from .ik import ik_worker_loop
from .state import ViewerState
from .ui import load_urdf_file
from .utils import safe_write_file


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

    state = ViewerState(tmp_dir=tempfile.mkdtemp(prefix="robot_viewer_"))

    server = viser.ViserServer(host=host, port=port, label=label)

    status_text = server.gui.add_text("Status", "Open a URDF file to begin.")

    upload_button = server.gui.add_upload_button(
        "Open URDF",
        mime_type="*/*",
        hint="Select a URDF file (.urdf, .xml).",
    )

    @upload_button.on_upload
    def _on_upload(event: GuiEvent[GuiUploadButtonHandle]) -> None:
        uploaded: UploadedFile = event.target.value
        if uploaded is None:
            return

        status_text.value = f"Loading {uploaded.name}..."

        try:
            path = safe_write_file(uploaded, state.tmp_dir)
            load_urdf_file(
                server,
                state,
                path,
                status_text,
                load_meshes=load_meshes,
                load_collision_meshes=load_collision_meshes,
                show_grid=show_grid,
            )
            status_text.value = f"Loaded {uploaded.name}."
        except Exception as exc:
            status_text.value = f"Failed to load {uploaded.name}: {exc!r}"

    ik_thread = threading.Thread(
        target=ik_worker_loop, args=(state, status_text), daemon=True
    )
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
