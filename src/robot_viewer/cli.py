from __future__ import annotations

import tempfile
import threading
from typing import Optional

import tyro
import viser
from viser._gui_handles import GuiEvent, GuiUploadButtonHandle, UploadedFile

from .ik import ik_worker_loop
from .state import ViewerState
from .ui import load_urdf_file, setup_file_actions, setup_viewer_actions
from .utils import safe_write_file


def main(
    host: str = "0.0.0.0",
    port: int = 8080,
    label: Optional[str] = "Robot Viewer",
    load_meshes: bool = True,
    show_grid: bool = True,
) -> None:
    """Start a robot viewer server.

    The viewer will be served at http://{host}:{port}. Use the "Upload URDF" button
    in the GUI to select a local URDF file, and it will be visualized in the 3D view.

    The visualization is powered by the `viser` library.
    """

    state = ViewerState(tmp_dir=tempfile.mkdtemp(prefix="robot_viewer_"))
    state.show_ground_plane = show_grid

    server = viser.ViserServer(host=host, port=port, label=label)

    status_text = setup_viewer_actions(server)
    file_text, upload_button = setup_file_actions(server)

    @upload_button.on_upload
    def _on_upload(event: GuiEvent[GuiUploadButtonHandle]) -> None:
        uploaded: UploadedFile = event.target.value
        if uploaded is None:
            return

        status_text.value = f"Loading {uploaded.name}..."

        try:
            path = safe_write_file(uploaded, state.tmp_dir)
            file_text.value = uploaded.name
            load_urdf_file(
                server,
                state,
                path,
                status_text,
                load_meshes=load_meshes,
            )
            status_text.value = f"Loaded {uploaded.name}."
        except Exception as exc:
            file_text.value = "No file loaded."
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
