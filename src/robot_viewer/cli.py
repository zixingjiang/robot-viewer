from __future__ import annotations

import os
import tempfile
import threading
import webbrowser
from typing import Annotated, Optional

import tyro
import viser
from viser import _messages
from viser._gui_handles import (
    GuiButtonHandle,
    GuiEvent,
    GuiUploadButtonHandle,
    UploadedFile,
)

from .ik import ik_worker_loop
from .state import ViewerState
from .ui import (
    load_robot_description,
    load_urdf_file,
    prune_stale_robot_roots,
    setup_file_actions,
    setup_viewer_actions,
)
from .utils import safe_write_file


def main(
    path: Annotated[
        tyro.conf.Positional[Optional[str]],
        tyro.conf.arg(
            help=(
                "Optional startup target. Treated as a URDF path by default, "
                "or as a robot_descriptions entry when --rd is set."
            )
        ),
    ] = None,
    host: Annotated[
        str,
        tyro.conf.arg(help="Host interface to bind the viewer server to."),
    ] = "0.0.0.0",
    port: Annotated[
        int,
        tyro.conf.arg(help="Port to serve the viewer web app on."),
    ] = 8080,
    rd: Annotated[
        bool,
        tyro.conf.arg(
            help=(
                "Interpret the positional startup target as a "
                "robot_descriptions entry name."
            )
        ),
    ] = False,
    open_browser: Annotated[
        bool,
        tyro.conf.arg(help="Automatically open the viewer in a browser tab."),
    ] = True,
) -> None:
    """Start a web-based robot viewer at http://{host}:{port}."""
    label = "Robot Viewer"
    load_meshes = True
    show_grid = True

    state = ViewerState(tmp_dir=tempfile.mkdtemp(prefix="robot_viewer_"))
    state.show_ground_plane = show_grid

    server = viser.ViserServer(host=host, port=port, label=label)

    if open_browser:
        browser_host = host
        if host in {"0.0.0.0", "::"}:
            browser_host = "localhost"
        viewer_url = f"http://{browser_host}:{port}"
        # Open asynchronously so startup is not blocked by browser behavior.
        threading.Thread(
            target=webbrowser.open,
            args=(viewer_url,),
            kwargs={"new": 2},
            daemon=True,
        ).start()

    status_text = setup_viewer_actions(server)
    file_text, upload_button, description_dropdown, load_description_button = (
        setup_file_actions(server)
    )

    @server.on_client_connect
    def _on_client_connect(_client: viser.ClientHandle) -> None:
        prune_stale_robot_roots(server, state)

    def _reload_connected_pages() -> None:
        for client in server.get_clients().values():
            try:
                client._websock_connection.queue_message(
                    _messages.RunJavascriptMessage(source="window.location.reload();")
                )
                client.flush()
            except Exception:
                pass

    if path is not None:
        if rd:
            status_text.value = f"Loading {path}..."
            with state.load_lock:
                try:
                    resolved_name = load_robot_description(
                        server,
                        state,
                        path,
                        status_text,
                        load_meshes=load_meshes,
                    )
                    file_text.value = f"{resolved_name} (robot_descriptions)"
                    status_text.value = f"Loaded {resolved_name}."
                    _reload_connected_pages()
                except Exception as exc:
                    file_text.value = "No file loaded."
                    status_text.value = (
                        f"Failed to load robot_descriptions entry {path}: {exc!r}"
                    )
        else:
            resolved_path = os.path.abspath(path)
            file_name = os.path.basename(resolved_path)

            if not os.path.isfile(resolved_path):
                file_text.value = "No file loaded."
                status_text.value = f"Startup file not found: {resolved_path}"
            else:
                with state.load_lock:
                    try:
                        status_text.value = f"Loading {file_name}..."
                        load_urdf_file(
                            server,
                            state,
                            resolved_path,
                            status_text,
                            load_meshes=load_meshes,
                        )
                        file_text.value = file_name
                        status_text.value = f"Loaded {file_name}."
                        _reload_connected_pages()
                    except Exception as exc:
                        file_text.value = "No file loaded."
                        status_text.value = f"Failed to load {file_name}: {exc!r}"

    @upload_button.on_upload
    def _on_upload(event: GuiEvent[GuiUploadButtonHandle]) -> None:
        uploaded: UploadedFile = event.target.value
        if uploaded is None:
            return

        status_text.value = f"Loading {uploaded.name}..."

        with state.load_lock:
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
                _reload_connected_pages()
            except Exception as exc:
                file_text.value = "No file loaded."
                status_text.value = f"Failed to load {uploaded.name}: {exc!r}"

    if description_dropdown is not None and load_description_button is not None:

        @load_description_button.on_click
        def _on_load_robot_description(_event: GuiEvent[GuiButtonHandle]) -> None:
            selected_name = description_dropdown.value
            status_text.value = f"Loading {selected_name}..."

            with state.load_lock:
                try:
                    resolved_name = load_robot_description(
                        server,
                        state,
                        selected_name,
                        status_text,
                        load_meshes=load_meshes,
                    )
                    file_text.value = f"{resolved_name} (robot_descriptions)"
                    status_text.value = f"Loaded {resolved_name}."
                    _reload_connected_pages()
                except Exception as exc:
                    file_text.value = "No file loaded."
                    status_text.value = f"Failed to load robot_descriptions entry {selected_name}: {exc!r}"

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


def run() -> None:
    tyro.cli(main)


if __name__ == "__main__":
    run()
