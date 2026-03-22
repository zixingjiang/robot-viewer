from __future__ import annotations

import tempfile
import threading
import webbrowser
from typing import Annotated, Optional

import tyro
import viser

from .ik import ik_worker_loop
from .loader import prune_stale_robot_roots
from .state import ViewerState
from .viewer import (
    load_startup_target,
    register_file_event_handlers,
    setup_file_actions,
    setup_viewer_actions,
)


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

    if path is not None:
        load_startup_target(
            server=server,
            state=state,
            path=path,
            rd=rd,
            status_text=status_text,
            file_text=file_text,
            load_meshes=load_meshes,
        )

    register_file_event_handlers(
        server=server,
        state=state,
        status_text=status_text,
        file_text=file_text,
        upload_button=upload_button,
        description_dropdown=description_dropdown,
        load_description_button=load_description_button,
        load_meshes=load_meshes,
    )

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
