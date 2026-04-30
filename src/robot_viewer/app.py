from __future__ import annotations

import tempfile
import threading
import webbrowser
from typing import Optional

import viser

from .ik import ik_worker_loop
from .scene import prune_stale_robot_roots, set_ground_plane_visible, set_world_frame_visible
from .state import ViewerState
from .viewer import (
    load_startup_target,
    register_file_event_handlers,
    setup_global_gui,
)


def start_viewer_app(
    *,
    path: Optional[str],
    host: str,
    port: int,
    rd: bool,
    open_browser: bool,
) -> None:
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

    status_text, upload_button, description_dropdown, load_description_button = (
        setup_global_gui(server, state)
    )

    set_ground_plane_visible(server, state, state.show_ground_plane)
    set_world_frame_visible(server, state, state.show_world_frame)

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
            load_meshes=load_meshes,
        )

    register_file_event_handlers(
        server=server,
        state=state,
        status_text=status_text,
        upload_button=upload_button,
        description_dropdown=description_dropdown,
        load_description_button=load_description_button,
        load_meshes=load_meshes,
    )

    ik_thread = threading.Thread(target=ik_worker_loop, args=(state, status_text), daemon=True)
    ik_thread.start()

    try:
        server.sleep_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.ik_running = False
        ik_thread.join(timeout=1.0)
        server.stop()
