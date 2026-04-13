from __future__ import annotations

from typing import Annotated, Optional

import tyro

from .app import start_viewer_app


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
    start_viewer_app(
        path=path,
        host=host,
        port=port,
        rd=rd,
        open_browser=open_browser,
    )


def run() -> None:
    tyro.cli(main)


if __name__ == "__main__":
    run()
