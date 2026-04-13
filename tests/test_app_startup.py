from __future__ import annotations

import unittest
from unittest.mock import patch

from robot_viewer.app import start_viewer_app


class _FakeThread:
    def __init__(self, target=None, args=(), kwargs=None, daemon=False):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.daemon = daemon
        self.started = False
        self.join_timeout = None

    def start(self) -> None:
        self.started = True

    def join(self, timeout=None) -> None:
        self.join_timeout = timeout


class _FakeServer:
    def __init__(self, host: str, port: int, label: str):
        self.host = host
        self.port = port
        self.label = label
        self._connect_handler = None
        self.stopped = False

    def on_client_connect(self, fn):
        self._connect_handler = fn
        return fn

    def sleep_forever(self) -> None:
        raise KeyboardInterrupt

    def stop(self) -> None:
        self.stopped = True


class AppStartupTests(unittest.TestCase):
    def test_start_viewer_app_wires_startup_and_handlers(self) -> None:
        fake_server = _FakeServer("127.0.0.1", 9090, "Robot Viewer")
        fake_file_text = object()
        fake_upload_button = object()
        fake_dropdown = object()
        fake_load_button = object()
        created_threads: list[_FakeThread] = []

        def _thread_factory(*args, **kwargs):
            thread = _FakeThread(*args, **kwargs)
            created_threads.append(thread)
            return thread

        with patch("robot_viewer.app.tempfile.mkdtemp", return_value="/tmp/rv"):
            with patch("robot_viewer.app.viser.ViserServer", return_value=fake_server):
                with patch("robot_viewer.app.threading.Thread", side_effect=_thread_factory):
                    with patch(
                        "robot_viewer.app.setup_viewer_actions", return_value="status-handle"
                    ):
                        with patch(
                            "robot_viewer.app.setup_file_actions",
                            return_value=(
                                fake_file_text,
                                fake_upload_button,
                                fake_dropdown,
                                fake_load_button,
                            ),
                        ):
                            with patch("robot_viewer.app.load_startup_target") as load_startup:
                                with patch(
                                    "robot_viewer.app.register_file_event_handlers"
                                ) as register_handlers:
                                    with patch("robot_viewer.app.prune_stale_robot_roots"):
                                        start_viewer_app(
                                            path="robot.urdf",
                                            host="127.0.0.1",
                                            port=9090,
                                            rd=False,
                                            open_browser=False,
                                        )

        self.assertEqual(load_startup.call_count, 1)
        self.assertEqual(register_handlers.call_count, 1)
        self.assertTrue(fake_server.stopped)
        self.assertEqual(len(created_threads), 1)
        self.assertTrue(created_threads[0].started)
        self.assertEqual(created_threads[0].join_timeout, 1.0)


if __name__ == "__main__":
    unittest.main()
