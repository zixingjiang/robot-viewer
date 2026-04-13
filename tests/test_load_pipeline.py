from __future__ import annotations

import threading
import unittest

from robot_viewer.load_pipeline import execute_model_load
from robot_viewer.model_sources import LoadResult, StartupFileNotFoundError


class _Handle:
    def __init__(self) -> None:
        self.value = ""


class _State:
    def __init__(self, *, has_previous_robot: bool) -> None:
        self.load_lock = threading.Lock()
        self.tmp_dir = "/tmp"
        self.current_urdf = object() if has_previous_robot else None


class _Source:
    def __init__(self, *, label: str, result: LoadResult | None, error: Exception | None):
        self.loading_label = label
        self._result = result
        self._error = error

    def load(self, *, load_meshes: bool, tmp_dir: str) -> LoadResult:
        del load_meshes
        del tmp_dir
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


class LoadPipelineTests(unittest.TestCase):
    def test_success_updates_labels_and_calls_mount(self) -> None:
        state = _State(has_previous_robot=False)
        status_text = _Handle()
        file_text = _Handle()
        mounted: list[tuple[object, str]] = []
        reload_calls: list[str] = []

        source = _Source(
            label="arm.urdf",
            result=LoadResult(
                urdf="URDF",
                source_path="/tmp/arm.urdf",
                file_label="arm.urdf",
                status_label="arm.urdf",
            ),
            error=None,
        )

        execute_model_load(
            state=state,
            source=source,
            status_text=status_text,
            file_text=file_text,
            load_meshes=True,
            mount_loaded_robot=lambda urdf, source_path: mounted.append(
                (urdf, source_path)
            ),
            reload_connected_pages=lambda: reload_calls.append("reload"),
        )

        self.assertEqual(mounted, [("URDF", "/tmp/arm.urdf")])
        self.assertEqual(status_text.value, "Loaded arm.urdf.")
        self.assertEqual(file_text.value, "arm.urdf")
        self.assertEqual(reload_calls, [])

    def test_success_with_previous_robot_triggers_reload(self) -> None:
        state = _State(has_previous_robot=True)
        status_text = _Handle()
        file_text = _Handle()
        reload_calls: list[str] = []

        source = _Source(
            label="arm.urdf",
            result=LoadResult(
                urdf="URDF",
                source_path="/tmp/arm.urdf",
                file_label="arm.urdf",
                status_label="arm.urdf",
            ),
            error=None,
        )

        execute_model_load(
            state=state,
            source=source,
            status_text=status_text,
            file_text=file_text,
            load_meshes=True,
            mount_loaded_robot=lambda _urdf, _source_path: None,
            reload_connected_pages=lambda: reload_calls.append("reload"),
        )

        self.assertEqual(reload_calls, ["reload"])

    def test_generic_failure_sets_failed_status(self) -> None:
        state = _State(has_previous_robot=False)
        status_text = _Handle()
        file_text = _Handle()
        source = _Source(label="arm.urdf", result=None, error=RuntimeError("boom"))

        execute_model_load(
            state=state,
            source=source,
            status_text=status_text,
            file_text=file_text,
            load_meshes=True,
            mount_loaded_robot=lambda _urdf, _source_path: None,
            reload_connected_pages=lambda: None,
        )

        self.assertEqual(file_text.value, "No file loaded.")
        self.assertEqual(
            status_text.value, "Failed to load arm.urdf: RuntimeError('boom')"
        )

    def test_startup_file_not_found_keeps_specific_message(self) -> None:
        state = _State(has_previous_robot=False)
        status_text = _Handle()
        file_text = _Handle()
        source = _Source(
            label="missing.urdf",
            result=None,
            error=StartupFileNotFoundError("/tmp/missing.urdf"),
        )

        execute_model_load(
            state=state,
            source=source,
            status_text=status_text,
            file_text=file_text,
            load_meshes=True,
            mount_loaded_robot=lambda _urdf, _source_path: None,
            reload_connected_pages=lambda: None,
        )

        self.assertEqual(file_text.value, "No file loaded.")
        self.assertEqual(status_text.value, "Startup file not found: /tmp/missing.urdf")


if __name__ == "__main__":
    unittest.main()
