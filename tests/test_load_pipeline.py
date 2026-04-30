from __future__ import annotations

import threading
import unittest

from robot_viewer.loader import LoadResult, StartupFileNotFoundError, execute_model_load


class _Handle:
    def __init__(self) -> None:
        self.value = ""


class _State:
    def __init__(self) -> None:
        self.load_lock = threading.Lock()
        self.tmp_dir = "/tmp"


class _Source:
    def __init__(self, *, label: str, result: LoadResult | None, error: Exception | None):
        self.loading_label = label
        self._result = result
        self._error = error
        self._failure_label = label

    @property
    def failure_label(self) -> str:
        return self._failure_label

    def load(self, *, load_meshes: bool, tmp_dir: str) -> LoadResult:
        del load_meshes
        del tmp_dir
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


class LoadPipelineTests(unittest.TestCase):
    def test_success_updates_labels_and_calls_mount(self) -> None:
        state = _State()
        status_text = _Handle()
        mounted: list[tuple[object, str]] = []

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
            load_meshes=True,
            mount_loaded_robot=lambda urdf, source_path: mounted.append(
                (urdf, source_path)
            ),
        )

        self.assertEqual(mounted, [("URDF", "/tmp/arm.urdf")])
        self.assertEqual(status_text.value, "Loaded arm.urdf.")

    def test_generic_failure_sets_failed_status(self) -> None:
        state = _State()
        status_text = _Handle()
        source = _Source(label="arm.urdf", result=None, error=RuntimeError("boom"))

        execute_model_load(
            state=state,
            source=source,
            status_text=status_text,
            load_meshes=True,
            mount_loaded_robot=lambda _urdf, _source_path: None,
        )

        self.assertEqual(
            status_text.value, "Failed to load arm.urdf: RuntimeError('boom')"
        )

    def test_startup_file_not_found_keeps_specific_message(self) -> None:
        state = _State()
        status_text = _Handle()
        source = _Source(
            label="missing.urdf",
            result=None,
            error=StartupFileNotFoundError("/tmp/missing.urdf"),
        )

        execute_model_load(
            state=state,
            source=source,
            status_text=status_text,
            load_meshes=True,
            mount_loaded_robot=lambda _urdf, _source_path: None,
        )

        self.assertEqual(status_text.value, "Startup file not found: /tmp/missing.urdf")


if __name__ == "__main__":
    unittest.main()
