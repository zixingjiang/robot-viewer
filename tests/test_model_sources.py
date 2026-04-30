from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from robot_viewer.loader import (
    PathModelSource,
    RobotDescriptionModelSource,
    UploadedModelSource,
)


class ModelSourcesTests(unittest.TestCase):
    def test_path_model_source_loads_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            relative_path = os.path.join(temp_dir, "robot.urdf")
            with open(relative_path, "w", encoding="utf-8") as urdf_file:
                urdf_file.write("<robot name='test'/>")

            with patch("robot_viewer.loader.load_urdf", return_value="URDF") as mock:
                source = PathModelSource(relative_path)
                result = source.load(load_meshes=True, tmp_dir=temp_dir)

            self.assertEqual(mock.call_count, 1)
            self.assertEqual(mock.call_args.args, (os.path.abspath(relative_path), True))
            self.assertEqual(result.source_path, os.path.abspath(relative_path))
            self.assertEqual(result.file_label, "robot.urdf")
            self.assertEqual(result.status_label, "robot.urdf")
            self.assertEqual(result.urdf, "URDF")

    def test_path_model_source_missing_file_raises(self) -> None:
        source = PathModelSource("does-not-exist.urdf")
        with self.assertRaises(FileNotFoundError):
            source.load(load_meshes=True, tmp_dir=tempfile.gettempdir())

    def test_uploaded_model_source_writes_temp_file_then_loads(self) -> None:
        uploaded = SimpleNamespace(name="uploaded.urdf", content=b"<robot/>")
        with patch(
            "robot_viewer.loader.safe_write_file",
            return_value="/tmp/uploaded.urdf",
        ) as mock_write:
            with patch(
                "robot_viewer.loader.load_urdf",
                return_value="UPLOADED_URDF",
            ) as mock_load:
                source = UploadedModelSource(uploaded)
                result = source.load(load_meshes=False, tmp_dir="/tmp")

        self.assertEqual(mock_write.call_count, 1)
        self.assertEqual(mock_write.call_args.args, (uploaded, "/tmp"))
        self.assertEqual(mock_load.call_count, 1)
        self.assertEqual(mock_load.call_args.args, ("/tmp/uploaded.urdf", False))
        self.assertEqual(result.source_path, "/tmp/uploaded.urdf")
        self.assertEqual(result.file_label, "uploaded.urdf")
        self.assertEqual(result.status_label, "uploaded.urdf")
        self.assertEqual(result.urdf, "UPLOADED_URDF")

    def test_robot_description_model_source_loads_resolved_name(self) -> None:
        with patch(
            "robot_viewer.loader.load_robot_description_urdf",
            return_value=("ur5_description", "URDF_OBJECT", "/tmp/ur5.urdf"),
        ) as mock_load:
            source = RobotDescriptionModelSource("ur5")
            result = source.load(load_meshes=True, tmp_dir="/tmp")

        self.assertEqual(mock_load.call_count, 1)
        self.assertEqual(mock_load.call_args.args, ("ur5",))
        self.assertEqual(result.source_path, "/tmp/ur5.urdf")
        self.assertEqual(result.file_label, "ur5_description (robot_descriptions)")
        self.assertEqual(result.status_label, "ur5_description")
        self.assertEqual(result.urdf, "URDF_OBJECT")


if __name__ == "__main__":
    unittest.main()
