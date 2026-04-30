from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from robot_viewer.loader import _detect_format, load_mjcf


_MINIMAL_URDF = """<?xml version="1.0"?>
<robot name="test">
  <link name="base"/>
</robot>
"""

_MINIMAL_MJCF = """<?xml version="1.0"?>
<mujoco model="test">
  <worldbody>
    <body name="base">
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


class FormatDetectionTests(unittest.TestCase):
    def test_detect_urdf(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "robot.urdf")
            Path(path).write_text(_MINIMAL_URDF)
            self.assertEqual(_detect_format(path), "urdf")

    def test_detect_mjcf(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "robot.xml")
            Path(path).write_text(_MINIMAL_MJCF)
            self.assertEqual(_detect_format(path), "mjcf")

    def test_detect_mjb(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.mjb")
            Path(path).write_bytes(b"\x00\x01")
            self.assertEqual(_detect_format(path), "mjcf")

    def test_detect_unknown_extension(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.txt")
            Path(path).write_text("hello")
            self.assertEqual(_detect_format(path), "unknown")

    def test_detect_unknown_xml_tag(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.xml")
            Path(path).write_text("<unknown_root/>")
            self.assertEqual(_detect_format(path), "unknown")


class MjcfLoadingTests(unittest.TestCase):
    def test_load_minimal_mjcf(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.xml")
            Path(path).write_text(_MINIMAL_MJCF)
            model, data = load_mjcf(path)
            self.assertIsNotNone(model)
            self.assertIsNotNone(data)
            self.assertEqual(model.nbody, 2)  # world + base
            self.assertEqual(model.ngeom, 1)

    def test_load_mjcf_nonexistent_raises(self) -> None:
        with self.assertRaises(Exception):
            load_mjcf("/nonexistent/file.xml")


if __name__ == "__main__":
    unittest.main()
