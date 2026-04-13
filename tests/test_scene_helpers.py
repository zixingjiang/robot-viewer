from __future__ import annotations

import unittest

import numpy as np

from robot_viewer.scene import compute_ground_plane_size


class _FakeUrdfCore:
    def __init__(self, transforms: dict[str, np.ndarray]) -> None:
        self.link_map = {name: object() for name in transforms.keys()}
        self._transforms = transforms

    def get_transform(self, link_name: str) -> np.ndarray:
        return self._transforms[link_name]


class _FakeViserUrdf:
    def __init__(self, urdf_core: _FakeUrdfCore) -> None:
        self._urdf = urdf_core


class _FakeState:
    def __init__(self, current_urdf: _FakeViserUrdf | None) -> None:
        self.current_urdf = current_urdf


class SceneHelpersTests(unittest.TestCase):
    def test_compute_ground_plane_size_defaults_without_robot(self) -> None:
        state = _FakeState(current_urdf=None)
        self.assertEqual(compute_ground_plane_size(state), (2, 2))

    def test_compute_ground_plane_size_from_link_transforms(self) -> None:
        transforms = {
            "base": np.array(
                [
                    [1.0, 0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0, -0.5],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            "tool": np.array(
                [
                    [1.0, 0.0, 0.0, 2.2],
                    [0.0, 1.0, 0.0, 1.8],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        }
        state = _FakeState(_FakeViserUrdf(_FakeUrdfCore(transforms)))
        self.assertEqual(compute_ground_plane_size(state), (6, 4))


if __name__ == "__main__":
    unittest.main()
