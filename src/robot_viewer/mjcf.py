from __future__ import annotations

from typing import Any

import numpy as np
import qpsolvers
import trimesh
import viser
import viser.transforms as vtf

from .state import RobotInstance, ViewerState
from .utils import rotation_matrix_to_wxyz, wxyz_to_rotation_matrix

# ---------------------------------------------------------------------------
# Mesh extraction from MuJoCo model
# ---------------------------------------------------------------------------

_MUJOCO_AVAILABLE = True
_MINK_AVAILABLE = True

try:
    import mujoco
    from mujoco import mj_id2name, mjtGeom, mjtObj
except ImportError:
    _MUJOCO_AVAILABLE = False

try:
    import mink
    from mink import SE3, SO3
except ImportError:
    _MINK_AVAILABLE = False


def _geom_rgba(mj_model: Any, geom_id: int) -> np.ndarray:
    """Resolve flat RGBA for a geom (material color > geom color)."""
    matid = mj_model.geom_matid[geom_id]
    if matid >= 0 and matid < mj_model.nmat:
        rgba = mj_model.mat_rgba[matid].copy()
    else:
        rgba = mj_model.geom_rgba[geom_id].copy()
    if np.all(rgba == 0):
        rgba = np.array([0.5, 0.5, 0.5, 1.0])
    return rgba


def _create_primitive_trimesh(geom_type: int, size: np.ndarray) -> trimesh.Trimesh:
    """Create a trimesh for a primitive MuJoCo geom type."""
    gtype = int(geom_type)
    if gtype == int(mjtGeom.mjGEOM_SPHERE):
        return trimesh.creation.icosphere(radius=max(size[0], 0.001), subdivisions=2)
    elif gtype == int(mjtGeom.mjGEOM_BOX):
        return trimesh.creation.box(extents=2.0 * size)
    elif gtype == int(mjtGeom.mjGEOM_CAPSULE):
        return trimesh.creation.capsule(radius=max(size[0], 0.001), height=max(2.0 * size[1], 0.001))
    elif gtype == int(mjtGeom.mjGEOM_CYLINDER):
        return trimesh.creation.cylinder(radius=max(size[0], 0.001), height=max(2.0 * size[2], 0.001), sections=24)
    elif gtype == int(mjtGeom.mjGEOM_ELLIPSOID):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        mesh.apply_scale(np.maximum(size, 0.001))
        return mesh
    raise ValueError(f"Unsupported primitive geom type: {geom_type}")


def _extract_mesh_geom(mj_model: Any, geom_id: int) -> trimesh.Trimesh:
    """Extract vertices and faces for a mesh geom, returning a trimesh."""
    mesh_id = int(mj_model.geom_dataid[geom_id])
    vert_start = int(mj_model.mesh_vertadr[mesh_id])
    vert_count = int(mj_model.mesh_vertnum[mesh_id])
    face_start = int(mj_model.mesh_faceadr[mesh_id])
    face_count = int(mj_model.mesh_facenum[mesh_id])

    vertices = mj_model.mesh_vert[vert_start : vert_start + vert_count]
    faces = mj_model.mesh_face[face_start : face_start + face_count]
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _body_mesh(
    mj_model: Any, body_id: int, *, geom_filter: Any = None
) -> trimesh.Trimesh | None:
    """Merge geoms under a body into a single trimesh in body-local space.

    Args:
        mj_model: MuJoCo model.
        body_id: Body index.
        geom_filter: Optional callable ``(geom_id) -> bool`` to filter geoms.
            When None, all geoms are included.
    """
    geoms = []
    for i in range(mj_model.ngeom):
        if int(mj_model.geom_bodyid[i]) != body_id:
            continue
        if mj_model.geom_rgba[i, 3] == 0:
            continue
        if geom_filter is not None and not geom_filter(i):
            continue
        geoms.append(i)

    if not geoms:
        return None

    meshes: list[trimesh.Trimesh] = []
    for gid in geoms:
        gtype = int(mj_model.geom_type[gid])
        if gtype == int(mjtGeom.mjGEOM_PLANE):
            continue
        if gtype == int(mjtGeom.mjGEOM_MESH):
            mesh = _extract_mesh_geom(mj_model, gid)
        else:
            mesh = _create_primitive_trimesh(gtype, mj_model.geom_size[gid])
        rgba = _geom_rgba(mj_model, gid)
        mesh.visual = trimesh.visual.ColorVisuals(
            vertex_colors=np.tile(
                (np.clip(rgba[:3], 0, 1) * 255).astype(np.uint8),
                (len(mesh.vertices), 1),
            )
        )
        pos = mj_model.geom_pos[gid]
        quat = mj_model.geom_quat[gid]
        transform = np.eye(4)
        transform[:3, :3] = vtf.SO3(quat).as_matrix()
        transform[:3, 3] = pos
        mesh.apply_transform(transform)
        meshes.append(mesh)

    if not meshes:
        return None

    result = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
    try:
        result.merge_vertices()
    except Exception:
        pass
    return result


_COLLISION_TINT = np.array([60, 120, 255], dtype=np.uint8)


def _tint_mesh(mesh: trimesh.Trimesh, tint: np.ndarray) -> None:
    """Tint a mesh's vertex colors with a uniform color (ignoring original)."""
    mesh.visual = trimesh.visual.ColorVisuals(
        vertex_colors=np.tile(tint, (len(mesh.vertices), 1))
    )


def _is_fixed_body(mj_model: Any, body_id: int) -> bool:
    """Check if a body is fixed (welded to world, not mocap)."""
    is_weld = int(mj_model.body_weldid[body_id]) == 0
    root_id = int(mj_model.body_rootid[body_id])
    return bool(is_weld and mj_model.body_mocapid[root_id] < 0)


def _body_name(mj_model: Any, body_id: int) -> str:
    name = mj_id2name(mj_model, mjtObj.mjOBJ_BODY, body_id)
    return name if name else f"body_{body_id}"


# ---------------------------------------------------------------------------
# ViserMjcf — renders a MuJoCo model in Viser
# ---------------------------------------------------------------------------


class ViserMjcf:
    """Manages Viser scene handles for a MuJoCo model.

    Creates mesh handles for all bodies under the given root frame,
    and provides methods to update transforms from MjData.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        mj_model: Any,
        mj_data: Any,
        root_name: str,
    ) -> None:
        self._server = server
        self._mj_model = mj_model
        self._mj_data = mj_data
        self._root_name = root_name

        self._dynamic_handles: dict[int, viser.GlbHandle] = {}
        self._fixed_handles: dict[int, viser.GlbHandle] = {}
        self._collision_dynamic: dict[int, viser.GlbHandle] = {}
        self._collision_fixed: dict[int, viser.GlbHandle] = {}
        self._has_collision_geoms: bool = False
        self._body_ids: list[int] = []
        self._body_id_to_name: dict[int, str] = {}

    def create_visual_handles(self) -> None:
        """Create mesh handles for all bodies in the model."""
        mj_model = self._mj_model

        mj_model = self._mj_model
        for body_id in range(mj_model.nbody):
            name = _body_name(mj_model, body_id)
            self._body_ids.append(body_id)
            self._body_id_to_name[body_id] = name

            pos = tuple(self._mj_data.xpos[body_id])
            mat = self._mj_data.xmat[body_id].reshape(3, 3)
            wxyz = tuple(vtf.SO3.from_matrix(mat).wxyz)

            visual_mesh = _body_mesh(
                mj_model, body_id, geom_filter=lambda gid: int(mj_model.geom_group[gid]) < 3
            )
            collision_mesh = _body_mesh(
                mj_model, body_id, geom_filter=lambda gid: int(mj_model.geom_group[gid]) >= 3
            )

            safe = name.replace("/", "_")

            if visual_mesh is not None:
                handle = self._server.scene.add_mesh_trimesh(
                    f"{self._root_name}/{safe}",
                    visual_mesh,
                    position=pos,
                    wxyz=wxyz,
                    cast_shadow=True,
                    receive_shadow=True,
                )
                if _is_fixed_body(mj_model, body_id):
                    self._fixed_handles[body_id] = handle
                else:
                    self._dynamic_handles[body_id] = handle

            if collision_mesh is not None:
                self._has_collision_geoms = True
                _tint_mesh(collision_mesh, _COLLISION_TINT)
                coll_handle = self._server.scene.add_mesh_trimesh(
                    f"{self._root_name}/{safe}_collision",
                    collision_mesh,
                    position=pos,
                    wxyz=wxyz,
                    visible=False,
                    cast_shadow=False,
                    receive_shadow=False,
                )
                if _is_fixed_body(mj_model, body_id):
                    self._collision_fixed[body_id] = coll_handle
                else:
                    self._collision_dynamic[body_id] = coll_handle

    def update_from_mjdata(self, mj_data: Any | None = None) -> None:
        """Update transforms of all dynamic bodies from mjdata."""
        data = mj_data if mj_data is not None else self._mj_data
        for body_id, handle in self._dynamic_handles.items():
            pos = tuple(data.xpos[body_id])
            mat = data.xmat[body_id].reshape(3, 3)
            wxyz = tuple(vtf.SO3.from_matrix(mat).wxyz)
            handle.position = pos
            handle.wxyz = wxyz
        for body_id, handle in self._collision_dynamic.items():
            pos = tuple(data.xpos[body_id])
            mat = data.xmat[body_id].reshape(3, 3)
            wxyz = tuple(vtf.SO3.from_matrix(mat).wxyz)
            handle.position = pos
            handle.wxyz = wxyz

    def get_joint_limits(self) -> list[tuple[str, float, float, int]]:
        """Return [(name, lower, upper, qpos_adr), ...] for controllable joints.

        Prefers actuated joints. Falls back to all non-free joints
        (hinge/slide) when there are no actuators.
        """
        mj_model = self._mj_model
        results: list[tuple[str, float, float, int]] = []

        if mj_model.nu > 0:
            for act_idx in range(mj_model.nu):
                trntype = int(mj_model.actuator_trntype[act_idx])
                if trntype not in (0, 1):
                    continue
                jnt_id = int(mj_model.actuator_trnid[act_idx, 0])
                if jnt_id < 0 or jnt_id >= mj_model.njnt:
                    continue
                name = mj_id2name(mj_model, mjtObj.mjOBJ_JOINT, jnt_id)
                if not name:
                    name = f"joint_{jnt_id}"
                lower = float(mj_model.jnt_range[jnt_id, 0])
                upper = float(mj_model.jnt_range[jnt_id, 1])
                qpos_adr = int(mj_model.jnt_qposadr[jnt_id])
                results.append((name, lower, upper, qpos_adr))
        else:
            for jnt_id in range(mj_model.njnt):
                jtype = int(mj_model.jnt_type[jnt_id])
                if jtype == 0:
                    continue
                name = mj_id2name(mj_model, mjtObj.mjOBJ_JOINT, jnt_id)
                if not name:
                    name = f"joint_{jnt_id}"
                lower = float(mj_model.jnt_range[jnt_id, 0])
                upper = float(mj_model.jnt_range[jnt_id, 1])
                qpos_adr = int(mj_model.jnt_qposadr[jnt_id])
                results.append((name, lower, upper, qpos_adr))

        return results

    def get_joint_values(self, qpos_adrs: list[int]) -> np.ndarray:
        """Read current joint values for the given qpos addresses."""
        return np.array([float(self._mj_data.qpos[adr]) for adr in qpos_adrs])

    def set_joint_values(self, values: np.ndarray, qpos_adrs: list[int]) -> None:
        """Set joint values, run forward kinematics, and update scene."""
        for v, adr in zip(values, qpos_adrs):
            self._mj_data.qpos[adr] = v
        mujoco.mj_forward(self._mj_model, self._mj_data)
        self.update_from_mjdata()

    def get_body_names(self) -> list[str]:
        """Return all body names for transform display."""
        return [self._body_id_to_name[bid] for bid in self._body_ids]

    def get_body_transform(self, name: str) -> np.ndarray | None:
        """Return 4x4 world transform for a body, or None if not found."""
        for bid, bname in self._body_id_to_name.items():
            if bname == name:
                pos = self._mj_data.xpos[bid]
                mat = self._mj_data.xmat[bid].reshape(3, 3)
                t = np.eye(4)
                t[:3, :3] = mat
                t[:3, 3] = pos
                return t
        return None

    def remove(self) -> None:
        """Remove all scene handles."""
        for handle in list(self._dynamic_handles.values()):
            try:
                handle.remove()
            except Exception:
                pass
        self._dynamic_handles.clear()
        for handle in list(self._fixed_handles.values()):
            try:
                handle.remove()
            except Exception:
                pass
        self._fixed_handles.clear()
        for handle in list(self._collision_dynamic.values()):
            try:
                handle.remove()
            except Exception:
                pass
        self._collision_dynamic.clear()
        for handle in list(self._collision_fixed.values()):
            try:
                handle.remove()
            except Exception:
                pass
        self._collision_fixed.clear()

    def set_collision_visible(self, visible: bool) -> None:
        """Show or hide collision meshes."""
        for handle in list(self._collision_dynamic.values()) + list(
            self._collision_fixed.values()
        ):
            handle.visible = visible


# ---------------------------------------------------------------------------
# Mink IK setup and step
# ---------------------------------------------------------------------------


def setup_mink_ik(
    server: viser.ViserServer,
    robot: RobotInstance,
    mj_model: Any,
    status_text: Any,
    cartesian_mode_checkbox: viser.GuiInputHandle[bool],
) -> None:
    """Set up mink IK for a MuJoCo robot — creates GUI elements and callbacks."""
    if not _MINK_AVAILABLE:
        cartesian_mode_checkbox.disabled = True
        server.gui.add_text("Cartesian IK", "mink is not installed.")
        return

    try:
        configuration = mink.Configuration(mj_model)
        robot.mink_configuration = configuration
        robot.mink_solver = (
            "daqp" if "daqp" in qpsolvers.available_solvers
            else qpsolvers.available_solvers[0]
        )

        pin_frame_names: set[str] = set()
        if hasattr(mj_model, "frames"):
            for frame in mj_model.frames:
                name = mj_id2name(mj_model, mjtObj.mjOBJ_BODY, int(frame.bodyid))
                if name:
                    pin_frame_names.add(str(frame.name))

        frame_options = sorted(pin_frame_names)
        if not frame_options:
            frame_options = [
                _body_name(mj_model, i) for i in range(mj_model.nbody)
            ]
        if not frame_options:
            raise RuntimeError("No valid frames found for Cartesian target")

        cartesian_frame_dropdown = server.gui.add_dropdown(
            "Target frame",
            options=frame_options,
            initial_value=frame_options[-1],
        )
        robot.cartesian_frame_dropdown = cartesian_frame_dropdown

        cartesian_target_handle = server.scene.add_transform_controls(
            f"{robot.root_name}/cartesian_target",
            scale=0.25,
            line_width=3.0,
            visible=False,
        )
        robot.cartesian_target_handle = cartesian_target_handle

        advanced_folder = server.gui.add_folder("Advanced", expand_by_default=False)
        with advanced_folder:
            frame_task_position_cost_slider = server.gui.add_slider(
                "Position cost",
                min=0.0,
                max=10.0,
                initial_value=robot.ik_frame_position_cost,
                step=0.1,
            )
            frame_task_orientation_cost_slider = server.gui.add_slider(
                "Orientation cost",
                min=0.0,
                max=10.0,
                initial_value=robot.ik_frame_orientation_cost,
                step=0.1,
            )
            posture_cost_slider = server.gui.add_slider(
                "Posture cost",
                min=0.0,
                max=1.0,
                initial_value=robot.ik_posture_cost,
                step=0.001,
            )

        def _set_joint_controls_enabled(enabled: bool) -> None:
            if robot.slider_handles is not None:
                for slider in robot.slider_handles:
                    slider.disabled = not enabled
            if robot.randomize_button is not None:
                robot.randomize_button.disabled = not enabled
            if robot.reset_button is not None:
                robot.reset_button.disabled = not enabled

        def _sync_configuration_from_sliders() -> None:
            if (
                robot.mink_configuration is None
                or robot.slider_handles is None
                or robot.qpos_adrs is None
            ):
                return
            q = np.array(robot.mink_configuration.q, copy=True)
            for adr, slider in zip(robot.qpos_adrs, robot.slider_handles):
                q[adr] = slider.value
            robot.mink_configuration.update(q)

        def _set_ik_tasks_from_current_configuration(frame_name: str) -> None:
            if robot.mink_configuration is None:
                return

            with robot.ik_lock:
                frame_task = mink.FrameTask(
                    frame_name,
                    frame_type="body",
                    position_cost=robot.ik_frame_position_cost,
                    orientation_cost=robot.ik_frame_orientation_cost,
                )
                posture_task = mink.PostureTask(mj_model, cost=robot.ik_posture_cost)
                posture_task.set_target_from_configuration(
                    robot.mink_configuration
                )
                config_limit = mink.ConfigurationLimit(mj_model)

                T_world = robot.mink_configuration.get_transform_frame_to_world(
                    frame_name, frame_type="body"
                )
                T_init = SE3.from_matrix(T_world.as_matrix())
                frame_task.set_target(T_init)

                robot.mink_tasks = [frame_task, posture_task]
                robot.mink_limits = [config_limit]

            if robot.cartesian_target_handle is not None:
                robot.cartesian_target_handle.position = tuple(
                    T_world.translation()
                )
                robot.cartesian_target_handle.wxyz = rotation_matrix_to_wxyz(
                    T_world.rotation().as_matrix()
                )

        @frame_task_position_cost_slider.on_update
        def _on_position_cost(_event: object) -> None:
            robot.ik_frame_position_cost = frame_task_position_cost_slider.value

        @frame_task_orientation_cost_slider.on_update
        def _on_orientation_cost(_event: object) -> None:
            robot.ik_frame_orientation_cost = frame_task_orientation_cost_slider.value

        @posture_cost_slider.on_update
        def _on_posture_cost(_event: object) -> None:
            robot.ik_posture_cost = posture_cost_slider.value

        @cartesian_target_handle.on_update
        def _on_target_update(event: viser.TransformControlsEvent) -> None:
            frame_task = robot.mink_tasks[0] if robot.mink_tasks else None
            if frame_task is None:
                return
            new_target = SE3.from_rotation_and_translation(
                SO3.from_matrix(
                    wxyz_to_rotation_matrix(
                        (
                            float(event.target.wxyz[0]),
                            float(event.target.wxyz[1]),
                            float(event.target.wxyz[2]),
                            float(event.target.wxyz[3]),
                        )
                    )
                ),
                np.array(event.target.position),
            )
            frame_task.set_target(new_target)

        @cartesian_frame_dropdown.on_update
        def _on_frame_change(_event: object) -> None:
            if robot.cartesian_frame_dropdown is None:
                return
            _set_ik_tasks_from_current_configuration(
                robot.cartesian_frame_dropdown.value
            )

        @cartesian_mode_checkbox.on_update
        def _on_cartesian_mode_change(_event: object) -> None:
            enabled = cartesian_mode_checkbox.value

            if not enabled:
                robot.ik_enabled = False
                _set_joint_controls_enabled(True)
                if robot.cartesian_target_handle is not None:
                    robot.cartesian_target_handle.visible = False
                return

            _set_joint_controls_enabled(False)
            if robot.cartesian_target_handle is not None:
                robot.cartesian_target_handle.visible = True

            if robot.cartesian_frame_dropdown is not None:
                _sync_configuration_from_sliders()
                _set_ik_tasks_from_current_configuration(
                    robot.cartesian_frame_dropdown.value
                )

            robot.ik_enabled = True

    except Exception as ik_exc:
        cartesian_mode_checkbox.disabled = True
        server.gui.add_text("Cartesian IK", f"Unavailable: {ik_exc!r}")


def mink_ik_step(robot: RobotInstance, dt: float) -> np.ndarray | None:
    """Run one mink IK step. Returns updated joint values or None on error."""
    if not _MINK_AVAILABLE:
        return None

    try:
        with robot.ik_lock:
            velocity = mink.solve_ik(
                robot.mink_configuration,
                robot.mink_tasks,
                dt,
                solver=robot.mink_solver,
                limits=robot.mink_limits,
            )
            robot.mink_configuration.integrate_inplace(velocity, dt)
            return np.array(robot.mink_configuration.q, copy=True)
    except Exception:
        robot.ik_enabled = False
        if robot.cartesian_mode_checkbox is not None:
            robot.cartesian_mode_checkbox.value = False
        return None
