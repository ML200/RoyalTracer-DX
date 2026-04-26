"""Build an in-memory scene graph from a parsed PBRT v4 statement stream.

Maintains the CTM and graphics-state stacks, resolves Include/Import,
collects shapes, materials, textures, lights, the camera, and named
object templates for instancing.
"""

import math
import os
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pbrt_parser import Param, Statement, parse_file


# -------------------------------------------------------------------------
# Transform helpers
# -------------------------------------------------------------------------

def _identity():
    return np.eye(4, dtype=np.float64)


def _translate(x, y, z):
    M = np.eye(4)
    M[:3, 3] = (x, y, z)
    return M


def _scale(x, y, z):
    M = np.eye(4)
    M[0, 0], M[1, 1], M[2, 2] = x, y, z
    return M


def _rotate(angle_deg, x, y, z):
    a = math.radians(angle_deg)
    n = math.sqrt(x * x + y * y + z * z)
    if n == 0:
        return np.eye(4)
    x, y, z = x / n, y / n, z / n
    c, s = math.cos(a), math.sin(a)
    C = 1 - c
    M = np.eye(4)
    M[0, 0] = x * x * C + c
    M[0, 1] = x * y * C - z * s
    M[0, 2] = x * z * C + y * s
    M[1, 0] = y * x * C + z * s
    M[1, 1] = y * y * C + c
    M[1, 2] = y * z * C - x * s
    M[2, 0] = z * x * C - y * s
    M[2, 1] = z * y * C + x * s
    M[2, 2] = z * z * C + c
    return M


def _lookat(ex, ey, ez, lx, ly, lz, ux, uy, uz):
    """gluLookAt-style world-to-camera matrix (PBRT convention)."""
    eye = np.array([ex, ey, ez], dtype=np.float64)
    look = np.array([lx, ly, lz], dtype=np.float64)
    up = np.array([ux, uy, uz], dtype=np.float64)
    f = look - eye
    fn = np.linalg.norm(f)
    if fn == 0:
        return np.eye(4)
    f /= fn
    un = np.linalg.norm(up)
    if un == 0:
        up_n = np.array([0.0, 1.0, 0.0])
    else:
        up_n = up / un
    s = np.cross(f, up_n)
    sn = np.linalg.norm(s)
    if sn < 1e-12:
        # f is parallel to up; pick an arbitrary side
        s = np.array([1.0, 0.0, 0.0])
    else:
        s /= sn
    u = np.cross(s, f)
    M = np.eye(4)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M


def _from_16(vals):
    """PBRT Transform/ConcatTransform: 16 numbers in row-major order."""
    return np.array(vals, dtype=np.float64).reshape(4, 4)


# -------------------------------------------------------------------------
# Graphics state
# -------------------------------------------------------------------------

@dataclass
class Material:
    """An in-line or named material definition.

    `kind` is the PBRT material type: 'diffuse', 'coateddiffuse', 'conductor',
    'dielectric', 'diffusetransmission', 'thindielectric', 'mix', etc.
    """
    kind: str
    params: Dict[str, Param] = field(default_factory=dict)
    name: Optional[str] = None  # set when defined via MakeNamedMaterial


@dataclass
class GraphicsState:
    ctm: np.ndarray = field(default_factory=_identity)
    material: Optional[Material] = None
    area_light: Optional[Tuple[str, Dict[str, Param]]] = None
    reverse_orientation: bool = False
    medium_inner: str = ""
    medium_outer: str = ""

    def copy(self):
        return GraphicsState(
            ctm=self.ctm.copy(),
            material=self.material,
            area_light=self.area_light,
            reverse_orientation=self.reverse_orientation,
            medium_inner=self.medium_inner,
            medium_outer=self.medium_outer,
        )


# -------------------------------------------------------------------------
# Scene types
# -------------------------------------------------------------------------

@dataclass
class ShapeEntry:
    """A single shape instance ready to be exported.

    For an actual geometric shape (Shape directive) `kind=='shape'`, with
    `shape_type`/`shape_params`. For an object instance, `kind=='instance'`
    and `instance_name` is the ObjectBegin name.
    """
    kind: str  # 'shape' or 'instance'
    ctm: np.ndarray = field(default_factory=_identity)
    material: Optional[Material] = None
    area_light: Optional[Tuple[str, Dict[str, Param]]] = None
    reverse_orientation: bool = False
    shape_type: str = ""
    shape_params: Dict[str, Param] = field(default_factory=dict)
    instance_name: str = ""


@dataclass
class TextureDef:
    name: str
    tclass_value: str  # 'spectrum', 'float', etc.
    tclass: str        # 'imagemap', 'constant', 'scale', 'mix', ...
    params: Dict[str, Param] = field(default_factory=dict)


@dataclass
class LightDef:
    kind: str
    params: Dict[str, Param] = field(default_factory=dict)
    ctm: np.ndarray = field(default_factory=_identity)


@dataclass
class CameraDef:
    kind: str
    params: Dict[str, Param] = field(default_factory=dict)
    world_to_camera: np.ndarray = field(default_factory=_identity)
    film_width: int = 1280
    film_height: int = 720


@dataclass
class Scene:
    base_dir: Path = field(default_factory=Path)
    shapes: List[ShapeEntry] = field(default_factory=list)
    objects: Dict[str, List[ShapeEntry]] = field(default_factory=dict)
    materials_by_name: Dict[str, Material] = field(default_factory=dict)
    textures: Dict[str, TextureDef] = field(default_factory=dict)
    lights: List[LightDef] = field(default_factory=list)
    camera: Optional[CameraDef] = None
    film_width: int = 1280
    film_height: int = 720


# -------------------------------------------------------------------------
# Builder
# -------------------------------------------------------------------------

class SceneBuilder:
    def __init__(self, root_path: Path):
        self.scene = Scene(base_dir=root_path.parent)
        self.root_path = root_path

        self.gs = GraphicsState()
        self.stack: List[GraphicsState] = []
        self.in_world = False
        self.world_ctm_at_world_begin: Optional[np.ndarray] = None

        # Object templates for instancing
        self.current_object: Optional[str] = None

        # Pending film parameters (captured before WorldBegin)
        self.film_width = 1280
        self.film_height = 720
        self.pending_camera: Optional[Tuple[str, Dict[str, Param], np.ndarray]] = None

        # Set of files already included to break cycles.
        self._included_files: set = set()

    # ---- public ----------------------------------------------------------

    def build(self, statements: List[Statement]) -> Scene:
        self._exec_block(statements, self.scene.base_dir)
        # After parsing, finalize the camera using the recorded CTM.
        if self.pending_camera is not None:
            kind, params, w2c = self.pending_camera
            self.scene.camera = CameraDef(
                kind=kind, params=params, world_to_camera=w2c,
                film_width=self.film_width, film_height=self.film_height)
        self.scene.film_width = self.film_width
        self.scene.film_height = self.film_height
        return self.scene

    # ---- internal --------------------------------------------------------

    def _exec_block(self, statements, base_dir: Path):
        for stmt in statements:
            self._exec(stmt, base_dir)

    def _exec(self, stmt: Statement, base_dir: Path):
        n = stmt.name

        # --- Transform stack ---------------------------------------------
        if n == "Identity":
            self.gs.ctm = _identity()
        elif n == "Translate":
            self.gs.ctm = self.gs.ctm @ _translate(*stmt.args)
        elif n == "Scale":
            self.gs.ctm = self.gs.ctm @ _scale(*stmt.args)
        elif n == "Rotate":
            self.gs.ctm = self.gs.ctm @ _rotate(*stmt.args)
        elif n == "LookAt":
            self.gs.ctm = self.gs.ctm @ _lookat(*stmt.args)
        elif n == "Transform":
            self.gs.ctm = _from_16(stmt.args)
        elif n == "ConcatTransform":
            self.gs.ctm = self.gs.ctm @ _from_16(stmt.args)
        elif n == "CoordinateSystem":
            pass  # named coord systems not tracked
        elif n == "CoordSysTransform":
            pass

        # --- Attribute / world stack -------------------------------------
        elif n == "AttributeBegin":
            self.stack.append(self.gs.copy())
        elif n == "AttributeEnd":
            if self.stack:
                self.gs = self.stack.pop()
        elif n == "WorldBegin":
            self.in_world = True
            # Capture the world-to-camera transform at WorldBegin time
            self.world_ctm_at_world_begin = self.gs.ctm.copy()
            if self.pending_camera is not None:
                kind, params, _ = self.pending_camera
                self.pending_camera = (kind, params, self.world_ctm_at_world_begin.copy())
            # Reset CTM for the world
            self.gs.ctm = _identity()
            self.gs.material = None
            self.gs.area_light = None
            self.gs.reverse_orientation = False

        # --- Scene-wide options ------------------------------------------
        elif n == "Camera":
            kind = stmt.args[0]
            self.pending_camera = (kind, stmt.params, self.gs.ctm.copy())
        elif n == "Film":
            params = stmt.params
            if "xresolution" in params:
                self.film_width = int(params["xresolution"].values[0])
            if "yresolution" in params:
                self.film_height = int(params["yresolution"].values[0])
        elif n in ("Sampler", "Integrator", "Filter", "PixelFilter",
                   "Accelerator", "ColorSpace", "Option",
                   "MakeNamedMedium", "ActiveTransform", "TransformTimes",
                   "Sides"):
            pass  # ignored for export

        # --- Materials / textures ----------------------------------------
        elif n == "Material":
            kind = stmt.args[0]
            self.gs.material = Material(kind=kind, params=stmt.params)
        elif n == "MakeNamedMaterial":
            mat_name = stmt.args[0]
            kind_param = stmt.params.get("type")
            kind = kind_param.values[0] if kind_param else "diffuse"
            mat = Material(kind=kind, params=stmt.params, name=mat_name)
            self.scene.materials_by_name[mat_name] = mat
        elif n == "NamedMaterial":
            mat_name = stmt.args[0]
            mat = self.scene.materials_by_name.get(mat_name)
            if mat is None:
                # forward reference; create a placeholder we can resolve later
                mat = Material(kind="diffuse", name=mat_name)
                self.scene.materials_by_name[mat_name] = mat
            self.gs.material = mat
        elif n == "Texture":
            tname, ttype, tclass = stmt.args
            self.scene.textures[tname] = TextureDef(
                name=tname, tclass_value=ttype, tclass=tclass, params=stmt.params)

        # --- Lights ------------------------------------------------------
        elif n == "AreaLightSource":
            self.gs.area_light = (stmt.args[0], stmt.params)
        elif n == "LightSource":
            self.scene.lights.append(LightDef(
                kind=stmt.args[0], params=stmt.params, ctm=self.gs.ctm.copy()))

        # --- Object instancing ------------------------------------------
        elif n == "ObjectBegin":
            self.current_object = stmt.args[0]
            self.scene.objects.setdefault(self.current_object, [])
        elif n == "ObjectEnd":
            self.current_object = None
        elif n == "ObjectInstance":
            entry = ShapeEntry(
                kind="instance", ctm=self.gs.ctm.copy(),
                instance_name=stmt.args[0])
            self.scene.shapes.append(entry)

        # --- Shapes ------------------------------------------------------
        elif n == "Shape":
            entry = ShapeEntry(
                kind="shape",
                ctm=self.gs.ctm.copy(),
                material=self.gs.material,
                area_light=self.gs.area_light,
                reverse_orientation=self.gs.reverse_orientation,
                shape_type=stmt.args[0],
                shape_params=stmt.params,
            )
            if self.current_object is not None:
                self.scene.objects[self.current_object].append(entry)
            else:
                self.scene.shapes.append(entry)

        elif n == "ReverseOrientation":
            self.gs.reverse_orientation = not self.gs.reverse_orientation
        elif n == "MediumInterface":
            self.gs.medium_inner, self.gs.medium_outer = stmt.args[0], stmt.args[1]

        # --- Includes ----------------------------------------------------
        elif n in ("Include", "Import"):
            sub_path = (Path(base_dir) / stmt.args[0]).resolve()
            if str(sub_path) in self._included_files:
                return
            self._included_files.add(str(sub_path))
            try:
                sub_stmts = parse_file(sub_path)
            except FileNotFoundError:
                print(f"  warning: include not found: {sub_path}")
                return
            self._exec_block(sub_stmts, sub_path.parent)

        # --- Unknown / skipped -------------------------------------------
        else:
            # Unknown directives are no-ops here; the parser has already
            # consumed their tokens.
            pass


def build_scene(path) -> Scene:
    path = Path(path).resolve()
    statements = parse_file(path)
    return SceneBuilder(path).build(statements)
