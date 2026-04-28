"""Export an in-memory PBRT Scene to a binary glTF 2.0 (.glb) file."""

import io
import math
import os
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pygltflib import (
    Accessor, Asset, Attributes, Buffer, BufferView, Camera, GLTF2, Image,
    Material, Mesh, Node, NormalMaterialTexture, PbrMetallicRoughness,
    Perspective, Primitive, Sampler, Scene as GLTFScene, Texture, TextureInfo,
)

from pbrt_parser import Param
from pbrt_scene import (
    CameraDef, LightDef, Material as PMaterial, Scene, ShapeEntry, TextureDef,
)


# glTF constants
COMP_UBYTE  = 5121
COMP_USHORT = 5123
COMP_UINT   = 5125
COMP_FLOAT  = 5126

TARGET_ARRAY = 34962
TARGET_ELEM  = 34963

WRAP_REPEAT = 10497
WRAP_CLAMP  = 33071
FILTER_LINEAR = 9729
FILTER_LINEAR_MIPMAP = 9987


# -------------------------------------------------------------------------
# Helpers for parameter extraction
# -------------------------------------------------------------------------

def _floats(p: Optional[Param], default=None):
    if p is None:
        return default
    return [float(v) for v in p.values]


def _first_str(p: Optional[Param]) -> Optional[str]:
    if p is None or not p.values:
        return None
    v = p.values[0]
    return v if isinstance(v, str) else None


def _is_twosided(light_params, pmat) -> bool:
    """Return True if the source explicitly marks the surface two-sided.

    PBRT v4: `AreaLightSource ... "bool twosided" true`
    Mitsuba: `<bsdf type="twosided">` wrapper (parser tags `_twosided`)
    """
    p = light_params.get("twosided") if light_params else None
    if p is not None and p.values:
        v = p.values[0]
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() == "true"
        try:
            return bool(int(v))
        except (TypeError, ValueError):
            pass
    if pmat is not None and "_twosided" in pmat.params:
        return True
    return False


def _blackbody_to_rgb(kelvin: float):
    """Approximate sRGB (linear) color of a blackbody emitter (Tanner Helland)."""
    t = max(1000.0, min(40000.0, kelvin)) / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * math.log(t) - 161.1195681661
        b = 0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307
    else:
        r = 329.698727446 * (t - 60) ** -0.1332047592
        g = 288.1221695283 * (t - 60) ** -0.0755148492
        b = 255.0
    r = max(0.0, min(255.0, r)) / 255.0
    g = max(0.0, min(255.0, g)) / 255.0
    b = max(0.0, min(255.0, b)) / 255.0
    return (r, g, b)


# -------------------------------------------------------------------------
# Mesh data extraction
# -------------------------------------------------------------------------

def _mesh_from_trianglemesh(params: Dict[str, Param]):
    P = _floats(params.get("P"))
    if P is None:
        return None
    indices = _floats(params.get("indices"))
    if indices is None:
        return None
    N = _floats(params.get("N"))
    uv = _floats(params.get("uv"))
    if uv is None:
        uv = _floats(params.get("st"))

    positions = np.array(P, dtype=np.float32).reshape(-1, 3)
    idx = np.array(indices, dtype=np.uint32).reshape(-1, 3)
    normals = (np.array(N, dtype=np.float32).reshape(-1, 3)
               if N and len(N) == positions.size else None)
    uvs = (np.array(uv, dtype=np.float32).reshape(-1, 2)
           if uv and len(uv) // 2 == len(positions) else None)
    return positions, idx, normals, uvs


def _mesh_from_plymesh(params: Dict[str, Param], base_dir: Path):
    fname = _first_str(params.get("filename"))
    if fname is None:
        return None
    path = (base_dir / fname).resolve()
    if not path.exists():
        print(f"  warning: PLY file not found: {path}")
        return None

    try:
        from plyfile import PlyData
    except ImportError:
        print("  error: plyfile not installed; install with `pip install plyfile`.")
        return None

    ply = PlyData.read(str(path))
    verts = ply["vertex"].data
    # Case-insensitive lookup: lowered name -> original property name.
    name_map = {n.lower(): n for n in verts.dtype.names}

    def _stack(*candidates):
        """Return float32 columns from the first candidate tuple all present
        in the vertex element (case-insensitive); else None."""
        for cand in candidates:
            if all(c in name_map for c in cand):
                cols = [verts[name_map[c]] for c in cand]
                return np.stack(cols, axis=1).astype(np.float32)
        return None

    positions = _stack(("x", "y", "z"))

    normals = _stack(
        ("nx", "ny", "nz"),
        ("normal_x", "normal_y", "normal_z"),
    )

    uvs = _stack(
        ("u", "v"),
        ("s", "t"),
        ("texture_u", "texture_v"),
        ("texture_s", "texture_t"),
    )

    if "face" in [el.name for el in ply.elements]:
        face_data = ply["face"].data
        face_field = "vertex_indices" if "vertex_indices" in face_data.dtype.names else "vertex_index"
        idx_list = []
        for f in face_data:
            poly = f[face_field]
            if len(poly) < 3:
                continue
            for i in range(1, len(poly) - 1):
                idx_list.extend([poly[0], poly[i], poly[i + 1]])
        idx = np.array(idx_list, dtype=np.uint32).reshape(-1, 3)
    else:
        idx = np.arange(len(positions), dtype=np.uint32).reshape(-1, 3)

    return positions, idx, normals, uvs


def _mesh_from_objmesh(params: Dict[str, Param], base_dir: Path):
    """Load a Wavefront OBJ file referenced by `filename` into the
    parallel-array form the rest of the exporter expects.

    OBJ allows separate streams of v/vt/vn that are recombined per face
    vertex; we deduplicate (vi, ti, ni) tuples into a single combined
    index so positions/normals/uvs all share the same vertex index space.
    """
    fname = _first_str(params.get("filename"))
    if fname is None:
        return None
    path = (base_dir / fname).resolve()
    if not path.exists():
        print(f"  warning: OBJ file not found: {path}")
        return None

    raw_v: List[Tuple[float, float, float]] = []
    raw_vt: List[Tuple[float, float]] = []
    raw_vn: List[Tuple[float, float, float]] = []
    combined: Dict[Tuple[int, int, int], int] = {}
    out_pos: List[Tuple[float, float, float]] = []
    out_uv: List[Tuple[float, float]] = []
    out_n: List[Tuple[float, float, float]] = []
    out_tris: List[Tuple[int, int, int]] = []
    saw_uv = False
    saw_n = False

    def get_combined(vi: int, ti: int, ni: int) -> int:
        key = (vi, ti, ni)
        idx = combined.get(key)
        if idx is not None:
            return idx
        idx = len(out_pos)
        combined[key] = idx
        out_pos.append(raw_v[vi] if 0 <= vi < len(raw_v) else (0.0, 0.0, 0.0))
        out_uv.append(raw_vt[ti] if 0 <= ti < len(raw_vt) else (0.0, 0.0))
        out_n.append(raw_vn[ni] if 0 <= ni < len(raw_vn) else (0.0, 0.0, 0.0))
        return idx

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                tag = parts[0]
                if tag == "v" and len(parts) >= 4:
                    raw_v.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif tag == "vt" and len(parts) >= 2:
                    u = float(parts[1])
                    v = float(parts[2]) if len(parts) >= 3 else 0.0
                    raw_vt.append((u, v))
                    saw_uv = True
                elif tag == "vn" and len(parts) >= 4:
                    raw_vn.append((float(parts[1]), float(parts[2]), float(parts[3])))
                    saw_n = True
                elif tag == "f" and len(parts) >= 4:
                    face_idx = []
                    for vstr in parts[1:]:
                        sp = vstr.split("/")
                        try:
                            vi = int(sp[0]) - 1 if sp[0] else -1
                        except ValueError:
                            vi = -1
                        ti = -1
                        ni = -1
                        if len(sp) > 1 and sp[1]:
                            try:
                                ti = int(sp[1]) - 1
                            except ValueError:
                                pass
                        if len(sp) > 2 and sp[2]:
                            try:
                                ni = int(sp[2]) - 1
                            except ValueError:
                                pass
                        # OBJ allows negative indices for "relative to end"
                        if vi < 0 and vi != -1:
                            vi = len(raw_v) + vi + 1
                        if ti < 0 and ti != -1:
                            ti = len(raw_vt) + ti + 1
                        if ni < 0 and ni != -1:
                            ni = len(raw_vn) + ni + 1
                        face_idx.append(get_combined(vi, ti, ni))
                    # Fan-triangulate
                    for i in range(1, len(face_idx) - 1):
                        out_tris.append((face_idx[0], face_idx[i], face_idx[i + 1]))
                # ignore o/g/s/usemtl/mtllib
    except OSError as e:
        print(f"  warning: could not read OBJ {path}: {e}")
        return None

    if not out_pos or not out_tris:
        return None

    positions = np.array(out_pos, dtype=np.float32)
    indices = np.array(out_tris, dtype=np.uint32)
    normals = np.array(out_n, dtype=np.float32) if saw_n else None
    uvs = np.array(out_uv, dtype=np.float32) if saw_uv else None
    return positions, indices, normals, uvs


def _mesh_from_sphere(params: Dict[str, Param]):
    """Tessellate a unit sphere centered at origin (radius from 'radius')."""
    radius = float(params["radius"].values[0]) if "radius" in params else 1.0
    stacks = 24
    slices = 48
    positions = []
    normals = []
    uvs = []
    for i in range(stacks + 1):
        v = i / stacks
        theta = v * math.pi
        for j in range(slices + 1):
            u = j / slices
            phi = u * 2.0 * math.pi
            x = math.sin(theta) * math.cos(phi)
            y = math.cos(theta)
            z = math.sin(theta) * math.sin(phi)
            positions.append((radius * x, radius * y, radius * z))
            normals.append((x, y, z))
            uvs.append((u, v))
    indices = []
    for i in range(stacks):
        for j in range(slices):
            a = i * (slices + 1) + j
            b = a + slices + 1
            # At the top pole (i==0) the (a, b, a+1) triangle collapses
            # because vertices a and a+1 share the pole position. Likewise
            # (a+1, b, b+1) collapses at the bottom pole (i==stacks-1).
            if i != 0:
                indices.append((a, b, a + 1))
            if i != stacks - 1:
                indices.append((a + 1, b, b + 1))
    return (np.array(positions, dtype=np.float32),
            np.array(indices, dtype=np.uint32),
            np.array(normals, dtype=np.float32),
            np.array(uvs, dtype=np.float32))


# -------------------------------------------------------------------------
# Tangent generation (glTF 2.0 TANGENT VEC4, .w = handedness)
# -------------------------------------------------------------------------

def _arbitrary_tangent(normals: np.ndarray) -> np.ndarray:
    """Pick a unit tangent perpendicular to each row in `normals`.

    Used for vertices where neighboring triangles couldn't contribute a
    well-defined tangent (degenerate UVs, isolated verts), so the VEC4
    we emit is still valid.
    """
    abs_n = np.abs(normals)
    smallest = np.argmin(abs_n, axis=1)
    axis = np.zeros_like(normals)
    axis[np.arange(len(normals)), smallest] = 1.0
    t = axis - normals * np.einsum("ij,ij->i", axis, normals)[:, None]
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return t


def _generate_tangents(positions: np.ndarray, indices: np.ndarray,
                       normals: np.ndarray, uvs: np.ndarray) -> np.ndarray:
    """Per-vertex glTF tangents (VEC4) with handedness in .w.

    Lengyel-style: accumulate per-triangle tangent/bitangent at each
    vertex, Gram-Schmidt against the vertex normal, then derive the
    handedness sign from the accumulated bitangent. This is what every
    glTF 2.0 viewer expects when TANGENT is supplied; renderers that
    would otherwise fall back to MikkTSpace will use these directly.
    """
    p0 = positions[indices[:, 0]]
    p1 = positions[indices[:, 1]]
    p2 = positions[indices[:, 2]]
    uv0 = uvs[indices[:, 0]]
    uv1 = uvs[indices[:, 1]]
    uv2 = uvs[indices[:, 2]]

    e1 = p1 - p0
    e2 = p2 - p0
    duv1 = uv1 - uv0
    duv2 = uv2 - uv0
    det = duv1[:, 0] * duv2[:, 1] - duv1[:, 1] * duv2[:, 0]
    safe = np.abs(det) > 1e-8
    inv_det = np.where(safe, 1.0 / np.where(safe, det, 1.0), 0.0)

    tri_t = (e1 * duv2[:, 1:2] - e2 * duv1[:, 1:2]) * inv_det[:, None]
    tri_b = (e2 * duv1[:, 0:1] - e1 * duv2[:, 0:1]) * inv_det[:, None]

    n_vert = len(positions)
    t_sum = np.zeros((n_vert, 3), dtype=np.float64)
    b_sum = np.zeros((n_vert, 3), dtype=np.float64)
    tri_t64 = tri_t.astype(np.float64)
    tri_b64 = tri_b.astype(np.float64)
    for k in range(3):
        np.add.at(t_sum, indices[:, k], tri_t64)
        np.add.at(b_sum, indices[:, k], tri_b64)

    n = normals.astype(np.float64)
    # Gram-Schmidt against the vertex normal so the tangent lies in the
    # tangent plane (consumers assume this when reconstructing TBN).
    n_dot_t = np.einsum("ij,ij->i", n, t_sum)
    t = t_sum - n * n_dot_t[:, None]
    t_len = np.linalg.norm(t, axis=1, keepdims=True)
    bad = (t_len[:, 0] < 1e-8)
    if bad.any():
        t[bad] = _arbitrary_tangent(n[bad])
        t_len[bad] = 1.0
    t = t / t_len

    # Sign convention (glTF): bitangent = cross(N, T) * tangent.w
    handed = np.where(
        np.einsum("ij,ij->i", np.cross(n, t), b_sum) < 0.0, -1.0, 1.0)

    out = np.empty((n_vert, 4), dtype=np.float32)
    out[:, :3] = t.astype(np.float32)
    out[:, 3] = handed.astype(np.float32)
    return out


# -------------------------------------------------------------------------
# GLTFExporter
# -------------------------------------------------------------------------

class GLTFExporter:
    def __init__(self, scene: Scene, base_dir: Path):
        self.scene = scene
        self.base_dir = base_dir

        self.g = GLTF2()
        self.g.asset = Asset(version="2.0", generator="pbrt2glb")
        self.g.extensionsUsed = []
        self.g.extensionsRequired = []
        self.bin = bytearray()

        # Caches
        self._tex_image_cache: Dict[Tuple, int] = {}   # (kind, key, ...) -> glTF image index
        self._tex_index_cache: Dict[Tuple, int] = {}   # (kind, key, ...) -> glTF texture index
        self._material_cache: Dict[Tuple, int] = {}    # (kind, params-tuple, light-tuple) -> material index
        self._template_meshes: Dict[str, List[int]] = {}  # template name -> list of node indices to reuse
        self._default_sampler_idx: Optional[int] = None
        self._default_material_idx: Optional[int] = None
        self._punctual_lights: List[dict] = []         # KHR_lights_punctual entries

    # ---------------------------------------------------------------
    # Top-level export
    # ---------------------------------------------------------------

    def export(self, output_path: Path):
        root_nodes: List[int] = []

        if self.scene.camera is not None:
            n = self._build_camera_node()
            if n is not None:
                root_nodes.append(n)

        for light in self.scene.lights:
            n = self._build_light_node(light)
            if n is not None:
                root_nodes.append(n)

        for shape in self.scene.shapes:
            n = self._build_shape_node(shape)
            if n is not None:
                root_nodes.append(n)

        if not root_nodes:
            print("  warning: no exportable content; writing empty scene")

        self.g.scenes = [GLTFScene(nodes=root_nodes)]
        self.g.scene = 0

        # Punctual lights extension
        if self._punctual_lights:
            if "KHR_lights_punctual" not in self.g.extensionsUsed:
                self.g.extensionsUsed.append("KHR_lights_punctual")
            self.g.extensions = self.g.extensions or {}
            self.g.extensions["KHR_lights_punctual"] = {"lights": self._punctual_lights}

        # Set up the binary buffer
        if self.bin:
            self.g.buffers = [Buffer(byteLength=len(self.bin))]
            self.g.set_binary_blob(bytes(self.bin))
        else:
            self.g.buffers = []

        self.g.save_binary(str(output_path))

    # ---------------------------------------------------------------
    # Buffer / accessor helpers
    # ---------------------------------------------------------------

    def _pad4(self):
        while len(self.bin) % 4 != 0:
            self.bin.append(0)

    def _add_buffer_view(self, data: bytes, target: Optional[int] = None) -> int:
        self._pad4()
        offset = len(self.bin)
        self.bin.extend(data)
        bv = BufferView(buffer=0, byteOffset=offset, byteLength=len(data))
        if target is not None:
            bv.target = target
        self.g.bufferViews.append(bv)
        return len(self.g.bufferViews) - 1

    def _add_accessor(self, bv_idx, count, comp_type, type_,
                      min_=None, max_=None) -> int:
        a = Accessor(
            bufferView=bv_idx, byteOffset=0, componentType=comp_type,
            count=count, type=type_)
        if min_ is not None:
            a.min = list(map(float, min_))
        if max_ is not None:
            a.max = list(map(float, max_))
        self.g.accessors.append(a)
        return len(self.g.accessors) - 1

    # ---------------------------------------------------------------
    # Mesh emission
    # ---------------------------------------------------------------

    def _emit_mesh(self, positions: np.ndarray, indices: np.ndarray,
                   normals: Optional[np.ndarray], uvs: Optional[np.ndarray],
                   material_idx: int, flip_winding: bool) -> int:
        # Ensure contiguous float32 / uint32
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        indices = np.ascontiguousarray(indices, dtype=np.uint32)
        if flip_winding:
            # PBRT's ReverseOrientation reverses both the geometric and the
            # shading normal. Flipping the winding flips the geometric one;
            # we also negate per-vertex normals so the shading frame stays
            # consistent for any spec-conformant glTF consumer.
            indices = indices[:, ::-1].copy()
            if normals is not None and len(normals) == len(positions):
                normals = -np.ascontiguousarray(normals, dtype=np.float32)
        idx_flat = indices.reshape(-1)

        bv_pos = self._add_buffer_view(positions.tobytes(), TARGET_ARRAY)
        acc_pos = self._add_accessor(
            bv_pos, len(positions), COMP_FLOAT, "VEC3",
            min_=positions.min(axis=0).tolist(),
            max_=positions.max(axis=0).tolist())

        attrs = Attributes(POSITION=acc_pos)

        normals_for_tangent: Optional[np.ndarray] = None
        if normals is not None and len(normals) == len(positions):
            normals = np.ascontiguousarray(normals, dtype=np.float32)
            # Normalize
            lens = np.linalg.norm(normals, axis=1, keepdims=True)
            lens[lens == 0] = 1.0
            normals = normals / lens
            bv_n = self._add_buffer_view(normals.tobytes(), TARGET_ARRAY)
            attrs.NORMAL = self._add_accessor(bv_n, len(normals), COMP_FLOAT, "VEC3")
            normals_for_tangent = normals

        uvs_for_tangent: Optional[np.ndarray] = None
        if uvs is not None and len(uvs) == len(positions):
            uvs = np.ascontiguousarray(uvs, dtype=np.float32)
            bv_uv = self._add_buffer_view(uvs.tobytes(), TARGET_ARRAY)
            attrs.TEXCOORD_0 = self._add_accessor(
                bv_uv, len(uvs), COMP_FLOAT, "VEC2")
            uvs_for_tangent = uvs

        # Per glTF 2.0: when TANGENT is supplied, consumers must use it
        # (avoids the per-triangle on-the-fly derivation that breaks at
        # degenerate or mirrored UVs). Skip when normals or UVs are
        # missing -- the spec says tangent only makes sense with both.
        if normals_for_tangent is not None and uvs_for_tangent is not None:
            tangents = _generate_tangents(
                positions, indices, normals_for_tangent, uvs_for_tangent)
            tangents = np.ascontiguousarray(tangents, dtype=np.float32)
            bv_t = self._add_buffer_view(tangents.tobytes(), TARGET_ARRAY)
            attrs.TANGENT = self._add_accessor(
                bv_t, len(tangents), COMP_FLOAT, "VEC4")

        bv_idx = self._add_buffer_view(idx_flat.tobytes(), TARGET_ELEM)
        acc_idx = self._add_accessor(bv_idx, len(idx_flat), COMP_UINT, "SCALAR")

        prim = Primitive(attributes=attrs, indices=acc_idx, material=material_idx)
        mesh = Mesh(primitives=[prim])
        self.g.meshes.append(mesh)
        return len(self.g.meshes) - 1

    # ---------------------------------------------------------------
    # Shape -> node
    # ---------------------------------------------------------------

    def _build_shape_node(self, shape: ShapeEntry) -> Optional[int]:
        if shape.kind == "instance":
            return self._build_instance_node(shape)

        mesh_data = self._extract_mesh(shape)
        if mesh_data is None:
            return None
        positions, indices, normals, uvs = mesh_data

        material_idx = self._convert_material(
            shape.material, shape.area_light, shape.shape_params)
        mesh_idx = self._emit_mesh(
            positions, indices, normals, uvs,
            material_idx=material_idx,
            flip_winding=shape.reverse_orientation)

        node = Node(mesh=mesh_idx, matrix=_to_gltf_matrix(shape.ctm))
        self.g.nodes.append(node)
        return len(self.g.nodes) - 1

    def _build_instance_node(self, shape: ShapeEntry) -> Optional[int]:
        template = self.scene.objects.get(shape.instance_name)
        if template is None or not template:
            return None

        # Make sure the template's meshes are emitted (cached by template name).
        if shape.instance_name not in self._template_meshes:
            child_proto = []  # list of (mesh_idx, ctm)
            for sub in template:
                if sub.kind != "shape":
                    continue
                mesh_data = self._extract_mesh(sub)
                if mesh_data is None:
                    continue
                positions, indices, normals, uvs = mesh_data
                material_idx = self._convert_material(
                    sub.material, sub.area_light, sub.shape_params)
                mesh_idx = self._emit_mesh(
                    positions, indices, normals, uvs,
                    material_idx=material_idx,
                    flip_winding=sub.reverse_orientation)
                child_proto.append((mesh_idx, sub.ctm.copy()))
            self._template_meshes[shape.instance_name] = child_proto

        proto = self._template_meshes[shape.instance_name]
        if not proto:
            return None

        # Build child nodes for this instance, sharing meshes across instances.
        child_indices = []
        for mesh_idx, sub_ctm in proto:
            child_node = Node(mesh=mesh_idx, matrix=_to_gltf_matrix(sub_ctm))
            self.g.nodes.append(child_node)
            child_indices.append(len(self.g.nodes) - 1)

        parent = Node(matrix=_to_gltf_matrix(shape.ctm), children=child_indices)
        self.g.nodes.append(parent)
        return len(self.g.nodes) - 1

    def _extract_mesh(self, shape: ShapeEntry):
        st = (shape.shape_type or "").lower()
        if st == "trianglemesh":
            return _mesh_from_trianglemesh(shape.shape_params)
        if st == "plymesh":
            return _mesh_from_plymesh(shape.shape_params, self.base_dir)
        if st == "objmesh":
            return _mesh_from_objmesh(shape.shape_params, self.base_dir)
        if st == "sphere":
            return _mesh_from_sphere(shape.shape_params)
        # bilinearmesh, curve, disk, cylinder etc. unsupported for now
        print(f"  warning: shape '{shape.shape_type}' not supported, skipped")
        return None

    # ---------------------------------------------------------------
    # Materials
    # ---------------------------------------------------------------

    def _ensure_default_material(self) -> int:
        if self._default_material_idx is None:
            mat = Material(
                name="default",
                pbrMetallicRoughness=PbrMetallicRoughness(
                    baseColorFactor=[0.8, 0.8, 0.8, 1.0],
                    metallicFactor=0.0, roughnessFactor=1.0))
            self.g.materials.append(mat)
            self._default_material_idx = len(self.g.materials) - 1
        return self._default_material_idx

    def _convert_material(self, pmat: Optional[PMaterial],
                          area_light: Optional[Tuple[str, Dict[str, Param]]],
                          shape_params: Optional[Dict[str, Param]] = None
                          ) -> int:
        if pmat is None and area_light is None:
            return self._ensure_default_material()

        kind = (pmat.kind if pmat else "diffuse").lower()
        params = pmat.params if pmat else {}
        mat = Material(name=pmat.name if pmat else None)
        mr = PbrMetallicRoughness(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=0.0, roughnessFactor=1.0)
        mat.pbrMetallicRoughness = mr

        # ---- kind-specific mapping -----------------------------------
        if kind in ("diffuse", "diffusetransmission"):
            self._apply_color(mr, params, "reflectance", default=(0.5, 0.5, 0.5))
            mr.metallicFactor = 0.0
            mr.roughnessFactor = 1.0
            if kind == "diffusetransmission":
                mat.doubleSided = True
                self._apply_color(mr, params, "transmittance",
                                  default=None, only_factor=True)
        elif kind in ("coateddiffuse",):
            self._apply_color(mr, params, "reflectance", default=(0.5, 0.5, 0.5))
            mr.metallicFactor = 0.0
            mr.roughnessFactor = self._roughness(params)
            self._apply_roughness_texture(mr, params, metallic_b=0)
        elif kind in ("conductor", "coatedconductor"):
            self._apply_color(mr, params, "reflectance",
                              default=(0.95, 0.93, 0.88))
            mr.metallicFactor = 1.0
            mr.roughnessFactor = self._roughness(params)
            self._apply_roughness_texture(mr, params, metallic_b=255)
        elif kind in ("dielectric", "thindielectric"):
            # Fully transmissive surface (glass). Alpha=0 so plain blending
            # also shows transparency in renderers that ignore
            # KHR_materials_transmission.
            mr.baseColorFactor = [1.0, 1.0, 1.0, 0.0]
            mr.metallicFactor = 0.0
            mr.roughnessFactor = self._roughness(params)
            self._apply_roughness_texture(mr, params, metallic_b=0)
            mat.alphaMode = "BLEND"
            mat.extensions = mat.extensions or {}
            mat.extensions["KHR_materials_transmission"] = {"transmissionFactor": 1.0}
            if "KHR_materials_transmission" not in self.g.extensionsUsed:
                self.g.extensionsUsed.append("KHR_materials_transmission")
            ior_p = params.get("eta") or params.get("ior")
            if ior_p is not None and ior_p.values:
                ior = float(ior_p.values[0]) if not isinstance(ior_p.values[0], str) else 1.5
                mat.extensions["KHR_materials_ior"] = {"ior": ior}
                if "KHR_materials_ior" not in self.g.extensionsUsed:
                    self.g.extensionsUsed.append("KHR_materials_ior")
        elif kind == "mix":
            # Take the first sub-material if provided as a named ref
            mats_p = params.get("materials")
            if mats_p and mats_p.values:
                first = mats_p.values[0]
                if isinstance(first, str):
                    sub = self.scene.materials_by_name.get(first)
                    if sub is not None:
                        return self._convert_material(sub, area_light)
            self._apply_color(mr, params, "reflectance", default=(0.5, 0.5, 0.5))
        elif kind in ("principled", "principledthin"):
            # Mitsuba's principled BSDF lines up with glTF's Disney-style
            # metallic-roughness model. Roughness is already perceptual
            # (no sqrt), and `metallic` maps to metallicFactor directly.
            self._apply_color(mr, params, "reflectance", default=(0.5, 0.5, 0.5))
            metallic_const = 0.0
            mp = params.get("metallic")
            if mp is not None and mp.values and not isinstance(mp.values[0], str):
                metallic_const = float(mp.values[0])
            mr.metallicFactor = max(0.0, min(1.0, metallic_const))
            rough_p = params.get("roughness")
            wrote_rough_tex = False
            if rough_p is not None and rough_p.values:
                first = rough_p.values[0]
                if isinstance(first, str):
                    metallic_b = int(round(255 * mr.metallicFactor))
                    wrote_rough_tex = self._apply_roughness_texture(
                        mr, params, metallic_b=metallic_b, sqrt_alpha=False)
                else:
                    mr.roughnessFactor = max(0.0, min(1.0, float(first)))
            if not wrote_rough_tex and rough_p is None:
                mr.roughnessFactor = 0.5
        elif kind == "subsurface":
            self._apply_color(mr, params, "reflectance", default=(0.7, 0.6, 0.5))
            mr.metallicFactor = 0.0
            mr.roughnessFactor = 0.7
        elif kind == "measured":
            mr.baseColorFactor = [0.7, 0.7, 0.7, 1.0]
        elif kind == "interface":
            return self._ensure_default_material()
        else:
            self._apply_color(mr, params, "reflectance", default=(0.7, 0.7, 0.7))

        # ---- area light -> emissive ----------------------------------
        if area_light is not None:
            _, light_params = area_light
            emission = self._emission_color(light_params)
            if emission is not None:
                mat.emissiveFactor = list(emission)
                # If above 1.0 in any channel, also enable strength extension.
                m = max(emission)
                if m > 1.0:
                    norm = [c / m for c in emission]
                    mat.emissiveFactor = norm
                    mat.extensions = mat.extensions or {}
                    mat.extensions["KHR_materials_emissive_strength"] = {
                        "emissiveStrength": m}
                    if "KHR_materials_emissive_strength" not in self.g.extensionsUsed:
                        self.g.extensionsUsed.append("KHR_materials_emissive_strength")
            # Area lights in PBRT/Mitsuba are one-sided by default; only
            # promote to doubleSided when the source explicitly opts in:
            #   PBRT v4 : AreaLightSource ... "bool twosided" true
            #   Mitsuba : <bsdf type="twosided"> wrapper (parser flags _twosided)
            if _is_twosided(light_params, pmat):
                mat.doubleSided = True

        # Promote any (non-emissive) twosided BSDF wrapper to glTF doubleSided.
        if pmat is not None and "_twosided" in pmat.params and not mat.doubleSided:
            mat.doubleSided = True

        # Normal map (any material kind)
        self._apply_normal_map(mat, params)

        # Alpha cutout / opacity. Sources, in priority order:
        #   shape_params["alpha"]  (PBRT v4 shape-level)
        #   pmat.params["_opacity"] (Mitsuba mask BSDF wrapper)
        # A texture source becomes glTF MASK with cutoff 0.5; a scalar
        # < 1 becomes BLEND with baseColorFactor[3] = scalar (or MASK
        # cutoff 0.5 when ~0).
        self._apply_alpha(mr, mat, params, shape_params)

        self.g.materials.append(mat)
        return len(self.g.materials) - 1

    def _apply_color(self, mr: PbrMetallicRoughness, params: Dict[str, Param],
                     pname: str, default=(1.0, 1.0, 1.0), only_factor=False):
        p = params.get(pname)
        if p is None:
            if default is not None:
                mr.baseColorFactor = [*default, 1.0]
            return
        if p.type == "texture" or (not p.type and p.values and
                                   isinstance(p.values[0], str)):
            tex_name = p.values[0]
            tex_idx = self._resolve_texture(tex_name)
            if tex_idx is not None and not only_factor:
                mr.baseColorTexture = self._wrap_texture_info(tex_idx, tex_name)
                mr.baseColorFactor = [1.0, 1.0, 1.0, 1.0]
                return
        if p.type == "blackbody" and p.values:
            t = float(p.values[0])
            r, g, b = _blackbody_to_rgb(t)
            mr.baseColorFactor = [r, g, b, 1.0]
            return
        vals = [v for v in p.values if not isinstance(v, str)]
        if len(vals) >= 3:
            mr.baseColorFactor = [float(vals[0]), float(vals[1]), float(vals[2]), 1.0]
        elif len(vals) == 1:
            v = float(vals[0])
            mr.baseColorFactor = [v, v, v, 1.0]
        elif default is not None:
            mr.baseColorFactor = [*default, 1.0]

    def _apply_normal_map(self, mat: Material, params: Dict[str, Param]):
        """Wire a glTF normalTexture from PBRT's `normalmap` param.

        PBRT v4 spells this `"string normalmap" "file.png"` (a direct
        filename), but some scenes write `"texture normalmap" "tex_name"`
        instead — handle both.
        """
        nm = params.get("normalmap")
        if nm is None or not nm.values:
            return
        v = nm.values[0]
        if not isinstance(v, str):
            return
        tex_idx = None
        tex_name_for_xform: Optional[str] = None
        if nm.type == "texture":
            tex_idx = self._resolve_texture(v)
            tex_name_for_xform = v
        else:
            path = (self.base_dir / v).resolve()
            if not path.exists():
                print(f"  warning: normal map not found: {path}")
                return
            tex_idx = self._texture_from_file(path)
        if tex_idx is None:
            return
        nti = self._wrap_texture_info(tex_idx, tex_name_for_xform,
                                      cls=NormalMaterialTexture)
        mat.normalTexture = nti

    def _apply_alpha(self, mr: PbrMetallicRoughness, mat: Material,
                     mat_params: Dict[str, Param],
                     shape_params: Optional[Dict[str, Param]]):
        """Set glTF alphaMode/alphaCutoff and bake alpha into the
        baseColor RGBA when an alpha source is present."""
        alpha = self._resolve_alpha_source(mat_params, shape_params)
        if alpha is None:
            return

        kind, val = alpha

        if kind == "float":
            a = max(0.0, min(1.0, float(val)))
            if a >= 0.999:
                return
            r, g, b, _ = mr.baseColorFactor
            if a <= 0.001:
                # Effectively invisible: write a clean MASK at cutoff 0.5
                # so the renderer culls the surface entirely.
                mr.baseColorFactor = [r, g, b, 0.0]
                mat.alphaMode = "MASK"
                mat.alphaCutoff = 0.5
                return
            mr.baseColorFactor = [r, g, b, a]
            mat.alphaMode = "BLEND"
            return

        # Texture source -> bake into baseColor RGBA, MASK at 0.5.
        alpha_tex_name = val
        alpha_path = self._resolve_texture_to_path(alpha_tex_name)
        if alpha_path is None or not alpha_path.exists():
            print(f"  warning: alpha texture not found: {alpha_tex_name}")
            return

        # Find the existing baseColor source so we can preserve its RGB
        # in the packed texture.
        base_tex_name: Optional[str] = None
        ref = mat_params.get("reflectance") if mat_params else None
        if ref is not None and ref.values and isinstance(ref.values[0], str):
            base_tex_name = ref.values[0]
        base_path = (self._resolve_texture_to_path(base_tex_name)
                     if base_tex_name else None)
        base_rgb = (float(mr.baseColorFactor[0]), float(mr.baseColorFactor[1]),
                    float(mr.baseColorFactor[2]))

        img_idx = self._pack_albedo_alpha_image(base_path, base_rgb, alpha_path)
        if img_idx is None:
            return

        sampler_idx = self._ensure_default_sampler()
        tex_key = ("albedo_alpha",
                   str(base_path) if base_path else None,
                   base_rgb, str(alpha_path))
        tex_idx = self._tex_index_cache.get(tex_key)
        if tex_idx is None:
            tex = Texture(source=img_idx, sampler=sampler_idx,
                          name=f"{alpha_path.stem}_albedo_alpha")
            self.g.textures.append(tex)
            tex_idx = len(self.g.textures) - 1
            self._tex_index_cache[tex_key] = tex_idx

        # Preserve any UV transform from the base or alpha texture.
        xform_name = base_tex_name or alpha_tex_name
        mr.baseColorTexture = self._wrap_texture_info(tex_idx, xform_name)
        mr.baseColorFactor = [1.0, 1.0, 1.0, 1.0]
        mat.alphaMode = "MASK"
        mat.alphaCutoff = 0.5

    def _resolve_alpha_source(self, mat_params: Optional[Dict[str, Param]],
                              shape_params: Optional[Dict[str, Param]]
                              ) -> Optional[Tuple[str, object]]:
        """Return the alpha/opacity source as ('texture', tex_name) or
        ('float', value). PBRT's shape-level ``alpha`` wins over a Mitsuba
        mask wrapper's ``_opacity`` carried on the material."""
        for src, key in ((shape_params, "alpha"), (mat_params, "_opacity")):
            if not src:
                continue
            p = src.get(key)
            if p is None or not p.values:
                continue
            v = p.values[0]
            if isinstance(v, str):
                return ("texture", v)
            try:
                return ("float", float(v))
            except (TypeError, ValueError):
                continue
        return None

    def _resolve_texture_to_path(self, tex_name: Optional[str]) -> Optional[Path]:
        """Resolve a named PBRT/Mitsuba texture down to its source image
        on disk. Returns None if the texture isn't an imagemap or its
        ``filename`` is missing."""
        if not tex_name:
            return None
        tdef = self.scene.textures.get(tex_name)
        if tdef is None or (tdef.tclass or "").lower() != "imagemap":
            return None
        fname = _first_str(tdef.params.get("filename"))
        if not fname:
            return None
        return (self.base_dir / fname).resolve()

    def _pack_albedo_alpha_image(self, base_path: Optional[Path],
                                 base_rgb: Tuple[float, float, float],
                                 alpha_path: Path) -> Optional[int]:
        """Build an RGBA PNG with RGB from ``base_path`` (or solid
        ``base_rgb`` when no base texture exists) and A from
        ``alpha_path``. Cached by source paths and base color."""
        key = ("albedo_alpha", str(base_path) if base_path else None,
               tuple(base_rgb), str(alpha_path))
        if key in self._tex_image_cache:
            return self._tex_image_cache[key]

        try:
            from PIL import Image as PILImage
            with PILImage.open(alpha_path) as a_im:
                a_im.load()
                # If the alpha source is RGBA, prefer its alpha channel;
                # otherwise treat the image's luminance as the mask.
                if a_im.mode == "RGBA":
                    alpha_arr = np.asarray(a_im.split()[-1], dtype=np.uint8)
                else:
                    alpha_arr = np.asarray(a_im.convert("L"), dtype=np.uint8)
            h, w = alpha_arr.shape

            if base_path is not None and base_path.exists():
                with PILImage.open(base_path) as b_im:
                    b_im.load()
                    b_rgb = b_im.convert("RGB")
                    if b_rgb.size != (w, h):
                        b_rgb = b_rgb.resize((w, h), PILImage.LANCZOS)
                    rgb_arr = np.asarray(b_rgb, dtype=np.uint8)
            else:
                r = int(round(255 * max(0.0, min(1.0, base_rgb[0]))))
                g = int(round(255 * max(0.0, min(1.0, base_rgb[1]))))
                b = int(round(255 * max(0.0, min(1.0, base_rgb[2]))))
                rgb_arr = np.empty((h, w, 3), dtype=np.uint8)
                rgb_arr[:, :, 0] = r
                rgb_arr[:, :, 1] = g
                rgb_arr[:, :, 2] = b

            rgba = np.empty((h, w, 4), dtype=np.uint8)
            rgba[..., :3] = rgb_arr
            rgba[..., 3] = alpha_arr
            out = PILImage.fromarray(rgba, mode="RGBA")
            buf = io.BytesIO()
            out.save(buf, format="PNG")
            data = buf.getvalue()
        except Exception as e:
            print(f"  warning: could not pack albedo+alpha "
                  f"({alpha_path.name}): {e}")
            return None

        bv_idx = self._add_buffer_view(data)
        image = Image(mimeType="image/png", bufferView=bv_idx,
                      name=alpha_path.stem + "_albedo_alpha")
        self.g.images.append(image)
        idx = len(self.g.images) - 1
        self._tex_image_cache[key] = idx
        return idx

    def _apply_roughness_texture(self, mr: PbrMetallicRoughness,
                                 params: Dict[str, Param],
                                 metallic_b: int = 0,
                                 sqrt_alpha: bool = True) -> bool:
        """If `roughness` is a texture reference, build a packed glTF
        metallicRoughness texture (G=roughness, B=metallic_b).

        `sqrt_alpha=True` means the source values are PBRT/Mitsuba α
        (GGX); `sqrt(α)` is applied to convert to glTF perceptual
        roughness. For sources that are already perceptual (Mitsuba's
        Disney/principled `roughness`), pass `sqrt_alpha=False`.
        """
        p = params.get("roughness")
        if p is None or not p.values:
            return False
        is_tex = (p.type == "texture" or
                  (not p.type and isinstance(p.values[0], str)))
        if not is_tex:
            return False
        tex_name = p.values[0]
        tdef = self.scene.textures.get(tex_name)
        if tdef is None:
            return False
        if tdef.tclass.lower() != "imagemap":
            return False
        fname = _first_str(tdef.params.get("filename"))
        if not fname:
            return False
        rough_path = (self.base_dir / fname).resolve()
        if not rough_path.exists():
            print(f"  warning: roughness texture not found: {rough_path}")
            return False

        img_idx = self._pack_metallic_roughness_image(
            rough_path, metallic_b, sqrt_alpha=sqrt_alpha)
        if img_idx is None:
            return False

        # Reuse one packed Texture per (image, metallic, sqrt) combination.
        tex_key = ("mr", str(rough_path), metallic_b, sqrt_alpha)
        tex_idx = self._tex_index_cache.get(tex_key)
        if tex_idx is None:
            sampler_idx = self._ensure_default_sampler()
            tex = Texture(source=img_idx, sampler=sampler_idx,
                          name=f"{tex_name}_mr")
            self.g.textures.append(tex)
            tex_idx = len(self.g.textures) - 1
            self._tex_index_cache[tex_key] = tex_idx

        mr.metallicRoughnessTexture = self._wrap_texture_info(
            tex_idx, tex_name)
        # The texture drives roughness; reset the factor to neutral.
        mr.roughnessFactor = 1.0
        return True

    def _pack_metallic_roughness_image(self, rough_path: Path,
                                       metallic_b: int = 0,
                                       sqrt_alpha: bool = True
                                       ) -> Optional[int]:
        """Build a glTF metallicRoughness image with G=roughness,
        B=metallic_b. If `sqrt_alpha`, apply sqrt to convert α→perceptual.
        """
        key = ("mr", str(rough_path), metallic_b, sqrt_alpha)
        if key in self._tex_image_cache:
            return self._tex_image_cache[key]
        try:
            from PIL import Image as PILImage
            with PILImage.open(rough_path) as src:
                src.load()
                gray = src.convert("L")
                arr = np.asarray(gray, dtype=np.uint8)
        except Exception as e:
            print(f"  warning: could not load roughness {rough_path}: {e}")
            return None

        a_norm = arr.astype(np.float32) / 255.0
        if sqrt_alpha:
            a_norm = np.sqrt(a_norm)  # PBRT α -> glTF perceptual roughness
        rough_uint = np.clip(a_norm * 255.0 + 0.5, 0, 255).astype(np.uint8)
        h, w = rough_uint.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :, 1] = rough_uint                              # G = roughness
        rgb[:, :, 2] = np.uint8(max(0, min(255, metallic_b)))  # B = metallic

        try:
            from PIL import Image as PILImage
            out = PILImage.fromarray(rgb)
            buf = io.BytesIO()
            out.save(buf, format="PNG")
            data = buf.getvalue()
        except Exception as e:
            print(f"  warning: could not encode MR image {rough_path}: {e}")
            return None

        bv_idx = self._add_buffer_view(data)
        image = Image(mimeType="image/png", bufferView=bv_idx,
                      name=rough_path.stem + "_mr")
        self.g.images.append(image)
        idx = len(self.g.images) - 1
        self._tex_image_cache[key] = idx
        return idx

    def _roughness(self, params: Dict[str, Param]) -> float:
        # PBRT v4 roughness is alpha (GGX). glTF wants perceptual roughness
        # where alpha = roughness^2.  Convert: glTF_rough = sqrt(alpha).
        # When no roughness is specified the surface is a perfect mirror
        # (PBRT v4 conductor/dielectric default; Mitsuba `conductor` is also
        # smooth). Returning 0.0 here keeps mirrors mirror-like.
        p = params.get("roughness")
        if p is None:
            ur = params.get("uroughness")
            vr = params.get("vroughness")
            if ur is None and vr is None:
                return 0.0
            a = float(ur.values[0]) if ur else float(vr.values[0])
            b = float(vr.values[0]) if vr else a
            alpha = 0.5 * (a + b)
        else:
            v = p.values[0]
            if isinstance(v, str):
                # Textured roughness: factor neutralised; a texture is
                # written separately by _apply_roughness_texture.
                return 1.0
            alpha = float(v)
        alpha = max(0.0, min(1.0, alpha))
        return math.sqrt(alpha)

    def _emission_color(self, light_params: Dict[str, Param]):
        # AreaLightSource "diffuse" usually uses 'L' (or 'rgb L') for radiance
        L = light_params.get("L")
        scale = 1.0
        sp = light_params.get("scale")
        if sp is not None and sp.values and not isinstance(sp.values[0], str):
            scale = float(sp.values[0])
        if L is not None:
            if L.type == "blackbody" and L.values:
                r, g, b = _blackbody_to_rgb(float(L.values[0]))
                return (r * scale, g * scale, b * scale)
            vals = [v for v in L.values if not isinstance(v, str)]
            if len(vals) >= 3:
                return (vals[0] * scale, vals[1] * scale, vals[2] * scale)
            if len(vals) == 1:
                v = vals[0] * scale
                return (v, v, v)
        return (1.0 * scale, 1.0 * scale, 1.0 * scale)

    # ---------------------------------------------------------------
    # Textures / images
    # ---------------------------------------------------------------

    def _resolve_texture(self, tex_name: str) -> Optional[int]:
        key = ("named", tex_name)
        if key in self._tex_index_cache:
            return self._tex_index_cache[key]
        tdef = self.scene.textures.get(tex_name)
        if tdef is None:
            return None
        idx = self._build_texture(tdef)
        if idx is not None:
            self._tex_index_cache[key] = idx
        return idx

    def _build_texture(self, tdef: TextureDef) -> Optional[int]:
        tclass = (tdef.tclass or "").lower()
        if tclass == "imagemap":
            fname = _first_str(tdef.params.get("filename"))
            if not fname:
                return None
            path = (self.base_dir / fname).resolve()
            img_idx = self._add_image(path)
            if img_idx is None:
                return None
            sampler_idx = self._ensure_default_sampler()
            tex = Texture(source=img_idx, sampler=sampler_idx, name=tdef.name)
            self.g.textures.append(tex)
            return len(self.g.textures) - 1
        if tclass == "constant":
            # A constant texture isn't really a texture in glTF — skip.
            return None
        if tclass in ("scale", "mix"):
            # Try to resolve nested texture reference
            nested = (tdef.params.get("tex") or tdef.params.get("tex1") or
                      tdef.params.get("tex2"))
            if nested and isinstance(nested.values[0], str):
                return self._resolve_texture(nested.values[0])
            return None
        # checkerboard, dots, marble, etc. are procedural — bake skipped
        print(f"  warning: texture class '{tdef.tclass}' not supported, skipped")
        return None

    def _texture_from_file(self, path: Path) -> Optional[int]:
        """Build a Texture for a direct file path (used for `normalmap`)."""
        key = ("file", str(path))
        if key in self._tex_index_cache:
            return self._tex_index_cache[key]
        img_idx = self._add_image(path)
        if img_idx is None:
            return None
        sampler_idx = self._ensure_default_sampler()
        tex = Texture(source=img_idx, sampler=sampler_idx, name=path.stem)
        self.g.textures.append(tex)
        idx = len(self.g.textures) - 1
        self._tex_index_cache[key] = idx
        return idx

    def _uv_transform_for(self, tex_name: str) -> Optional[dict]:
        """Build a KHR_texture_transform dict from a PBRT texture's mapping params,
        or None if the transform is the identity."""
        tdef = self.scene.textures.get(tex_name)
        if tdef is None:
            return None
        p = tdef.params

        def _f(name: str, default: float) -> float:
            x = p.get(name)
            if x is None or not x.values:
                return default
            v = x.values[0]
            return float(v) if not isinstance(v, str) else default

        uscale, vscale = _f("uscale", 1.0), _f("vscale", 1.0)
        udelta, vdelta = _f("udelta", 0.0), _f("vdelta", 0.0)
        if uscale == 1.0 and vscale == 1.0 and udelta == 0.0 and vdelta == 0.0:
            return None
        return {"scale": [uscale, vscale], "offset": [udelta, vdelta]}

    def _wrap_texture_info(self, idx: int, tex_name: Optional[str],
                           cls=TextureInfo):
        ti = cls(index=idx)
        if tex_name is not None:
            xform = self._uv_transform_for(tex_name)
            if xform is not None:
                ti.extensions = {"KHR_texture_transform": xform}
                if "KHR_texture_transform" not in self.g.extensionsUsed:
                    self.g.extensionsUsed.append("KHR_texture_transform")
        return ti

    def _ensure_default_sampler(self) -> int:
        if self._default_sampler_idx is None:
            s = Sampler(
                magFilter=FILTER_LINEAR, minFilter=FILTER_LINEAR_MIPMAP,
                wrapS=WRAP_REPEAT, wrapT=WRAP_REPEAT)
            self.g.samplers.append(s)
            self._default_sampler_idx = len(self.g.samplers) - 1
        return self._default_sampler_idx

    def _add_image(self, path: Path) -> Optional[int]:
        key = ("file", str(path))
        if key in self._tex_image_cache:
            return self._tex_image_cache[key]
        if not path.exists():
            print(f"  warning: image not found: {path}")
            return None
        mime, data = _load_image_bytes(path)
        if data is None:
            return None
        bv_idx = self._add_buffer_view(data)
        img = Image(mimeType=mime, bufferView=bv_idx, name=path.stem)
        self.g.images.append(img)
        idx = len(self.g.images) - 1
        self._tex_image_cache[key] = idx
        return idx

    # ---------------------------------------------------------------
    # Camera
    # ---------------------------------------------------------------

    def _build_camera_node(self) -> Optional[int]:
        cam: CameraDef = self.scene.camera
        if cam is None or cam.kind != "perspective":
            return None
        params = cam.params
        fov = float(params["fov"].values[0]) if "fov" in params else 90.0
        # PBRT v4 fov is along the smaller image dimension; convert to glTF yfov.
        w = max(1, cam.film_width)
        h = max(1, cam.film_height)
        if w >= h:
            yfov_deg = fov
        else:
            tan_half = math.tan(math.radians(fov) / 2.0) * (h / w)
            yfov_deg = math.degrees(2.0 * math.atan(tan_half))
        yfov_rad = math.radians(yfov_deg)
        aspect = w / h
        c = Camera(
            type="perspective",
            perspective=Perspective(
                aspectRatio=aspect, yfov=yfov_rad, znear=0.01, zfar=10000.0))
        self.g.cameras.append(c)
        cam_idx = len(self.g.cameras) - 1

        # PBRT's CTM at WorldBegin is world-to-camera. The glTF camera node
        # matrix should be camera-to-world, i.e. the inverse.
        try:
            cam_to_world = np.linalg.inv(cam.world_to_camera)
        except np.linalg.LinAlgError:
            cam_to_world = np.eye(4)
        node = Node(camera=cam_idx, matrix=_to_gltf_matrix(cam_to_world),
                    name="camera")
        self.g.nodes.append(node)
        return len(self.g.nodes) - 1

    # ---------------------------------------------------------------
    # Lights
    # ---------------------------------------------------------------

    def _build_light_node(self, light: LightDef) -> Optional[int]:
        kind = (light.kind or "").lower()
        params = light.params
        color = (1.0, 1.0, 1.0)
        intensity = 1.0

        L = params.get("L") or params.get("I")
        if L is not None:
            if L.type == "blackbody" and L.values:
                color = _blackbody_to_rgb(float(L.values[0]))
            else:
                vals = [v for v in L.values if not isinstance(v, str)]
                if len(vals) >= 3:
                    m = max(vals[0], vals[1], vals[2], 1e-6)
                    color = (vals[0] / m, vals[1] / m, vals[2] / m)
                    intensity = m
                elif len(vals) == 1:
                    intensity = float(vals[0])

        scale = params.get("scale")
        if scale is not None and scale.values and not isinstance(scale.values[0], str):
            intensity *= float(scale.values[0])

        light_def: dict
        if kind == "point":
            light_def = {"type": "point", "color": list(color),
                         "intensity": intensity}
            translation = light.ctm[:3, 3].tolist()
            return self._add_punctual_light(light_def, translation=translation)
        if kind == "spot":
            cone = float(params["coneangle"].values[0]) if "coneangle" in params else 30.0
            falloff = float(params["conedeltaangle"].values[0]) if "conedeltaangle" in params else 5.0
            outer = math.radians(cone)
            inner = max(0.0, math.radians(cone - falloff))
            light_def = {"type": "spot", "color": list(color),
                         "intensity": intensity,
                         "spot": {"innerConeAngle": inner, "outerConeAngle": outer}}
            translation = light.ctm[:3, 3].tolist()
            return self._add_punctual_light(light_def, translation=translation)
        if kind == "distant":
            light_def = {"type": "directional", "color": list(color),
                         "intensity": intensity}
            # Direction: from -> to, in PBRT given as positional 'from'/'to' or
            # via the CTM (-Z is the forward of glTF directional lights).
            f = params.get("from")
            t = params.get("to")
            direction = None
            if f is not None and t is not None:
                fv = np.array(_floats(f, [0, 0, 0]) + [1.0])
                tv = np.array(_floats(t, [0, 0, -1]) + [1.0])
                fv = light.ctm @ fv
                tv = light.ctm @ tv
                d = (tv[:3] - fv[:3])
                ln = np.linalg.norm(d)
                if ln > 0:
                    direction = d / ln
            if direction is None:
                direction = -light.ctm[:3, 2]
                ln = np.linalg.norm(direction)
                if ln > 0:
                    direction = direction / ln
            return self._add_punctual_light(
                light_def, direction=direction.tolist())
        if kind in ("infinite", "goniometric", "projection"):
            print(f"  warning: light '{kind}' not exportable to glTF, skipped")
            return None
        print(f"  warning: unknown light '{kind}', skipped")
        return None

    def _add_punctual_light(self, light_def: dict,
                            translation=None, direction=None) -> int:
        self._punctual_lights.append(light_def)
        light_index = len(self._punctual_lights) - 1
        node = Node(name=f"light_{light_index}")
        if translation is not None:
            node.translation = translation
        if direction is not None:
            # glTF directional lights point along -Z of their node, so build
            # a rotation that rotates -Z onto `direction`.
            node.rotation = _rotation_to_dir(direction)
        node.extensions = {"KHR_lights_punctual": {"light": light_index}}
        self.g.nodes.append(node)
        return len(self.g.nodes) - 1


# -------------------------------------------------------------------------
# Misc helpers
# -------------------------------------------------------------------------

def _to_gltf_matrix(M: np.ndarray):
    """Convert a 4x4 row-major matrix to glTF's column-major flat list of 16."""
    return [float(x) for x in M.T.reshape(-1)]


def _rotation_to_dir(direction):
    """Return a quaternion that rotates -Z onto `direction`."""
    d = np.array(direction, dtype=np.float64)
    n = np.linalg.norm(d)
    if n == 0:
        return [0.0, 0.0, 0.0, 1.0]
    d /= n
    base = np.array([0.0, 0.0, -1.0])
    dot = float(np.dot(base, d))
    if dot >= 1.0 - 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    if dot <= -1.0 + 1e-9:
        # 180 deg around any axis perpendicular to base
        return [1.0, 0.0, 0.0, 0.0]
    axis = np.cross(base, d)
    axis_n = np.linalg.norm(axis)
    if axis_n == 0:
        return [0.0, 0.0, 0.0, 1.0]
    axis /= axis_n
    angle = math.acos(max(-1.0, min(1.0, dot)))
    s = math.sin(angle / 2.0)
    return [float(axis[0] * s), float(axis[1] * s), float(axis[2] * s),
            float(math.cos(angle / 2.0))]


def _load_image_bytes(path: Path):
    """Returns (mime_type, bytes) or (None, None) on failure."""
    ext = path.suffix.lower()
    if ext == ".png":
        return ("image/png", path.read_bytes())
    if ext in (".jpg", ".jpeg"):
        return ("image/jpeg", path.read_bytes())
    # Fallback: load via Pillow / imageio and re-encode as PNG
    arr = None
    try:
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            im.load()
            if im.mode in ("I;16", "I"):
                im = im.convert("L")
            elif im.mode == "P":
                im = im.convert("RGBA")
            elif im.mode not in ("L", "LA", "RGB", "RGBA"):
                im = im.convert("RGBA")
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return ("image/png", buf.getvalue())
    except Exception:
        pass
    try:
        import imageio.v3 as iio
        arr = iio.imread(str(path))
    except Exception as e:
        print(f"  warning: could not load image {path}: {e}")
        return (None, None)

    a = np.asarray(arr)
    if a.dtype.kind == "f":
        a = np.clip(a, 0.0, 1.0)
        a = (a * 255.0 + 0.5).astype(np.uint8)
    elif a.dtype == np.uint16:
        a = (a // 257).astype(np.uint8)
    elif a.dtype != np.uint8:
        a = a.astype(np.uint8)
    try:
        from PIL import Image as PILImage
        im = PILImage.fromarray(a)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return ("image/png", buf.getvalue())
    except Exception as e:
        print(f"  warning: could not encode image {path}: {e}")
        return (None, None)


def export_to_glb(scene: Scene, output_path):
    """Top-level helper used by the CLI."""
    output_path = Path(output_path)
    exporter = GLTFExporter(scene, scene.base_dir)
    exporter.export(output_path)
