"""Build an in-memory Scene from a Mitsuba 3 XML scene file.

Translates Mitsuba's plugin/parameter vocabulary into the PBRT-flavoured
``Scene`` data classes used by the existing glTF/OBJ exporters, so they
can stay unchanged. This is a pragmatic parser that targets the plugins
found in typical Mitsuba 3 example scenes; unrecognised plugins are
skipped with a warning rather than aborting the whole conversion.

Mappings used:
    BSDF            diffuse/roughdiffuse        -> "diffuse"
                    dielectric/thindielectric/
                      roughdielectric           -> "dielectric"/"thindielectric"
                    conductor/roughconductor    -> "conductor"
                    plastic/roughplastic        -> "coateddiffuse"
                    principled/principledthin   -> "principled" (handled by
                                                   the exporters as a Disney-
                                                   style PBR material)
                    twosided                    -> inner BSDF + ``_twosided``
                    bumpmap/normalmap           -> inner BSDF + ``normalmap``
                    mask                        -> inner BSDF
                    blendbsdf/blend             -> first sub-BSDF
    Shape           ply                         -> "plymesh" (lazy)
                    obj                         -> "objmesh" (lazy)
                    rectangle/disk/cube/cylinder -> "trianglemesh" (eager)
                    sphere                      -> "sphere"
                    shapegroup/instance         -> object templates +
                                                   ``ShapeEntry(kind="instance")``
    Emitter         point/spot/directional/
                    constant                    -> ``LightDef``
                    area (inline on a shape)    -> shape's ``area_light``
    Sensor          perspective                 -> ``CameraDef`` (with a
                                                   Z-flip so its world-to-
                                                   camera matches the glTF
                                                   ``-Z forward`` convention
                                                   used by the exporter)
"""

import math
import os
import warnings
import xml.etree.ElementTree as ET
from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pbrt_parser import Param
from pbrt_scene import (
    CameraDef, LightDef, Material, Scene, ShapeEntry, TextureDef,
    _identity, _lookat, _rotate, _scale, _translate,
)


# -------------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------------

def _split_floats(text: str) -> List[float]:
    """Mitsuba accepts both 'x, y, z' and 'x y z' forms."""
    parts = text.replace(",", " ").split()
    return [float(p) for p in parts]


def _parse_bool(text: str) -> bool:
    if isinstance(text, bool):
        return text
    return str(text).strip().lower() in ("true", "1", "yes", "on")


def _xyz_attribs(elem: ET.Element) -> Optional[Tuple[float, float, float]]:
    """Mitsuba <translate>/<scale>/<vector>/<point> can use either a
    'value="x y z"' attribute or separate 'x','y','z' attributes.
    Returns None if neither form is present.
    """
    if "value" in elem.attrib:
        vals = _split_floats(elem.attrib["value"])
        if len(vals) == 1:
            return (vals[0], vals[0], vals[0])
        if len(vals) >= 3:
            return (vals[0], vals[1], vals[2])
        return None
    if any(k in elem.attrib for k in ("x", "y", "z")):
        return (float(elem.attrib.get("x", 0.0)),
                float(elem.attrib.get("y", 0.0)),
                float(elem.attrib.get("z", 0.0)))
    return None


def _strip_ns(tag: str) -> str:
    """ElementTree prepends namespaces as '{ns}tag'; Mitsuba scenes
    rarely use one, but be defensive."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


# -------------------------------------------------------------------------
# Scalar parameter unpacking (Mitsuba <float>/<integer>/<string>/<rgb>/...)
# -------------------------------------------------------------------------

# Mitsuba parameter tag -> PBRT-style Param.type. The exporters key off
# this string to decide whether a value is a texture reference, a colour,
# a blackbody, etc.
_SCALAR_TYPE_MAP = {
    "float":    "float",
    "integer":  "integer",
    "boolean":  "bool",
    "string":   "string",
    "rgb":      "rgb",
    "color":    "rgb",
    "spectrum": "rgb",      # we collapse spectra to rgb on read
    "vector":   "vector3",
    "point":    "point3",
    "normal":   "normal3",
}


def _read_scalar_param(elem: ET.Element, defaults: Dict[str, str]) -> Optional[Tuple[str, Param]]:
    tag = _strip_ns(elem.tag).lower()
    name = elem.attrib.get("name")
    if name is None:
        return None
    ptype = _SCALAR_TYPE_MAP.get(tag)
    if ptype is None:
        return None

    raw = _resolve_default(elem.attrib.get("value", ""), defaults)
    values: List[Any]

    if tag == "string":
        values = [raw]
    elif tag == "boolean":
        values = [_parse_bool(raw)]
    elif tag == "integer":
        try:
            values = [int(float(raw))]
        except ValueError:
            values = [0]
    elif tag == "float":
        try:
            values = [float(raw)]
        except ValueError:
            values = [0.0]
    elif tag in ("rgb", "color", "spectrum"):
        nums = _split_floats(raw)
        if len(nums) == 1:
            values = [nums[0], nums[0], nums[0]]
        elif len(nums) == 0:
            values = [1.0, 1.0, 1.0]
        else:
            values = nums[:3] if len(nums) >= 3 else (nums + [nums[-1]] * (3 - len(nums)))
    elif tag in ("vector", "point", "normal"):
        nums = _split_floats(raw)
        values = nums[:3] if len(nums) >= 3 else nums + [0.0] * (3 - len(nums))
    else:
        values = [raw]

    return name, Param(type=ptype, name=name, values=values)


def _resolve_default(text: str, defaults: Dict[str, str]) -> str:
    """Expand Mitsuba '$name' placeholders using a flat defaults table."""
    if not text or "$" not in text:
        return text
    out = []
    i = 0
    while i < len(text):
        c = text[i]
        if c == "$":
            # take ident chars
            j = i + 1
            while j < len(text) and (text[j].isalnum() or text[j] in "_"):
                j += 1
            key = text[i + 1:j]
            if key in defaults:
                out.append(defaults[key])
            else:
                out.append(text[i:j])
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


# -------------------------------------------------------------------------
# Builder
# -------------------------------------------------------------------------

class MitsubaSceneBuilder:
    def __init__(self, root_path: Path):
        self.scene = Scene(base_dir=root_path.parent)
        self.root_path = root_path

        self._defaults: Dict[str, str] = {}
        self._material_ids: Dict[str, Material] = {}
        self._texture_ids: Dict[str, TextureDef] = {}
        self._included_files: set = set()
        self._current_group: Optional[str] = None

        # Counters for anonymous ids
        self._next_mat_id = 1
        self._next_tex_id = 1
        self._next_group_id = 1

    # --------------------------------------------------------------
    # Top-level
    # --------------------------------------------------------------

    def build(self, root: ET.Element) -> Scene:
        self._walk_top(root, self.scene.base_dir)
        return self.scene

    def _walk_top(self, root: ET.Element, base_dir: Path):
        for elem in list(root):
            self._dispatch_top(elem, base_dir)

    def _dispatch_top(self, elem: ET.Element, base_dir: Path):
        tag = _strip_ns(elem.tag).lower()

        if tag == "default":
            name = elem.attrib.get("name")
            if name and "value" in elem.attrib:
                self._defaults.setdefault(name, elem.attrib["value"])
            return

        if tag == "include":
            fname = _resolve_default(elem.attrib.get("filename", ""), self._defaults)
            if not fname:
                return
            sub = (base_dir / fname).resolve()
            if str(sub) in self._included_files:
                return
            self._included_files.add(str(sub))
            try:
                sub_root = ET.parse(sub).getroot()
            except (FileNotFoundError, ET.ParseError) as e:
                print(f"  warning: Mitsuba include failed: {sub} ({e})")
                return
            self._walk_top(sub_root, sub.parent)
            return

        if tag == "sensor":
            self._process_sensor(elem)
            return

        if tag == "shape":
            self._process_shape(elem)
            return

        if tag == "bsdf":
            mat = self._build_material(elem)
            if mat is not None:
                self._register_material(elem.attrib.get("id"), mat)
            return

        if tag == "texture":
            tdef = self._build_texture(elem)
            if tdef is not None:
                self._register_texture(elem.attrib.get("id"), tdef)
            return

        if tag == "emitter":
            self._process_global_emitter(elem)
            return

        # integrator, sampler, film, medium, phase, etc. are skipped
        return

    # --------------------------------------------------------------
    # Defaults / id registration
    # --------------------------------------------------------------

    def _register_material(self, eid: Optional[str], mat: Material):
        if eid:
            mat.name = eid
            self._material_ids[eid] = mat
            self.scene.materials_by_name[eid] = mat
        else:
            anon = f"mtsmat_{self._next_mat_id}"
            self._next_mat_id += 1
            mat.name = anon
            self.scene.materials_by_name[anon] = mat

    def _register_texture(self, eid: Optional[str], tdef: TextureDef):
        if eid:
            tdef.name = eid
            self._texture_ids[eid] = tdef
            self.scene.textures[eid] = tdef
        else:
            anon = f"mtstex_{self._next_tex_id}"
            self._next_tex_id += 1
            tdef.name = anon
            self.scene.textures[anon] = tdef

    # --------------------------------------------------------------
    # Transforms
    # --------------------------------------------------------------

    def _read_transform(self, elem: ET.Element) -> np.ndarray:
        """Walk a <transform> element's children and accumulate into a
        single 4x4 matrix. Mitsuba composes children left-to-right with
        post-multiplication: ``M = M @ child`` -- the first child is
        applied first to a point in the local frame, matching PBRT."""
        M = _identity()
        for child in elem:
            tag = _strip_ns(child.tag).lower()
            if tag == "translate":
                xyz = _xyz_attribs(child)
                if xyz is not None:
                    M = M @ _translate(*xyz)
            elif tag == "scale":
                xyz = _xyz_attribs(child)
                if xyz is not None:
                    M = M @ _scale(*xyz)
            elif tag == "rotate":
                ax = float(child.attrib.get("x", 0.0))
                ay = float(child.attrib.get("y", 0.0))
                az = float(child.attrib.get("z", 0.0))
                angle = float(child.attrib.get("angle", 0.0))
                M = M @ _rotate(angle, ax, ay, az)
            elif tag == "matrix":
                vals = _split_floats(_resolve_default(child.attrib.get("value", ""),
                                                      self._defaults))
                if len(vals) == 16:
                    M = M @ np.array(vals, dtype=np.float64).reshape(4, 4)
            elif tag == "lookat":
                origin = self._point_attrib(child, ("origin",), (0.0, 0.0, 0.0))
                target = self._point_attrib(child, ("target",), (0.0, 0.0, 1.0))
                up = self._point_attrib(child, ("up",), (0.0, 1.0, 0.0))
                M = M @ _mitsuba_lookat(origin, target, up)
        return M

    def _point_attrib(self, elem: ET.Element, names: Tuple[str, ...],
                      default: Tuple[float, float, float]):
        for n in names:
            if n in elem.attrib:
                vals = _split_floats(elem.attrib[n])
                if len(vals) >= 3:
                    return (vals[0], vals[1], vals[2])
        return default

    # --------------------------------------------------------------
    # Sensor (camera)
    # --------------------------------------------------------------

    def _process_sensor(self, elem: ET.Element):
        if elem.attrib.get("type") != "perspective":
            return

        fov = 39.3077  # Mitsuba's default fov
        fov_axis = "x"
        to_world = _identity()
        width, height = 768, 576

        for child in elem:
            tag = _strip_ns(child.tag).lower()
            name = child.attrib.get("name")
            if tag == "transform" and name == "to_world":
                to_world = self._read_transform(child)
            elif tag == "float" and name == "fov":
                fov = float(_resolve_default(child.attrib.get("value", "39.3077"),
                                             self._defaults))
            elif tag == "string" and name == "fov_axis":
                fov_axis = _resolve_default(child.attrib.get("value", "x"),
                                            self._defaults).strip().lower()
            elif tag == "film":
                w, h = self._read_film(child)
                width, height = w, h
            # near/far/sampler ignored

        # Mitsuba 3's perspective sensor has its local frame oriented as
        # (+X = left, +Y = up, +Z = forward). glTF's camera is the 180°-
        # rotated version: (+X = right, +Y = up, -Z = forward). Compose
        # Mitsuba's `to_world` with a Y-axis 180° rotation -- flipping
        # both the +X and +Z columns -- so the eventual cam-to-world
        # matrix the glTF exporter writes places the glTF camera the
        # same way Mitsuba placed its sensor (and stays a proper
        # rotation, det = +1).
        flip_xz = np.diag([-1.0, 1.0, -1.0, 1.0])
        cam_to_world_gltf = to_world @ flip_xz
        try:
            world_to_camera = np.linalg.inv(cam_to_world_gltf)
        except np.linalg.LinAlgError:
            world_to_camera = _identity()

        pbrt_fov = self._mitsuba_fov_to_pbrt_smaller(fov, fov_axis, width, height)
        self.scene.camera = CameraDef(
            kind="perspective",
            params={"fov": Param(type="float", name="fov", values=[pbrt_fov])},
            world_to_camera=world_to_camera,
            film_width=width, film_height=height,
        )
        self.scene.film_width = width
        self.scene.film_height = height

    def _read_film(self, elem: ET.Element) -> Tuple[int, int]:
        w, h = 768, 576
        for child in elem:
            tag = _strip_ns(child.tag).lower()
            if tag == "integer":
                name = child.attrib.get("name")
                v = int(float(_resolve_default(child.attrib.get("value", "0"),
                                               self._defaults)))
                if name == "width":
                    w = v
                elif name == "height":
                    h = v
        return w, h

    def _mitsuba_fov_to_pbrt_smaller(self, fov: float, axis: str,
                                     w: int, h: int) -> float:
        """PBRT v4's `Camera "perspective"` ``fov`` is along the smaller
        image dimension. Convert any of Mitsuba's fov-axis conventions to
        that representation so the existing exporter math stays valid.
        """
        axis = (axis or "x").strip().lower()
        if axis == "smaller":
            return fov
        if axis == "larger":
            # Switch sides
            return self._mitsuba_fov_to_pbrt_smaller(
                fov, "x" if w >= h else "y", w, h)

        # Resolve to vertical fov first
        if axis == "y":
            yfov_deg = fov
        elif axis == "diagonal":
            diag = math.hypot(w, h) or 1.0
            t = math.tan(math.radians(fov) / 2.0) * (h / diag)
            yfov_deg = math.degrees(2.0 * math.atan(t))
        else:  # 'x' (Mitsuba default) or anything else
            denom = max(1, w)
            t = math.tan(math.radians(fov) / 2.0) * (h / denom)
            yfov_deg = math.degrees(2.0 * math.atan(t))

        # Then return fov along the smaller dimension
        if w >= h:
            return yfov_deg
        denom = max(1, h)
        t = math.tan(math.radians(yfov_deg) / 2.0) * (w / denom)
        return math.degrees(2.0 * math.atan(t))

    # --------------------------------------------------------------
    # Materials (BSDFs)
    # --------------------------------------------------------------

    def _build_material(self, elem: ET.Element) -> Optional[Material]:
        btype = (elem.attrib.get("type") or "").strip().lower()

        # ---- Wrappers: unwrap into the inner BSDF, attaching extras ----
        if btype == "twosided":
            inner = self._first_child_bsdf(elem)
            if inner is None:
                return Material(kind="diffuse", params={
                    "_twosided": Param(type="bool", name="_twosided", values=[True])
                })
            inner.params.setdefault(
                "_twosided", Param(type="bool", name="_twosided", values=[True]))
            return inner

        if btype in ("bumpmap", "normalmap"):
            inner = self._first_child_bsdf(elem)
            inner = inner if inner is not None else Material(kind="diffuse", params={})
            tex_or_path = self._first_texture_value(elem)
            if tex_or_path is not None:
                ptype, val = tex_or_path
                inner.params["normalmap"] = Param(
                    type=ptype, name="normalmap", values=[val])
            return inner

        if btype in ("mask",):
            inner = self._first_child_bsdf(elem)
            if inner is None:
                return Material(kind="diffuse", params={})
            return inner

        if btype in ("blendbsdf", "blend"):
            sub = self._first_child_bsdf(elem)
            if sub is None:
                return Material(kind="diffuse", params={})
            return sub

        # ---- Leaf BSDFs: collect Mitsuba params, translate to PBRT ----
        params: Dict[str, Param] = {}
        for child in elem:
            ctag = _strip_ns(child.tag).lower()
            if ctag == "ref":
                # Reference to a named texture — fold it into params if
                # it carries a 'name'
                ref_name = child.attrib.get("name")
                if ref_name:
                    rid = child.attrib.get("id", "")
                    if rid in self._texture_ids:
                        params[ref_name] = Param(
                            type="texture", name=ref_name, values=[rid])
                continue

            if ctag == "texture":
                # Inline texture inside a BSDF; register as anonymous and
                # reference it by its parameter name.
                tdef = self._build_texture(child)
                if tdef is not None:
                    self._register_texture(child.attrib.get("id"), tdef)
                    pname = child.attrib.get("name")
                    if pname:
                        params[pname] = Param(
                            type="texture", name=pname, values=[tdef.name])
                continue

            if ctag == "bsdf":
                # Nested BSDF inside a non-wrapper plugin (e.g. coatings):
                # we don't try to merge PBR coatings with their substrate;
                # just keep the outer plugin's params.
                continue

            scalar = _read_scalar_param(child, self._defaults)
            if scalar is not None:
                params[scalar[0]] = scalar[1]

        kind, params = self._mitsuba_bsdf_to_pbrt(btype, params)
        return Material(kind=kind, params=params)

    def _first_child_bsdf(self, elem: ET.Element) -> Optional[Material]:
        for child in elem:
            if _strip_ns(child.tag).lower() == "bsdf":
                return self._build_material(child)
            if _strip_ns(child.tag).lower() == "ref":
                rid = child.attrib.get("id")
                if rid and rid in self._material_ids:
                    return self._material_ids[rid]
        return None

    def _first_texture_value(self, elem: ET.Element
                              ) -> Optional[Tuple[str, str]]:
        """For bumpmap/normalmap wrappers: return the texture they wrap as
        either ('texture', tex_name) or ('string', filename)."""
        for child in elem:
            ctag = _strip_ns(child.tag).lower()
            if ctag == "texture":
                tdef = self._build_texture(child)
                if tdef is not None:
                    self._register_texture(child.attrib.get("id"), tdef)
                    return ("texture", tdef.name)
            elif ctag == "ref":
                rid = child.attrib.get("id")
                if rid and rid in self._texture_ids:
                    return ("texture", rid)
            elif ctag == "string" and child.attrib.get("name") == "filename":
                fname = _resolve_default(child.attrib.get("value", ""),
                                         self._defaults)
                if fname:
                    return ("string", fname)
        return None

    def _mitsuba_bsdf_to_pbrt(self, btype: str, params: Dict[str, Param]
                               ) -> Tuple[str, Dict[str, Param]]:
        """Translate Mitsuba parameter names to the PBRT names the
        existing exporters look up. The exporters' material code reads
        `reflectance`, `roughness`, `eta`, `normalmap`, etc."""

        out: Dict[str, Param] = dict(params)
        # Carry Mitsuba's "alpha" -> PBRT's "roughness" (both are α/GGX).
        if "alpha" in params and "roughness" not in params:
            out["roughness"] = Param(
                type=params["alpha"].type, name="roughness",
                values=list(params["alpha"].values))
        # Anisotropic alpha_u/alpha_v -> uroughness/vroughness
        if "alpha_u" in params and "uroughness" not in params:
            p = params["alpha_u"]
            out["uroughness"] = Param(p.type, "uroughness", list(p.values))
        if "alpha_v" in params and "vroughness" not in params:
            p = params["alpha_v"]
            out["vroughness"] = Param(p.type, "vroughness", list(p.values))

        if btype in ("diffuse", "roughdiffuse"):
            kind = "diffuse"
            self._rename(out, "reflectance", "reflectance")  # already named
        elif btype in ("dielectric", "roughdielectric"):
            kind = "dielectric"
            self._collapse_ior(out)
            self._rename(out, "specular_transmittance", "transmittance")
        elif btype == "thindielectric":
            kind = "thindielectric"
            self._collapse_ior(out)
        elif btype in ("conductor", "roughconductor"):
            kind = "conductor"
            self._rename(out, "specular_reflectance", "reflectance")
        elif btype in ("plastic", "roughplastic"):
            kind = "coateddiffuse"
            self._rename(out, "diffuse_reflectance", "reflectance")
        elif btype in ("principled", "principledthin"):
            kind = "principled"
            # `base_color` in Mitsuba is what PBRT/glTF exporters call the
            # base albedo; alias it onto `reflectance` so unaware code paths
            # still pick up the colour.
            self._rename(out, "base_color", "reflectance", keep_original=True)
        elif btype in ("difftrans", "diffusetransmission", "diffuse_transmission"):
            kind = "diffusetransmission"
            self._rename(out, "transmittance", "transmittance")
        elif btype in ("null", "passthrough", "transparent"):
            kind = "interface"
        else:
            # Unknown: best-effort fallback so we still get a colour.
            kind = "diffuse"
            self._rename(out, "base_color", "reflectance", keep_original=True)

        return kind, out

    @staticmethod
    def _rename(d: Dict[str, Param], old: str, new: str, keep_original: bool = False):
        if old not in d or new in d:
            return
        p = d[old]
        d[new] = Param(type=p.type, name=new, values=list(p.values))
        if not keep_original:
            del d[old]

    @staticmethod
    def _collapse_ior(d: Dict[str, Param]):
        """Mitsuba dielectrics expose ``int_ior`` / ``ext_ior`` (often as
        named materials like 'water'/'bk7'). Reduce these to a single
        relative ``eta`` so the exporters' KHR_materials_ior path picks
        it up.
        """
        if "eta" in d:
            return
        int_ior = _named_ior_to_float(d.get("int_ior"))
        ext_ior = _named_ior_to_float(d.get("ext_ior"))
        if int_ior is None and ext_ior is None:
            return
        if int_ior is None:
            int_ior = 1.5
        if ext_ior is None:
            ext_ior = 1.000277  # air
        if ext_ior <= 0:
            return
        eta = int_ior / ext_ior
        d["eta"] = Param(type="float", name="eta", values=[eta])

    # --------------------------------------------------------------
    # Textures
    # --------------------------------------------------------------

    def _build_texture(self, elem: ET.Element) -> Optional[TextureDef]:
        ttype = (elem.attrib.get("type") or "").strip().lower()
        params: Dict[str, Param] = {}
        for child in elem:
            scalar = _read_scalar_param(child, self._defaults)
            if scalar is not None:
                params[scalar[0]] = scalar[1]
            elif _strip_ns(child.tag).lower() == "transform":
                # bake uscale/vscale/udelta/vdelta from a 2D transform
                M = self._read_transform(child)
                # Best-effort: pull diagonal scale + translation in xy
                params.setdefault("uscale",
                                  Param("float", "uscale", [float(M[0, 0])]))
                params.setdefault("vscale",
                                  Param("float", "vscale", [float(M[1, 1])]))
                params.setdefault("udelta",
                                  Param("float", "udelta", [float(M[0, 3])]))
                params.setdefault("vdelta",
                                  Param("float", "vdelta", [float(M[1, 3])]))

        # Mitsuba 'bitmap' carries the file in 'filename'; the exporter
        # already looks for that.
        if ttype == "bitmap":
            return TextureDef(name="", tclass_value="spectrum",
                              tclass="imagemap", params=params)
        if ttype in ("checkerboard",):
            return TextureDef(name="", tclass_value="spectrum",
                              tclass="checkerboard", params=params)
        if ttype in ("constant", "uniform"):
            return TextureDef(name="", tclass_value="spectrum",
                              tclass="constant", params=params)
        # Procedural / unsupported: we still register it so refs resolve,
        # but the exporter will warn and skip.
        return TextureDef(name="", tclass_value="spectrum",
                          tclass=ttype or "constant", params=params)

    # --------------------------------------------------------------
    # Shapes
    # --------------------------------------------------------------

    def _process_shape(self, elem: ET.Element):
        stype = (elem.attrib.get("type") or "").strip().lower()

        if stype == "shapegroup":
            name = elem.attrib.get("id") or f"_mtsgrp_{self._next_group_id}"
            self._next_group_id += 1
            self.scene.objects.setdefault(name, [])
            saved = self._current_group
            self._current_group = name
            try:
                for child in elem:
                    if _strip_ns(child.tag).lower() == "shape":
                        self._process_shape(child)
            finally:
                self._current_group = saved
            return

        if stype == "instance":
            ref = None
            to_world = _identity()
            for child in elem:
                ctag = _strip_ns(child.tag).lower()
                if ctag == "ref":
                    ref = child.attrib.get("id") or ref
                elif ctag == "transform" and child.attrib.get("name") == "to_world":
                    to_world = self._read_transform(child)
            if ref is None:
                return
            self.scene.shapes.append(ShapeEntry(
                kind="instance", ctm=to_world, instance_name=ref))
            return

        # Regular shape: collect transform / material / emitter / params
        to_world = _identity()
        material: Optional[Material] = None
        area_light: Optional[Tuple[str, Dict[str, Param]]] = None
        flip_normals = False
        params: Dict[str, Param] = {}

        for child in elem:
            ctag = _strip_ns(child.tag).lower()
            if ctag == "transform" and child.attrib.get("name") == "to_world":
                to_world = self._read_transform(child)
            elif ctag == "boolean" and child.attrib.get("name") == "flip_normals":
                flip_normals = _parse_bool(_resolve_default(
                    child.attrib.get("value", "false"), self._defaults))
            elif ctag == "bsdf":
                material = self._build_material(child)
                # Anonymous inline BSDF — register so cache keys are stable.
                if material is not None:
                    self._register_material(child.attrib.get("id"), material)
            elif ctag == "ref":
                rid = child.attrib.get("id")
                if rid and rid in self._material_ids:
                    material = self._material_ids[rid]
            elif ctag == "emitter":
                etype = (child.attrib.get("type") or "").strip().lower()
                if etype == "area":
                    eparams: Dict[str, Param] = {}
                    for ec in child:
                        scalar = _read_scalar_param(ec, self._defaults)
                        if scalar is not None:
                            # Mitsuba's area emitter uses 'radiance'; PBRT
                            # exporters look for 'L'.
                            n, p = scalar
                            if n == "radiance":
                                p = Param(p.type, "L", list(p.values))
                                n = "L"
                            eparams[n] = p
                    area_light = ("diffuse", eparams)
            else:
                scalar = _read_scalar_param(child, self._defaults)
                if scalar is not None:
                    params[scalar[0]] = scalar[1]

        pbrt_type, pbrt_params, baked_ctm = self._mitsuba_to_pbrt_shape(
            stype, params, to_world)
        if pbrt_type is None:
            print(f"  warning: Mitsuba shape '{stype}' not supported, skipped")
            return

        entry = ShapeEntry(
            kind="shape",
            ctm=baked_ctm,
            material=material,
            area_light=area_light,
            reverse_orientation=flip_normals,
            shape_type=pbrt_type,
            shape_params=pbrt_params,
        )
        if self._current_group is not None:
            self.scene.objects[self._current_group].append(entry)
        else:
            self.scene.shapes.append(entry)

    def _mitsuba_to_pbrt_shape(self, stype: str, params: Dict[str, Param],
                                to_world: np.ndarray
                                ) -> Tuple[Optional[str], Dict[str, Param], np.ndarray]:
        if stype == "ply":
            return "plymesh", params, to_world
        if stype == "obj":
            # Use a lazy "objmesh" handler in the exporters (see
            # gltf_export._mesh_from_objmesh).
            face_normals = False
            fn = params.get("face_normals")
            if fn is not None and fn.values:
                face_normals = bool(fn.values[0])
            out = dict(params)
            out["_face_normals"] = Param("bool", "_face_normals", [face_normals])
            return "objmesh", out, to_world
        if stype == "sphere":
            # Mitsuba sphere takes a `center` (point) and `radius` in the
            # local frame. PBRT's `sphere` is centered at the local origin
            # with `radius`. Bake the center into the CTM so the existing
            # exporter math (which assumes origin-centred) works.
            center = (0.0, 0.0, 0.0)
            cp = params.get("center")
            if cp is not None and len(cp.values) >= 3:
                center = (float(cp.values[0]), float(cp.values[1]),
                          float(cp.values[2]))
            new_ctm = to_world @ _translate(*center)
            radius = float(params.get("radius").values[0]) if "radius" in params else 1.0
            return "sphere", {"radius": Param("float", "radius", [radius])}, new_ctm
        if stype == "rectangle":
            return "trianglemesh", _rectangle_mesh_params(), to_world
        if stype == "cube":
            return "trianglemesh", _cube_mesh_params(), to_world
        if stype == "disk":
            return "trianglemesh", _disk_mesh_params(), to_world
        if stype == "cylinder":
            p0 = (0.0, 0.0, 0.0)
            p1 = (0.0, 0.0, 1.0)
            radius = 1.0
            if "p0" in params and len(params["p0"].values) >= 3:
                p0 = tuple(params["p0"].values[:3])
            if "p1" in params and len(params["p1"].values) >= 3:
                p1 = tuple(params["p1"].values[:3])
            if "radius" in params:
                radius = float(params["radius"].values[0])
            return "trianglemesh", _cylinder_mesh_params(p0, p1, radius), to_world
        if stype in ("serialized",):
            print("  warning: Mitsuba 'serialized' shapes are not supported "
                  "(convert to PLY/OBJ first), skipped")
            return None, {}, to_world
        return None, {}, to_world

    # --------------------------------------------------------------
    # Global emitters (point/spot/directional/constant/envmap)
    # --------------------------------------------------------------

    def _process_global_emitter(self, elem: ET.Element):
        etype = (elem.attrib.get("type") or "").strip().lower()
        ctm = _identity()
        params: Dict[str, Param] = {}

        for child in elem:
            ctag = _strip_ns(child.tag).lower()
            if ctag == "transform" and child.attrib.get("name") == "to_world":
                ctm = self._read_transform(child)
            else:
                scalar = _read_scalar_param(child, self._defaults)
                if scalar is not None:
                    params[scalar[0]] = scalar[1]

        if etype == "point":
            pos = params.get("position")
            if pos is not None and len(pos.values) >= 3:
                ctm = ctm @ _translate(float(pos.values[0]),
                                       float(pos.values[1]),
                                       float(pos.values[2]))
            kind = "point"
            self._rename(params, "intensity", "I", keep_original=True)
        elif etype == "spot":
            kind = "spot"
            self._rename(params, "intensity", "I", keep_original=True)
            self._rename(params, "cutoff_angle", "coneangle")
            self._rename(params, "beam_width", "conedeltaangle")
        elif etype == "directional":
            kind = "distant"
            self._rename(params, "irradiance", "L", keep_original=True)
            # If direction is given explicitly, emit synthetic from/to so
            # the exporter's directional-light logic picks it up.
            d = params.get("direction")
            if d is not None and len(d.values) >= 3:
                params["from"] = Param("point3", "from", [0.0, 0.0, 0.0])
                params["to"] = Param("point3", "to", [
                    float(d.values[0]), float(d.values[1]), float(d.values[2])])
        elif etype == "constant":
            kind = "infinite"
            self._rename(params, "radiance", "L", keep_original=True)
        elif etype == "envmap":
            kind = "infinite"
            # The exporter doesn't bake env maps; record so the warning fires.
        else:
            print(f"  warning: Mitsuba emitter '{etype}' not supported, skipped")
            return

        self.scene.lights.append(LightDef(kind=kind, params=params, ctm=ctm))


# -------------------------------------------------------------------------
# Local helpers
# -------------------------------------------------------------------------

def _mitsuba_lookat(origin, target, up) -> np.ndarray:
    """Mitsuba <lookat> builds a *camera-to-world* matrix where the
    camera's local +Z axis points from origin to target. The Z-flip
    applied at sensor-export time turns this into the glTF-style
    camera-to-world (where -Z is forward), so what we return here is the
    raw Mitsuba placement.
    """
    o = np.asarray(origin, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    u = np.asarray(up, dtype=np.float64)
    fwd = t - o
    n = np.linalg.norm(fwd)
    if n == 0:
        return np.eye(4)
    fwd /= n
    side = np.cross(u, fwd)
    sn = np.linalg.norm(side)
    if sn < 1e-12:
        # up parallel to forward; pick something perpendicular
        side = np.array([1.0, 0.0, 0.0])
    else:
        side /= sn
    up_corrected = np.cross(fwd, side)
    M = np.eye(4)
    M[:3, 0] = side
    M[:3, 1] = up_corrected
    M[:3, 2] = fwd
    M[:3, 3] = o
    return M


def _named_ior_to_float(p: Optional[Param]) -> Optional[float]:
    """Return a float for a Mitsuba IOR parameter, looking up named
    materials when needed."""
    if p is None or not p.values:
        return None
    v = p.values[0]
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return _IOR_NAMES.get(v.strip().lower())
    return None


# Subset of Mitsuba's named-IOR table (sufficient for common scenes).
_IOR_NAMES = {
    "vacuum": 1.0,
    "air": 1.000277,
    "helium": 1.000036,
    "hydrogen": 1.000132,
    "water": 1.3330,
    "ethanol": 1.361,
    "diamond": 2.419,
    "polypropylene": 1.49,
    "polystyrene": 1.59,
    "bk7": 1.5046,
    "bromine": 1.661,
    "fused quartz": 1.458,
    "amber": 1.55,
    "pyrex": 1.470,
}


# ---- analytic shapes -> trianglemesh params -----------------------------

def _to_floats_param(name: str, vals) -> Param:
    """Wrap a numpy float array (or list) as a Param the exporter's
    `_mesh_from_trianglemesh` accepts."""
    flat = [float(x) for x in np.asarray(vals).reshape(-1).tolist()]
    return Param(type="point3" if name == "P" else
                 ("normal3" if name == "N" else
                  ("float" if name in ("uv", "st") else "integer")),
                 name=name, values=flat)


def _trianglemesh_params(P, indices, N=None, uv=None) -> Dict[str, Param]:
    out: Dict[str, Param] = {
        "P": _to_floats_param("P", P),
        "indices": _to_floats_param("indices", indices),
    }
    if N is not None:
        out["N"] = _to_floats_param("N", N)
    if uv is not None:
        out["uv"] = _to_floats_param("uv", uv)
    return out


def _rectangle_mesh_params() -> Dict[str, Param]:
    """Mitsuba 'rectangle': unit XY square spanning [-1,1]^2 with normal +Z."""
    P = np.array([
        [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
    ], dtype=np.float32)
    N = np.tile(np.array([0, 0, 1], dtype=np.float32), (4, 1))
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    return _trianglemesh_params(P, indices, N, uv)


def _cube_mesh_params() -> Dict[str, Param]:
    """Mitsuba 'cube': spans [-1,1]^3 with per-face normals."""
    # 6 faces * 4 verts each so per-face normals stay flat.
    faces = []
    for axis, sign in [(0, +1), (0, -1), (1, +1), (1, -1), (2, +1), (2, -1)]:
        n = [0, 0, 0]
        n[axis] = sign
        # Build the face by picking the two non-axis dims as the local frame.
        d1, d2 = [d for d in (0, 1, 2) if d != axis]
        verts = []
        for su in (-1, 1):
            for sv in (-1, 1):
                p = [0, 0, 0]
                p[axis] = sign
                p[d1] = su
                p[d2] = sv
                verts.append((p, n))
        # Reorder so the winding matches the outward normal (sign)
        if sign > 0:
            order = (0, 1, 3, 2)
        else:
            order = (0, 2, 3, 1)
        faces.append([verts[o] for o in order])

    P = []
    N = []
    uv = []
    indices = []
    for f in faces:
        base = len(P)
        for p, n in f:
            P.append(p)
            N.append(n)
        uv.extend([[0, 0], [1, 0], [1, 1], [0, 1]])
        indices.append([base, base + 1, base + 2])
        indices.append([base, base + 2, base + 3])
    return _trianglemesh_params(np.asarray(P, dtype=np.float32),
                                np.asarray(indices, dtype=np.uint32),
                                np.asarray(N, dtype=np.float32),
                                np.asarray(uv, dtype=np.float32))


def _disk_mesh_params(segments: int = 64) -> Dict[str, Param]:
    """Mitsuba 'disk': unit disk in the XY plane with normal +Z."""
    P = [[0.0, 0.0, 0.0]]
    uv = [[0.5, 0.5]]
    for i in range(segments):
        a = 2.0 * math.pi * i / segments
        x, y = math.cos(a), math.sin(a)
        P.append([x, y, 0.0])
        uv.append([0.5 + 0.5 * x, 0.5 + 0.5 * y])
    indices = []
    for i in range(segments):
        a = i + 1
        b = (i + 1) % segments + 1
        indices.append([0, a, b])
    N = [[0.0, 0.0, 1.0]] * len(P)
    return _trianglemesh_params(np.asarray(P, dtype=np.float32),
                                np.asarray(indices, dtype=np.uint32),
                                np.asarray(N, dtype=np.float32),
                                np.asarray(uv, dtype=np.float32))


def _cylinder_mesh_params(p0, p1, radius: float, segments: int = 32) -> Dict[str, Param]:
    """Tessellate a Mitsuba cylinder between p0 and p1."""
    p0v = np.asarray(p0, dtype=np.float64)
    p1v = np.asarray(p1, dtype=np.float64)
    axis = p1v - p0v
    h = float(np.linalg.norm(axis))
    if h == 0:
        # Degenerate — fall back to a tiny stub so we don't crash.
        axis = np.array([0.0, 0.0, 1.0])
        h = 1.0
    else:
        axis /= h
    # Build an orthonormal frame around the axis.
    if abs(axis[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    side = np.cross(ref, axis)
    side /= np.linalg.norm(side)
    other = np.cross(axis, side)

    P, N, uv = [], [], []
    for i in range(segments + 1):
        a = 2.0 * math.pi * i / segments
        c = math.cos(a) * radius
        s = math.sin(a) * radius
        offset = c * side + s * other
        normal = (c * side + s * other) / max(1e-12, radius)
        P.append((p0v + offset).tolist())
        N.append(normal.tolist())
        uv.append([i / segments, 0.0])
        P.append((p1v + offset).tolist())
        N.append(normal.tolist())
        uv.append([i / segments, 1.0])

    indices = []
    for i in range(segments):
        a = i * 2
        b = a + 1
        c = a + 2
        d = a + 3
        indices.append([a, c, d])
        indices.append([a, d, b])
    return _trianglemesh_params(np.asarray(P, dtype=np.float32),
                                np.asarray(indices, dtype=np.uint32),
                                np.asarray(N, dtype=np.float32),
                                np.asarray(uv, dtype=np.float32))


# -------------------------------------------------------------------------
# Public entrypoint
# -------------------------------------------------------------------------

def build_scene(path) -> Scene:
    path = Path(path).resolve()
    try:
        tree = ET.parse(path)
    except ET.ParseError as e:
        raise ValueError(f"Mitsuba XML parse error in {path}: {e}") from e
    root = tree.getroot()
    return MitsubaSceneBuilder(path).build(root)
