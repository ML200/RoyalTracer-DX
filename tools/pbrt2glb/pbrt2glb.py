"""Scene converter - PBRT v4 / Mitsuba 3 (.pbrt, .xml, .zip) -> glTF (.glb)
and/or OBJ.

Usage:
    pbrt2glb [path/to/scene.{pbrt,xml,zip}] [--format glb|obj|both]

If no path is given, prompts on stdin and presents a format-selection
menu after the input is chosen. Outputs go into a `<stem>_export/`
subfolder next to the input.
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import mitsuba_scene
import pbrt_scene
from gltf_export import export_to_glb
from obj_export import export_to_obj
from pbrt_scene import Scene


_SCENE_EXTS = (".pbrt", ".xml")


# -------------------------------------------------------------------------
# Format detection
# -------------------------------------------------------------------------

def _is_pbrt(path: Path) -> bool:
    return path.suffix.lower() == ".pbrt"


def _is_mitsuba(path: Path) -> bool:
    return path.suffix.lower() == ".xml"


def _build_scene(path: Path) -> Scene:
    """Dispatch to the parser appropriate for the file extension."""
    if _is_pbrt(path):
        return pbrt_scene.build_scene(path)
    if _is_mitsuba(path):
        return mitsuba_scene.build_scene(path)
    raise ValueError(f"unrecognised input format (expected .pbrt or .xml): {path}")


# -------------------------------------------------------------------------
# Zip handling
# -------------------------------------------------------------------------

def _extract_zip(zip_path: Path) -> Tuple[Path, Path]:
    """Extract `zip_path` and locate a scene file (.pbrt or .xml).
    Returns (extract_dir, scene_path). Caller is responsible for
    cleaning up the extract_dir.
    """
    tmp = Path(tempfile.mkdtemp(prefix=f"pbrt2glb_{zip_path.stem}_"))
    print(f"  extracting zip to: {tmp}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)

    candidates: List[Path] = sorted(
        p for ext in _SCENE_EXTS for p in tmp.rglob(f"*{ext}"))
    if not candidates:
        raise FileNotFoundError(
            f"no .pbrt or .xml scene found inside {zip_path.name}")

    if len(candidates) == 1:
        return tmp, candidates[0]

    # Prefer a scene whose stem matches the zip stem
    stem_match = [c for c in candidates
                  if c.stem.lower() == zip_path.stem.lower()]
    if len(stem_match) == 1:
        return tmp, stem_match[0]

    print(f"\nMultiple scene files found in {zip_path.name}:")
    for i, p in enumerate(candidates):
        rel = p.relative_to(tmp)
        print(f"  [{i + 1}] {rel}")
    while True:
        try:
            raw = input(f"Choose [1-{len(candidates)}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            shutil.rmtree(tmp, ignore_errors=True)
            sys.exit(0)
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(candidates):
                return tmp, candidates[idx]
        except ValueError:
            pass
        print("  invalid choice, try again")


# -------------------------------------------------------------------------
# Format menu
# -------------------------------------------------------------------------

def _ask_output_formats() -> List[str]:
    print()
    print("Output formats:")
    print("  [1] glb  (binary glTF, embedded textures)")
    print("  [2] obj  (Wavefront OBJ + PBR-MTL + textures)")
    print("  [3] both")
    while True:
        try:
            raw = input("Choice [1-3] (default 1): ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
        if raw == "" or raw == "1":
            return ["glb"]
        if raw == "2":
            return ["obj"]
        if raw == "3":
            return ["glb", "obj"]
        print("  invalid choice, try again")


def _formats_from_arg(arg: Optional[str]) -> Optional[List[str]]:
    if arg is None:
        return None
    if arg == "glb":
        return ["glb"]
    if arg == "obj":
        return ["obj"]
    if arg == "both":
        return ["glb", "obj"]
    return None


# -------------------------------------------------------------------------
# Conversion
# -------------------------------------------------------------------------

def _convert_one(input_path: Path, formats: Optional[List[str]]):
    cleanup_dir: Optional[Path] = None

    if input_path.suffix.lower() == ".zip":
        cleanup_dir, scene_path = _extract_zip(input_path)
        anchor_dir = input_path.parent
        anchor_stem = input_path.stem
        actual_path = scene_path
    else:
        actual_path = input_path
        anchor_dir = input_path.parent
        anchor_stem = input_path.stem

    try:
        if not (_is_pbrt(actual_path) or _is_mitsuba(actual_path)):
            raise ValueError(
                f"unrecognised input format (expected .pbrt or .xml): {actual_path}")

        out_dir = anchor_dir / f"{anchor_stem}_export"
        out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        fmt_name = "PBRT" if _is_pbrt(actual_path) else "Mitsuba"
        print(f"  parsing ({fmt_name}): {actual_path}")
        scene = _build_scene(actual_path)
        t_parse = time.perf_counter() - t0
        print(f"  parsed in {t_parse:.2f}s "
              f"(shapes={len(scene.shapes)}, "
              f"objects={len(scene.objects)}, "
              f"materials={len(scene.materials_by_name)}, "
              f"textures={len(scene.textures)}, "
              f"lights={len(scene.lights)})")

        if formats is None:
            formats = _ask_output_formats()
        print(f"  output dir: {out_dir}")

        for f in formats:
            t1 = time.perf_counter()
            if f == "glb":
                target = out_dir / f"{anchor_stem}.glb"
                print(f"  -> writing GLB: {target.name}")
                export_to_glb(scene, target)
                size_mb = target.stat().st_size / (1024 * 1024)
                print(f"     done in {time.perf_counter() - t1:.2f}s "
                      f"({size_mb:.2f} MB)")
            elif f == "obj":
                print(f"  -> writing OBJ: {anchor_stem}.obj + .mtl + textures/")
                export_to_obj(scene, out_dir, anchor_stem)
                print(f"     done in {time.perf_counter() - t1:.2f}s")
            else:
                print(f"  -> skipping unknown format {f!r}")
    finally:
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)


# -------------------------------------------------------------------------
# CLI entry
# -------------------------------------------------------------------------

def _prompt_for_path() -> str:
    print()
    print("=== pbrt2glb ===  PBRT v4 / Mitsuba 3 -> glTF / OBJ converter")
    print("  Inputs : .pbrt, .xml, or .zip (containing a .pbrt or .xml)")
    print("  Outputs: .glb (binary glTF) and/or .obj + .mtl (Wavefront PBR)")
    print()
    print("Drop the input path below (with or without quotes).")
    print("Type 'q' to quit.\n")
    while True:
        try:
            raw = input("Input file: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if not raw:
            continue
        if raw.lower() in ("q", "quit", "exit"):
            sys.exit(0)
        if (raw.startswith('"') and raw.endswith('"')) or \
           (raw.startswith("'") and raw.endswith("'")):
            raw = raw[1:-1]
        return raw


def main():
    ap = argparse.ArgumentParser(
        prog="pbrt2glb",
        description="Convert PBRT v4 or Mitsuba 3 scenes to glTF (.glb) "
                    "and/or Wavefront OBJ.")
    ap.add_argument("input", nargs="?",
                    help=".pbrt, .xml, or .zip (containing one of those)")
    ap.add_argument("--format", "-f",
                    choices=("glb", "obj", "both"),
                    help="output format (skip the menu prompt)")
    ap.add_argument("--once", action="store_true",
                    help="convert once and exit (otherwise loops "
                         "in interactive mode)")
    args = ap.parse_args()

    inputs: List[str] = []
    if args.input:
        inputs.append(args.input)
    interactive = not args.input
    formats_override = _formats_from_arg(args.format)

    while True:
        if not inputs:
            if not interactive:
                break
            inputs.append(_prompt_for_path())

        raw = inputs.pop(0)
        path = Path(raw).expanduser().resolve()

        if not path.exists():
            print(f"  ERROR: no such file: {path}")
            if not interactive or args.once:
                return 1
            continue

        try:
            _convert_one(path, formats_override)
        except KeyboardInterrupt:
            print()
            return 0
        except Exception as exc:
            print(f"  ERROR: {exc}")
            if os.environ.get("PBRT2GLB_DEBUG"):
                traceback.print_exc()

        if not interactive or args.once:
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
