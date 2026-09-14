"""Prepare the official X-NavDP training split for the shared depth generator.

Only the authored /Root/Meshes scene geometry is exported; lighting/HDR domes,
materials and physics schemas do not define depth surfaces. USD composition and
instance transforms are resolved before the Z-up centimetre -> Y-up metre map.
The depth mesh uses the benchmark RTX defaults: no back-face culling and zero
subdivision refinement (USD doubleSided alone is ignored by RTX).
See https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_common.html
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess

import numpy as np

REPOSITORY = "InternRobotics/GRScenes"
REVISION = "4760b9031881c985e3582075cb2d8dbde1530a68"
SPLIT_REVISION = "7cee38a8d8308874d2b8488783c612f42060ac41"


def official_training_scenes(split: dict[str, list[str]]) -> set[str]:
    training = split["home_train"] + split["commercial_train"]
    evaluation = split["home_eval"] + split["commercial_eval"]
    if len(training) != len(set(training)) or set(training) & set(evaluation):
        raise ValueError("official scene split contains duplicates or evaluation leakage")
    return set(training)


def validate_scene_selection(scenes: list[dict], split: dict) -> None:
    allowed = official_training_scenes(split)
    if any(s["scene_id"] not in allowed for s in scenes):
        raise ValueError("generation may only use official training scenes")
    for scene in scenes:
        if scene["source_family"] != scene["scene_id"][:15]:
            raise ValueError("source family disagrees with the frozen ID grouping")
        if scene["scene_id"] not in split[scene["domain"] + "_train"]:
            raise ValueError("scene domain disagrees with official split")


def habitat_from_usd(up_axis: str, meters_per_unit: float) -> np.ndarray:
    if not np.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValueError("USD stage units must be finite and positive")
    rotation = {"Y": np.eye(3), "Z": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])}[up_axis]
    result = np.eye(4)
    result[:3, :3] = rotation * meters_per_unit
    return result


def export_geometry(source: Path, search_root: Path, destination: Path) -> dict:
    from pxr import Ar, Usd, UsdGeom
    import trimesh

    context = Ar.ResolverContext(Ar.DefaultResolverContext([str(search_root)]))
    stage = Usd.Stage.Open(str(source), context)
    # A partially composed stage can still render plausible but unsafe depths.
    errors = [str(error) for p in stage.Traverse() for error in p.GetPrimIndex().localErrors]
    if errors:
        raise ValueError("unresolved USD composition: " + "\n".join(errors))
    root = stage.GetPrimAtPath("/Root/Meshes")
    if not root:
        raise ValueError(f"missing GRScenes geometry root: {source}")
    units = UsdGeom.GetStageMetersPerUnit(stage)
    axis = str(UsdGeom.GetStageUpAxis(stage))
    basis = habitat_from_usd(axis, units)
    transforms = UsdGeom.XformCache()
    result = trimesh.Scene()
    triangles = 0
    material = trimesh.visual.material.PBRMaterial(doubleSided=True)
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        if mesh.ComputeVisibility() == UsdGeom.Tokens.invisible or mesh.ComputePurpose() == UsdGeom.Tokens.guide:
            continue
        authored_counts = mesh.GetFaceVertexCountsAttr().Get()
        if not authored_counts:
            # GRScenes authors Mesh containers with geometry on child prims.
            continue
        counts = np.asarray(authored_counts, dtype=np.int64)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if not np.all(counts == 3) or indices.size != counts.sum():
            raise ValueError(f"expected authored triangle geometry: {prim.GetPath()}")
        if not np.isfinite(points).all() or indices.min() < 0 or indices.max() >= len(points):
            raise ValueError(f"invalid USD mesh: {prim.GetPath()}")
        faces = indices.reshape(-1, 3)
        holes = set(mesh.GetHoleIndicesAttr().Get() or [])
        if holes:
            faces = faces[[i not in holes for i in range(len(faces))]]
        if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
            faces = faces[:, ::-1]
        # Gf matrices act on row vectors; trimesh/glTF matrices on columns.
        world = basis @ np.asarray(transforms.GetLocalToWorldTransform(prim)).T
        # Habitat/Recast does not repair winding for reflected glTF nodes.
        # A = (A F) F, F=diag(-1,1,1): bake F into vertices and reverse
        # winding, leaving a positive-determinant instance with identical points.
        if np.linalg.det(world[:3, :3]) < 0:
            points = points * [-1, 1, 1]
            faces = faces[:, ::-1]
            world = world @ np.diag([-1., 1., 1., 1.])
        digest = hashlib.sha256(points.tobytes() + faces.tobytes()).hexdigest()
        if digest not in result.geometry:
            geometry = trimesh.Trimesh(points, faces, process=False)
            geometry.visual = trimesh.visual.TextureVisuals(material=material)
            result.add_geometry(geometry, geom_name=digest,
                                node_name=str(prim.GetPath()), transform=world)
        else:
            result.graph.update(frame_to=str(prim.GetPath()), matrix=world, geometry=digest)
        triangles += len(faces)
    if not triangles:
        raise ValueError(f"empty scene: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".glb.partial")
    temporary.write_bytes(result.export(file_type="glb", include_normals=False))
    os.replace(temporary, destination)
    return {"triangles": triangles, "meshes": len(result.geometry),
            "usd_up_axis": axis, "usd_meters_per_unit": units,
            "habitat_from_usd": basis.tolist(), "bounds_m": result.bounds.tolist(),
            "size_bytes": destination.stat().st_size}


def converter_fingerprint() -> str:
    source = inspect.getsource(export_geometry) + inspect.getsource(habitat_from_usd)
    return hashlib.sha256(source.encode()).hexdigest()


def validate_prepared_assets(root: Path, scenes: list[dict]) -> None:
    manifest = json.loads((root / "download_manifest.json").read_text())
    split = json.loads((root / "scene_split.json").read_text())
    validate_scene_selection(scenes, split)
    if (manifest["repository"] != REPOSITORY or manifest["commit"] != REVISION
            or manifest["converter_sha256"] != converter_fingerprint()):
        raise ValueError("unexpected GRScenes revision")
    from curvenav.data_generation.assets import _validate_glb
    for scene in scenes:
        path = root / "stages" / (scene["scene_id"] + ".glb")
        _validate_glb(path)
        metadata = json.loads(path.with_suffix(".json").read_text())
        if metadata["size_bytes"] != path.stat().st_size:
            raise ValueError(f"prepared scene size mismatch: {path}")


def _prepare(config_path: Path, seven_zip: str) -> None:
    config = json.loads(config_path.read_text())
    project = config_path.resolve().parent.parent
    source_root = (project / config["usd_root"]).resolve()
    output = (project / config["asset_root"]).resolve()
    archive_root = (project / config["archive_root"]).resolve()
    split = json.loads((source_root / "scene_split.json").read_text())
    scenes = config["selected_scenes"]
    validate_scene_selection(scenes, split)
    output.mkdir(parents=True, exist_ok=True)
    (output / "scene_split.json").write_text(json.dumps(split, indent=2) + "\n")
    index = json.loads((archive_root / "manifest.json").read_text())
    if index["repository"] != REPOSITORY or index["revision"] != REVISION:
        raise ValueError("unexpected scene archive revision")
    for item in index["files"]:
        if (archive_root / item["path"]).stat().st_size != item["size"]:
            raise ValueError(f"incomplete archive: {item['path']}")
    for domain in ("home", "commercial"):
        pending = [f"{domain}_scenes/scenes/{scene['scene_id']}/start_result_navigation.usd"
                   for scene in scenes if scene["domain"] == domain
                   and not (source_root / f"{domain}_scenes/scenes/{scene['scene_id']}/start_result_navigation.usd").is_file()]
        if pending:
            subprocess.run([seven_zip, "x", "-y", "-bsp0",
                str(archive_root / "scenes/GRScenes-100" / (domain + "_scenes.zip")),
                *pending, "-o" + str(source_root)], check=True)
    fingerprint = converter_fingerprint()
    for scene in scenes:
        scene_id, domain = scene["scene_id"], scene["domain"]
        raw = source_root / (domain + "_scenes")
        layer = raw / "scenes" / scene_id / "start_result_navigation.usd"
        # Reuse the already installed shared object library, without modifying it.
        search_root = (project / config["shared_scene_root"] / ("internscenes_" + domain)).resolve()
        destination = output / "stages" / (scene_id + ".glb")
        metadata_path = destination.with_suffix(".json")
        identity = {"converter_sha256": fingerprint, "revision": REVISION,
                    "source_sha256": hashlib.sha256(layer.read_bytes()).hexdigest()}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if any(metadata[k] != v for k, v in identity.items()):
                raise ValueError(f"stale geometry cache; use a new asset_root: {destination}")
        else:
            metadata = {**identity, **export_geometry(layer, search_root, destination)}
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        (output / "stages" / (scene_id + ".stage_config.json")).write_text(json.dumps({
            "render_asset": scene_id + ".glb", "collision_asset": scene_id + ".glb",
            "up": [0, 1, 0], "front": [0, 0, -1], "units_to_meters": 1.0,
        }))
        scene_dir = output / "scenes"
        scene_dir.mkdir(exist_ok=True)
        (scene_dir / (scene_id + ".scene_instance.json")).write_text(json.dumps({
            "stage_instance": {"template_name": scene_id}, "object_instances": []}))
        print(json.dumps({"scene": scene_id, **metadata}), flush=True)
    (output / config["scene_dataset"]).write_text(json.dumps({
        "stages": {"paths": {".json": ["stages"]}},
        "scene_instances": {"paths": {".json": ["scenes"]}},
    }))
    (output / "download_manifest.json").write_text(json.dumps({
        "repository": REPOSITORY, "commit": REVISION, "split_revision": SPLIT_REVISION,
        "scenes": sorted(s["scene_id"] for s in scenes), "converter_sha256": fingerprint,
    }, indent=2) + "\n")
    validate_prepared_assets(output, scenes)


def prepare(config_path: Path, seven_zip: str) -> None:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    output = (config_path.parent.parent / config["asset_root"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _prepare(config_path, seven_zip)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--seven-zip", default="7zz")
    args = parser.parse_args()
    prepare(args.config, args.seven_zip)


if __name__ == "__main__":
    main()
