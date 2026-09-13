"""Correct Scene-N1 face-corner normals incorrectly declared as vertex normals."""

import argparse

from pxr import Usd, UsdGeom


def repair_normals(stage):
    repaired = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        normals = mesh.GetNormalsAttr().Get()
        if normals is None or mesh.GetNormalsInterpolation() != UsdGeom.Tokens.vertex:
            continue
        if len(normals) == len(mesh.GetPointsAttr().Get()):
            continue
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if len(normals) != len(indices):
            raise ValueError(f"{prim.GetPath()}: normals match neither vertices nor face corners")
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        repaired.append(str(prim.GetPath()))
    return repaired


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene")
    args = parser.parse_args()
    stage = Usd.Stage.Open(args.scene)
    repaired = repair_normals(stage)
    stage.GetRootLayer().Save()
    print(f"Corrected normals interpolation on {len(repaired)} meshes", flush=True)
