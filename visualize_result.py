#!/usr/bin/env python3
"""
Small vispy visualizer for Any6D result directories.
Loads final_mesh_demo.obj, the *_pose.txt matrix, and pointcloud.ply, then displays them.
Usage: python visualize_result.py ./results/demo_mustard
"""

import argparse
import os
import glob
import numpy as np
import trimesh
import vispy.scene
from vispy.scene import visuals

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False


def find_pose_file(dirpath):
    """Find the single file ending with _pose.txt in dirpath."""
    pattern = os.path.join(dirpath, "*_pose.txt")
    files = glob.glob(pattern)
    if len(files) == 0:
        raise FileNotFoundError(f"No *_pose.txt file in {dirpath}")
    if len(files) > 1:
        raise FileNotFoundError(f"Multiple *_pose.txt files in {dirpath}: {files}")
    return files[0]


def load_data(dirpath):
    """Load mesh, pose matrix, and point cloud from a result directory."""
    dirpath = os.path.abspath(dirpath)
    mesh_path = os.path.join(dirpath, "final_mesh_demo.obj")
    pose_path = find_pose_file(dirpath)
    ply_path = os.path.join(dirpath, "pointcloud.ply")

    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"Point cloud not found: {ply_path}")

    # Load mesh (can be a Scene with multiple geometries)
    mesh_obj = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh_obj, trimesh.Scene):
        geoms = list(mesh_obj.geometry.values())
        if not geoms:
            raise ValueError(f"No geometry in {mesh_path}")
        mesh_obj = geoms[0]
    vertices = np.asarray(mesh_obj.vertices, dtype=np.float32)
    faces = np.asarray(mesh_obj.faces, dtype=np.uint32)

    # 4x4 pose (object -> camera/world); apply to mesh vertices
    pose = np.loadtxt(pose_path, dtype=np.float64)
    assert pose.shape == (4, 4), f"Expected 4x4 pose, got {pose.shape}"
    ones = np.ones((len(vertices), 1), dtype=np.float64)
    vertices_h = np.hstack([vertices, ones])
    vertices_t = (pose @ vertices_h.T).T[:, :3].astype(np.float32)

    # Point cloud
    if HAS_O3D:
        pcd = o3d.io.read_point_cloud(ply_path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
    else:
        from plyfile import PlyData
        ply = PlyData.read(ply_path)
        v = ply["vertex"]
        pts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

        if "red" in v.data.dtype.names:
            r, g, b = v["red"], v["green"], v["blue"]
            colors = np.stack([r, g, b], axis=1).astype(np.float32) / 255.0
        else:
            colors = None

    # invert y axis
    pts[:, 2] = -pts[:, 2]
    pts[:, 1] = -pts[:, 1]
        
    return {
        "vertices": vertices_t,
        "faces": faces,
        "points": pts,
        "point_colors": colors,
        "pose": pose,
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize Any6D result (mesh + pose + point cloud)")
    parser.add_argument("dir", type=str, help="Result directory (e.g. ./results/demo_mustard)")
    args = parser.parse_args()

    data = load_data(args.dir)

    canvas = vispy.scene.SceneCanvas(keys="interactive", show=True, title="Any6D Result")
    view = canvas.central_widget.add_view()
    view.bgcolor = (0.12, 0.12, 0.14, 1)

    # Mesh (posed) – slight blue tint
    mesh_visual = visuals.Mesh(
        vertices=data["vertices"],
        faces=data["faces"],
        color=(0.45, 0.55, 0.85, 0.9),
        shading="smooth",
    )
    view.add(mesh_visual)

    # Point cloud
    face_color = data["point_colors"] if data["point_colors"] is not None else (1.0, 0.85, 0.4, 0.6)
    scatter = visuals.Markers()
    scatter.set_data(
        data["points"],
        face_color=face_color,
        edge_width=0,
        size=2.0,
    )
    view.add(scatter)

    # Coordinate axes for orientation
    axis = visuals.XYZAxis(parent=view.scene)

    view.camera = "turntable"
    view.camera.depth_value = 1e4

    # Center view on data
    all_pts = np.vstack([data["vertices"], data["points"]])
    center = all_pts.mean(axis=0)
    radius = float(np.linalg.norm(all_pts - center, axis=1).max()) * 1.2
    view.camera.center = center
    view.camera.scale_factor = radius

    print("Controls: rotate = drag, zoom = scroll, pan = right-drag")
    vispy.app.run()


if __name__ == "__main__":
    main()
