"""Step 3: view any .ply. Usage: python view_cloud.py cloud.ply [cameras.ply]"""
import sys, open3d as o3d
geoms = [o3d.io.read_point_cloud(p) for p in sys.argv[1:]]
o3d.visualization.draw_geometries(geoms)
