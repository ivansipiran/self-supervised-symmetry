import argparse
import open3d as o3d
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='', help='')
parser.add_argument('--output', type=str, default='', help='')

opt = parser.parse_args()

pc = o3d.io.read_point_cloud(opt.input)
pc = np.asarray(pc.points)
pc = pc - np.mean(pc, axis=0)
pc = pc / np.max(np.abs(pc))

#Apply a random rotation to the point cloud
R = np.random.randn(3,3)
U, S, V = np.linalg.svd(R)
R = U @ V
pc = pc @ R

pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc))
o3d.io.write_point_cloud(opt.output, pc,  write_ascii=True)

