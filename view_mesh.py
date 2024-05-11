import polyscope as ps
import open3d as o3d
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='', help='')
opt = parser.parse_args()

tmesh = o3d.io.read_triangle_mesh(opt.input)
P = tmesh.sample_points_uniformly(number_of_points=10000)
points = np.asarray(P.points)
points = points - np.mean(points, axis=0)
points = points / np.max(np.abs(points))

P.points = o3d.utility.Vector3dVector(points)
filename = 'data/'+opt.input.split('/')[-1].split('.')[0] + '.xyz'
#print(filename)

o3d.io.write_point_cloud(filename, P,  write_ascii=True)

#points = np.asarray(tmesh.vertices)
#triangles = np.asarray(tmesh.triangles)

ps.init()

mesh = ps.register_point_cloud("mesh", points)
ps.show()

