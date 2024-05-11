import polyscope as ps
import open3d as o3d
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_pc', type=str, default='', help='')
parser.add_argument('--input_mat', type=str, default='', help='')
opt = parser.parse_args()

ps.init()
pc1 = o3d.io.read_point_cloud(opt.input_pc)
matrix = np.loadtxt(opt.input_mat)

#pc2 = o3d.io.read_point_cloud('prueba.xyz')
pc = np.asarray(pc1.points)
cm = pc[:, :3].mean(axis=0)
print('Center:', cm)
pc[:, :3] = pc[:, :3] - cm

pc1 = pc# + np.array([0,0,-0.5])
pc2 = pc1@matrix# + np.array([0,0,0.5])

#pc2 = np.asarray(pc2.points)

print(pc1.shape)
#print(pc2.shape)


poc1 = ps.register_point_cloud("pc1",pc1)
poc2 = ps.register_point_cloud("pc2",pc2)

poc1.add_color_quantity("color", pc)
poc2.add_color_quantity("color", pc)


ps.show()
