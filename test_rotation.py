import argparse
import os
import numpy as np
import polyscope as ps

def translate(tx, ty, tz):
    return np.array([
        [1,0,0,tx],
        [0,1,0,ty],
        [0,0,1,tz],
        [0,0,0,1]], dtype = np.float32)

def rodrigues(normal, angle):
    normal = normal / np.linalg.norm(normal)
    #Rodrigues formula
    skew = lambda x: np.array([[0, -x[2], x[1]],
                               [x[2], 0, -x[0]],
                               [-x[1], x[0], 0]])
    R = np.eye(3) + np.sin(angle)*skew(normal) + (1-np.cos(angle))*np.matmul(skew(normal), skew(normal))
    
    return R


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='', help='')
parser.add_argument('--id', type=str, default='', help='')
opt = parser.parse_args()

points = np.load(os.path.join(opt.path, 'points'+str(opt.id)+'.npz'))['points']

with open(os.path.join(opt.path, 'points' + str(opt.id)+'_sym.txt')) as f:
    num_symmetries = int(f.readline().strip())

    for i in range(num_symmetries):
        L = f.readline().strip().split()
        if L[0]=='axis':
            L = L[1:]
            L = [float(x) for x in L]
            normal = np.array(L[3:6])
            point = np.array(L[:3])

print(f'Normal: {normal}, Point: {point}')
normal = normal / np.linalg.norm(normal)
print(np.linalg.norm(normal))

R1 = rodrigues(normal, np.pi/6)
R2 = rodrigues(normal, np.pi/3)
R3 = rodrigues(normal, np.pi/2)

#substract point from points
points = points - point

#Tranpose points
points = points.transpose(1,0)

#Apply rotation
points1 = np.matmul(R1, points)
points2 = np.matmul(R2, points)
points3 = np.matmul(R3, points)

#Transpose back
points = points.transpose(1,0)
points1 = points1.transpose(1,0)
points2 = points2.transpose(1,0)
points3 = points3.transpose(1,0)

#Add point
points = points + point
points1 = points1 + point
points2 = points2 + point
points3 = points3 + point

#Show in polyscope
ps.init()
pc = ps.register_point_cloud("pc",points)
poc1 = ps.register_point_cloud("pc1",points1)
poc2 = ps.register_point_cloud("pc2",points2)
poc3 = ps.register_point_cloud("pc3",points3)

ps.show()



