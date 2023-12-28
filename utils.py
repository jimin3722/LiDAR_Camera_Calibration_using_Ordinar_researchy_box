import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from itertools import combinations
import random 
from numpy.linalg import inv
import math 
from numpy import linalg as LA
import copy

def make_color(pts, col):
    if col == 'red' or col == 1: 
        color = np.array([[255, 0, 0]])    
    elif col == 'blue' or col == 2: 
        color = np.array([[0, 0, 255]])
    elif col == 'green' or  col == 0 : 
        color = np.array([[0, 255, 0]])
    elif col == 'yellow' or col == 3: 
        color = np.array([[255, 255, 0]])
    elif col == 'magenta' or col == 4: 
        color = np.array([[255, 0, 255]])    
    elif col == 'cyan' or col == 5: 
        color = np.array([[0, 255, 255]])   
    elif col == 'black' or col == 10: 
        color = np.array([[0, 0, 0]])

    return np.repeat(color, pts.shape[0], axis = 0)


def y_rotation(cuboid_ptx, gamma): 
    
    L1_ptx, L2_ptx, L3_ptx = copy.deepcopy(cuboid_ptx[0]), copy.deepcopy(cuboid_ptx[1]) , copy.deepcopy(cuboid_ptx[2])
    
    # Make A
    L1_ptx[:, 1] = -L1_ptx[:, 1] # [x   -y] 
    temp = copy.deepcopy(L2_ptx[:, 0])
    L2_ptx[:, 0] = copy.deepcopy(L2_ptx[:, 1])
    L2_ptx[:, 1] = temp

    A = np.concatenate((L1_ptx, L2_ptx), axis = 0)
    A = A[:, :2] # [_, _, z] -> [_, _]
  
    # eigenvalues, fing gamma
    A_T_A = A.T @ A
    vals, vecs = LA.eig(A_T_A)
    
    cos, sin = vecs[:, np.argmin(vals)]
    gamma = np.arctan2(sin, cos) # radian 
    gamma = math.degrees(gamma) # degree

    # rot_mat = np.array([[np.cos(math.radians(gamma)), -np.sin(math.radians(gamma)), 0], 
    #                     [np.sin(math.radians(gamma)), np.cos(math.radians(gamma)), 0],
    #                     [0, 0, 1]])
    rot_mat = np.array([[np.cos(math.radians(gamma)), 0 ,np.sin(math.radians(gamma))],
                        [0, 1, 0], 
                        [-np.sin(math.radians(gamma)), 0 , np.cos(math.radians(gamma))],
                        ])
                      
    cuboid_ptx[0] = (rot_mat @ cuboid_ptx[0].T).T
    cuboid_ptx[1] = (rot_mat @ cuboid_ptx[1].T).T
    cuboid_ptx[2] = (rot_mat @ cuboid_ptx[2].T).T


    return cuboid_ptx , rot_mat, gamma

def z_rotation(cuboid_ptx, gamma): 

    L1_ptx, L2_ptx, L3_ptx = copy.deepcopy(cuboid_ptx[0]), copy.deepcopy(cuboid_ptx[1]) , copy.deepcopy(cuboid_ptx[2])
    L1_ptx[:, 1] = -L1_ptx[:, 1] # [x   -y] 

    temp = copy.deepcopy(L3_ptx[:, 0])
    L3_ptx[:, 0] = copy.deepcopy(L3_ptx[:, 1])
    L3_ptx[:, 1] = temp

    A = np.concatenate((L1_ptx, L3_ptx), axis = 0)
    A = A[:, :2] # [_, _, z] -> [_, _]
    x = np.array([[np.cos(math.radians(gamma))], 
                  [np.sin(math.radians(gamma))]]) 

    # eigenvalues, fing gamma
    A_T_A = A.T @ A
    vals, vecs = LA.eig(A_T_A)
    
    cos, sin = vecs[:, np.argmin(vals)]

    gamma = np.arctan2(sin, cos)
    gamma = math.degrees(gamma)
    # Target box rotation 

    rot_mat = np.array([[np.cos(math.radians(gamma)), -np.sin(math.radians(gamma)), 0],
                        [np.sin(math.radians(gamma)), np.cos(math.radians(gamma)),0], 
                        [0, 0, 1]])
    # rot_mat = np.array([[1.0, 0.0, 0.0], 
    #                     [0.0, np.cos(math.radians(gamma)), -np.sin(math.radians(gamma))],
    #                     [0.0, np.sin(math.radians(gamma)), np.cos(math.radians(gamma))], 
    #                     ])

    cuboid_ptx[0] = (rot_mat @ cuboid_ptx[0].T).T
    cuboid_ptx[1] = (rot_mat @ cuboid_ptx[1].T).T
    cuboid_ptx[2] = (rot_mat @ cuboid_ptx[2].T).T

    return cuboid_ptx, rot_mat, gamma

def translate(cuboid_ptx, t) :
    L1_ptx, L2_ptx, L3_ptx = copy.deepcopy(cuboid_ptx[0]), copy.deepcopy(cuboid_ptx[1]) , copy.deepcopy(cuboid_ptx[2])
    L1_len, L2_len, L3_len = L1_ptx.shape[0], L2_ptx.shape[0], L3_ptx.shape[0]
        

    v_1, v_2, v_3 =   np.array([[1, 0, 0]]), np.array([[0, 1, 0]]), np.array([[0, 0, 1]])
    #v_1, v_2, v_3 =   np.array([[1, 0, 0]]), np.array([[0, 0, 1]]), np.array([[0, 1, 0]])
    #v_1, v_2, v_3 =   np.array([[0, 1, 0]]), np.array([[0, 0, 1]]), np.array([[1, 0, 0]])
    #v_1, v_2, v_3 =   np.array([[0, 1, 0]]), np.array([[1, 0, 0]]), np.array([[0, 0, 1]])
    # v_1, v_2, v_3 =   np.array([[0, 0, 1]]), np.array([[0, 1, 0]]), np.array([[1, 0, 0]])
    # v_1, v_2, v_3 =   np.array([[0, 0, 1]]), np.array([[1, 0, 0]]), np.array([[0, 1, 0]])
    
    # make B
    B = np.empty((0, 3), float)
    for _ in range(L1_len) : 
        B = np.append(B, v_1, axis = 0)
    
    for _ in range(L2_len) : 
        B = np.append(B, v_2, axis = 0)
    
    for _ in range(L3_len) : 
        B = np.append(B, v_3, axis = 0)

    # make c 
    c = np.empty((0, 1), float)
    for i in range(L1_len) : 
        temp = v_1 @ L1_ptx[i, :].T
        c = np.append(c, [temp], axis = 0)

    for i in range(L2_len) : 
        temp = v_2 @ L2_ptx[i, :].T
        c = np.append(c, [temp], axis = 0)
    
    for i in range(L3_len) : 
        temp = v_3 @ L3_ptx[i, :].T
        c = np.append(c, [temp], axis = 0)
    
    
    t = inv(B.T @ B) @ B.T @ c

    cuboid_ptx[0] -= t.T 
    cuboid_ptx[1] -= t.T 
    cuboid_ptx[2] -= t.T 
    return cuboid_ptx, t


def plotPlane(plot, normal, d, values):
    # x, y, z
    x, y = np.meshgrid(values, values)
    z = (-normal[0] * x - normal[1] * y - d) * 1. / normal[2]
    # draw plot
    plot.plot_surface(x, y, z, alpha = 0.2)


def scatter_points(plot,x, y, z) :
    plot.scatter(x, y, z, s = 10)

def find_origin_point(cuboid_normals): 
    a, b, c, k1 = cuboid_normals[0][0],cuboid_normals[0][1],cuboid_normals[0][2],cuboid_normals[0][3] 
    d, e, f, k2  = cuboid_normals[1][0],cuboid_normals[1][1],cuboid_normals[1][2],cuboid_normals[1][3] 
    h, i , j, k3 = cuboid_normals[2][0],cuboid_normals[2][1],cuboid_normals[2][2],cuboid_normals[2][3]
    L1_vec = np.array([a, b, c ])
    L2_vec = np.array([d, e, f ])
    L3_vec = np.array([h, i, h ])


    A = np.array([[a, b, c], [d, e, f ], [h, i, j]])
    b = np.array([[-k1], [-k2], [-k3]])
    origin = (inv(A) @ b)


    cos_L1_L2 = L1_vec @ L2_vec/ (LA.norm(L1_vec) * LA.norm(L2_vec))
    cos_L1_L3 = L1_vec @ L3_vec/ (LA.norm(L1_vec) * LA.norm(L3_vec))
    cos_L2_L3 = L2_vec @ L3_vec/ (LA.norm(L2_vec) * LA.norm(L3_vec))
    print(cos_L1_L2)
    print(cos_L1_L3)
    print(cos_L2_L3)

    return origin



def find_outliers(inliers, outliers, viz_mode):
    pcd = o3d.geometry.PointCloud()
    outliers_color = make_color(outliers, 1) # outliers color 
    outliers_pts_color_concat = np.concatenate((outliers, outliers_color), axis = 1)

    inliers_color = make_color(inliers, 0)
    inliers[:, 3:] = inliers_color[:, :]
    total = np.concatenate((inliers, outliers_pts_color_concat), axis = 0)

    if viz_mode :
        pcd.points = o3d.utility.Vector3dVector(total[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(total[:, 3:])
        o3d.visualization.draw_geometries([pcd])


def find_perpendicular_planes(normal_vecs) : 
    num_planes = normal_vecs.shape[0]
    comb = list(combinations(range(num_planes), 3))
    best_error = 100
    best_comb = []
    for i in comb :
        print (i)
        error = np.abs(np.dot(normal_vecs[i[0]],normal_vecs[i[1]])) + np.abs(np.dot(normal_vecs[i[0]],normal_vecs[i[2]])) + np.abs(np.dot(normal_vecs[i[0]],normal_vecs[i[2]])) 
        print(error)
        if (error < best_error) : 
            best_error = error  
            best_comb = i
    perpendicular_plane_idx = best_comb 
    return perpendicular_plane_idx

def pnp_solve(camera, lidar):
    intrinsic = np.load('./camera_infromation/intrinsic.npy')
    dist = np.load('./camera_infromation/distortion_coefficient.npy')

    retval, rvec, tvec = cv2.solvePnP(lidar, camera, intrinsic, dist)
    R, _ = cv2.Rodrigues(rvec)
    RT = np.column_stack((R,tvec))

    return RT


def plot_box_corners(cuboid_ptx ,box_ptx, index):
    L1, L2, L3 = cuboid_ptx[0], cuboid_ptx[1], cuboid_ptx[2]
    cuboid_ptx_total = np.concatenate((L1, L2, L3), axis = 0)
    # bb = box_ptx.T # n x 3 
    bb = box_ptx # n x 3 

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_axis_off()

    # 그리드 사이즈 동일하게 조정
    max_range = np.max(cuboid_ptx_total)  # cuboid_ptx_total의 최대값
    ax.set_box_aspect([max_range, max_range, max_range])
    
    ax.set_xlabel('x [meter]')
    ax.set_ylabel('y [meter]')
    ax.set_zlabel('z [meter]')
    ax.view_init(30, index)
    ax.scatter(cuboid_ptx_total[:, 0], cuboid_ptx_total[:, 1], cuboid_ptx_total[:, 2], s = 5,color ='black')
    ax.scatter(bb[:, 0], bb[:, 1], bb[:, 2], s = 25, color = 'red')
    
    # for i in range(len(box_ptx)-1):
    #     if i == 0 :
    #         color = 'gray'
    #     if i == 1:
    #         color = 'blue'
    #     if i == 2:
    #         color = 'yellow'
    #     if i == 3:
    #         color = 'red'
    #     if i == 4:
    #         color = 'magenta'
    #     if i == 5:
    #         color = 'cyan'
    #     ax.scatter(bb[i, 0], bb[i, 1], bb[i, 2], s = 150, color = color)
    
    # plt.savefig('./result/case2/'+str(index)+'.jpg')
    plt.show()