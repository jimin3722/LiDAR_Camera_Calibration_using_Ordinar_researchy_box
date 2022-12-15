import open3d as o3d
import numpy as np
import random 
from itertools import combinations
from numpy import linalg as LA
import copy
import matplotlib.pyplot as plt
from numpy.linalg import inv
import math 
import cv2

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
    
def plane(pts):
    max_iteration = 500
    inliers = []
    equation = []
    thresh = 0.01
    n_points = pts.shape[0]
    best_eq = []
    best_inliers = []

    for _ in range(max_iteration):

        # Samples 3 random points
        id_samples = random.sample(range(0, n_points), 3)
        pt_samples = pts[id_samples]

        vecA = pt_samples[1, :] - pt_samples[0, :]
        vecB = pt_samples[2, :] - pt_samples[0, :]

        # Now we compute the cross product of vecA and vecB to get vecC which is normal to the plane
        vecC = np.cross(vecA, vecB)

        vecC = vecC / np.linalg.norm(vecC)
        k = -np.sum(np.multiply(vecC, pt_samples[1, :]))
        plane_eq = [vecC[0], vecC[1], vecC[2], k]

        pt_id_inliers = []  # list of inliers ids
        dist_pt = (
            plane_eq[0] * pts[:, 0] + plane_eq[1] * pts[:, 1] + plane_eq[2] * pts[:, 2] + plane_eq[3]
        ) / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)

        # Select indexes where distance is biggers than the threshold
        pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]
        if len(pt_id_inliers) > len(best_inliers):
            best_eq = plane_eq
            best_inliers = pt_id_inliers
        inliers = best_inliers
        equation = best_eq

    return equation, inliers

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

def find_plane(pcd, viz_mode) : 
    # pcd to numpy 
    pcd_np = np.asarray(pcd.points)
    # add normal vectors 
    normal_vecs = np.empty((0, 4), dtype=np.float32)
    # add planse pts 
    planes_pts = []
    # count number of planes 
    for idx in range(3):
        equation, inliers = plane(pcd_np) # find a plane
        normal_vecs = np.append(normal_vecs, [equation], axis = 0) # add normal_vector

        find_plane = pcd_np[inliers] # n x 3
        planes_pts.append(find_plane)
        find_plane_color = make_color(find_plane, idx) # n x 3 
        find_plane_color_concat = np.concatenate((find_plane, find_plane_color), axis = 1) # n x 6

        if idx == 0:
            total = find_plane_color_concat # 첫번째 plane 은 넘어감 
        else :
            total = np.concatenate((total, find_plane_color_concat), axis = 0) # 두번째 plane 은 concat 

        pcd_np = np.delete(pcd_np, inliers, 0) # 찾은 면의 points 는 날려주기 
    
    if viz_mode :
        pcd_plane = o3d.geometry.PointCloud()
        pcd_plane.points = o3d.utility.Vector3dVector(total[:, :3])
        pcd_plane.colors = o3d.utility.Vector3dVector(total[:, 3:])
        o3d.visualization.draw_geometries([pcd_plane])

    return normal_vecs, planes_pts, total, pcd_np

def find_perpendicular_plane(normal_vecs) : 
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

def L2_plane(L1_equation, pts):
    n_points = pts.shape[0]
    max_iter = 50
    thresh = 0.01
    best_inliers = []
    best_eq = []
    best_inliers = []
    for _ in range(max_iter) : 
        id_samples = random.sample(range(0, n_points), 2)
        pt_samples = pts[id_samples]
        vecA = pt_samples[1, :] - pt_samples[0, :]
        vecB = L1_equation[:3] + pt_samples[0, :]
        vecC = np.cross(vecA, vecB)

        vecC = vecC / np.linalg.norm(vecC)
        k = -np.sum(np.multiply(vecC, pt_samples[1, :]))
        plane_eq = [vecC[0], vecC[1], vecC[2], k]

        pt_id_inliers = []  # list of inliers ids
        dist_pt = (
            plane_eq[0] * pts[:, 0] + plane_eq[1] * pts[:, 1] + plane_eq[2] * pts[:, 2] + plane_eq[3]
        ) / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)

        # Select indexes where distance is biggers than the threshold
        pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]
        if len(pt_id_inliers) > len(best_inliers):
            best_eq = plane_eq
            best_inliers = pt_id_inliers
        inliers = best_inliers
        equation = best_eq

    return equation, inliers

def L3_plane(L1_equation, L2_equation, pts):
    n_points = pts.shape[0]
    max_iter = 50
    thresh = 0.02
    best_inliers = []
    best_eq = []
    best_inliers = []
    for _ in range(max_iter) : 
        id_samples = random.sample(range(0, n_points), 1)
        pt_samples = pts[id_samples]
        vecA = L1_equation[:3] + pt_samples[0, :]
        vecB = L2_equation[:3] + pt_samples[0, :]
        vecC = np.cross(vecA, vecB)

        vecC = vecC / np.linalg.norm(vecC)
        k = -np.sum(np.multiply(vecC, pt_samples[0, :]))
        plane_eq = [vecC[0], vecC[1], vecC[2], k]

        pt_id_inliers = []  # list of inliers ids
        dist_pt = (
            plane_eq[0] * pts[:, 0] + plane_eq[1] * pts[:, 1] + plane_eq[2] * pts[:, 2] + plane_eq[3]
        ) / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)

        # Select indexes where distance is biggers than the threshold
        pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]
        if len(pt_id_inliers) > len(best_inliers):
            best_eq = plane_eq
            best_inliers = pt_id_inliers
        inliers = best_inliers
        equation = best_eq
    
    return equation, inliers

def find_fit_orthogonal_plane(planes_idx, planes_pts) :
    print(f'선택받은 면 {planes_idx[0]}번째 면, {planes_idx[1]}번째 면, {planes_idx[2]}번째 면') 
    
    L1_pts = planes_pts[planes_idx[0]]
    L2_pts = planes_pts[planes_idx[1]]
    L3_pts = planes_pts[planes_idx[2]]

    best_inliners = 0
    square_equations = []
    square_inliers = []
    # L1 에서 3 포인트 샘플링
    best_error = 100
    
    for _ in range(100) : 
        L1_equation, L1_inliers = plane(L1_pts)
        L2_equation, L2_inliers = L2_plane(L1_equation, L2_pts)
        L3_equation, L3_inliers = L3_plane(L1_equation, L2_equation, L3_pts)
        total_inliers =  L1_inliers.shape[0] + L2_inliers.shape[0]+ L3_inliers.shape[0]
        error = np.abs(np.dot(L1_equation[0],L1_equation[1])) + np.abs(np.dot(L2_equation[0],L2_equation[2])) + np.abs(np.dot(L3_equation[0],L3_equation[2])) 
        if total_inliers > best_inliners :
            best_inliners = total_inliers
            best_L1_inliers = L1_inliers
            best_L2_inliers = L2_inliers
            best_L3_inliers = L3_inliers
            square_equations = [L1_equation, L2_equation, L3_equation]
    
    L1_pts = L1_pts[best_L1_inliers]
    L2_pts = L2_pts[best_L2_inliers]
    L3_pts = L3_pts[best_L3_inliers]
    total = np.concatenate((L1_pts, L2_pts), axis = 0)
    total = np.concatenate((total, L3_pts), axis = 0)
    square_pts = [L1_pts, L2_pts, L3_pts]
    def visual():
        total_color = make_color(total, 0)
        pcd_plane = o3d.geometry.PointCloud()
        pcd_plane.points = o3d.utility.Vector3dVector(total)
        pcd_plane.colors = o3d.utility.Vector3dVector(total_color)
        o3d.visualization.draw_geometries([pcd_plane])

    return square_equations, square_pts

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
    cuboid_ptx[2] = (rot_mat @cuboid_ptx[2].T).T


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
    cuboid_ptx[2] = (rot_mat @cuboid_ptx[2].T).T

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

def plot_box_corners(cuboid_ptx ,box_ptx, index):
    L1, L2, L3 = cuboid_ptx[0], cuboid_ptx[1], cuboid_ptx[2]
    cuboid_ptx_total = np.concatenate((L1, L2, L3), axis = 0)
    bb = box_ptx.T # n x 3 

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_axis_off()
    
    ax.set_xlabel('x [meter]')
    ax.set_ylabel('y [meter]')
    ax.set_zlabel('z [meter]')
    ax.view_init(30, index)
    ax.scatter(cuboid_ptx_total[:, 0], cuboid_ptx_total[:, 1], cuboid_ptx_total[:, 2], s = 5,color ='black')
    for i in range(6):
        if i == 0 :
            color = 'gray'
        if i == 1:
            color = 'blue'
        if i == 2:
            color = 'yellow'
        if i == 3:
            color = 'red'
        if i == 4:
            color = 'magenta'
        if i == 5:
            color = 'cyan'
        ax.scatter(bb[i, 0], bb[i, 1], bb[i, 2], s = 150, color = color)
    
    # plt.savefig('./result/case2/'+str(index)+'.jpg')
    plt.show()

def pnp_solve(camera, lidar):
    intrinsic = np.load('./camera_infromation/intrinsic.npy')
    dist = np.load('./camera_infromation/distortion_coefficient.npy')

    retval, rvec, tvec = cv2.solvePnP(lidar, camera, intrinsic, dist)
    R, _ = cv2.Rodrigues(rvec)

    RT = np.column_stack((R,tvec))
    
    return RT

if __name__ == '__main__' :

    # Realsense corners in pixel coordinate
    camera_position = np.array([[211, 222], [262, 286], [167, 226], [329, 115], [373, 178], [272, 135]], dtype=float)


    # case 1
    pcd = o3d.io.read_point_cloud ('dataset/1669082752.657106637.pcd', format = 'pcd')    
    # case 2 
    # pcd = o3d.io.read_point_cloud ('dataset2/1669530287.516644239.pcd', format = 'pcd')

    normal_vecs, planes_pts, inliers, outliers = find_plane(pcd = pcd, viz_mode = False)
    find_outliers(inliers = inliers, outliers = outliers, viz_mode = False)
    perpendicular_plane_idx = find_perpendicular_plane(normal_vecs) # It will be L1, L2 and L3 plane
    cuboid_normals, cuboid_ptx = find_fit_orthogonal_plane(perpendicular_plane_idx, planes_pts)
    
    # copy cuboid original points
    copy_cuboid_ptx = copy.deepcopy(cuboid_ptx)
    
    # set box size [meter]
    # case 1  
    box_ptx = np.array([[0.0, -0.21, 0.0, 0.0, -0.21, 0.0],   
                        [0.0, 0.0, 0.456, 0.0, 0.0, 0.456,],
                        [0.0,0.0,0.0 ,0.41,0.39,0.39]])                              

    # case 2 
    # box_ptx = np.array([[0.0, 0.21, 0.21, 0.0, 0.0, 0.0],
    #                     [0.0, 0.0, 0.0, 0.0, 0.456, 0.456],
    #                     [0.0, 0.0, 0.39, 0.39, 0.0, 0.39]])                              


    # set initial parameters
    z_gamma = 0
    y_gamma = 0
    t = np.array([[0.0],[0.0],[0.0]])
    
    # save rotation and translation
    rot_z = []
    rot_y = []
    trans = []
    
    # set iteration rotation and translation 
    iter = 14

    # find fit rot and trans 
    for _ in range(iter): 
        cuboid_ptx,rot_mat_y, y_gamma = y_rotation(cuboid_ptx, y_gamma)
        cuboid_ptx,rot_mat_z, z_gamma = z_rotation(cuboid_ptx, z_gamma)
        cuboid_ptx, t = translate(cuboid_ptx, t)   
        rot_y.append(rot_mat_y)
        rot_z.append(rot_mat_z)
        trans.append(t)

    # move world box coordinate
    for i in reversed(range(iter)):
        t = trans[i]
        rot_mat_y = rot_y[i]
        rot_mat_z = rot_z[i]
        box_ptx = rot_mat_y.T @ (rot_mat_z.T @(box_ptx + t))

    # plot hypothesis box corners
    # for i in range(360):  
    #     plot_box_corners(copy_cuboid_ptx, box_ptx, i)
    plot_box_corners(copy_cuboid_ptx, box_ptx, i)

    box_ptx = box_ptx.T # 6 x 3 


    # fing box to camera R|T 
    box_to_camerea_extrinsic = pnp_solve(camera_position, box_ptx) # 3 x 4

    intrinsic = np.load('./camera_infromation/intrinsic.npy')

    box_ptx = box_ptx.T # 6 x 3 -> 3 x 6
    ones = np.ones((1,6))
    box_ptx = np.concatenate((box_ptx, ones), axis = 0)

    camera_model =  box_to_camerea_extrinsic @ box_ptx 
    project_points = intrinsic @ camera_model
    
    project_points_sensor =  project_points / project_points[2, :]


    np.save('./camera_infromation/reporject', project_points_sensor)