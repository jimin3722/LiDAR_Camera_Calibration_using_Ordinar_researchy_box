import numpy as np
import random 
import open3d as o3d
from utils import make_color


"""
Implementation of planar RANSAC.
Class for Plane object, which finds the equation of a infinite plane using RANSAC algorithim.
Call `fit(.)` to randomly take 3 points of pointcloud to verify inliers based on a threshold.
![Plane](https://raw.githubusercontent.com/leomariga/pyRANSAC-3D/master/doc/plano.gif "Plane")
---
"""

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

def find_plane(pcd, viz_mode) : 
    # pcd to numpy 
    pcd_np = np.asarray(pcd.points)
    # add normal vectors 
    normal_vecs = np.empty((0, 4), dtype=np.float32)
    # add planse pts 
    planes_pts = []
    # count number of planes 
    for idx in range(3): # config number of planes
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

def L2_plane(L1_equation, pts):
    n_points = pts.shape[0]
    max_iter = 500
    thresh = 0.01
    best_inliers = []
    best_eq = []
    best_inliers = []

    for _ in range(max_iter) : 
        id_samples = random.sample(range(0, n_points), 2)
        pt_samples = pts[id_samples]
        vecA = pt_samples[1, :] - pt_samples[0, :]
        vecB = L1_equation[:3] #+ pt_samples[0, :]
        vecC = np.cross(vecA, vecB)

        # print(np.dot(vecB,vecC)) 
        # # print(np.dot(L1_equation[:3],vecC)) 
        # print("----------------")

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
    
    # print(np.dot(L1_equation[:3],vecC)) 
    # print("----------------")
    
    return equation, inliers


def L3_plane(L1_equation, L2_equation, pts):
    n_points = pts.shape[0]
    max_iter = 500
    thresh = 0.02
    best_inliers = []
    best_eq = []
    best_inliers = []
    for _ in range(max_iter) : 
        id_samples = random.sample(range(0, n_points), 1)
        pt_samples = pts[id_samples]
        vecA = L1_equation[:3] #+ pt_samples[0, :]
        vecB = L2_equation[:3] #+ pt_samples[0, :]
        
        vecC = np.cross(vecA, vecB)

        # print("vecB",vecB) 
        # print("vecA",vecA) 
        # print(np.dot(vecA,vecC)) 
        # print("----------------")

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

def fit_orthogonal_planes(planes_idx, planes_pts, viz_mode = True) :
    print(f'선택받은 면 {planes_idx[0]}번째 면, {planes_idx[1]}번째 면, {planes_idx[2]}번째 면') 

    print(len(planes_pts[0]))
    print(len(planes_pts[1]))
    print(len(planes_pts[2]))
    
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
        #print("len(L2_inliers)",len(L2_inliers))
        
        L3_equation, L3_inliers = L3_plane(L1_equation, L2_equation, L3_pts)
        
        #print("len(L3_inliers)",len(L3_inliers))
        
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

    if viz_mode :
        visual()

    return square_equations, square_pts, square_equations
