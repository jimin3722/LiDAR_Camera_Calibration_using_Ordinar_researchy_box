import open3d as o3d
import numpy as np
from utils import * 
from ransac import *

def search_for_plane_candidnates(pcd):
    normal_vecs, planes_pts, inliers, outliers = find_plane(pcd, viz_mode = True)
    find_outliers(inliers = inliers, outliers = outliers, viz_mode = True)
    return normal_vecs, planes_pts, inliers, outliers


def box_refinement(cuboid_ptx, iter):
    # set initial parameters
    z_gamma = 0
    y_gamma = 0
    t = np.array([[0.0],[0.0],[0.0]])
    
    # save rotation and translation
    rot_z = []
    rot_y = []
    trans = []
    
    # set iteration rotation and translation 

    # find fit rot and trans 
    for _ in range(iter): 
        cuboid_ptx,rot_mat_y, y_gamma = y_rotation(cuboid_ptx, y_gamma)
        cuboid_ptx,rot_mat_z, z_gamma = z_rotation(cuboid_ptx, z_gamma)
        cuboid_ptx, t = translate(cuboid_ptx, t)   
        rot_y.append(rot_mat_y)
        rot_z.append(rot_mat_z)
        trans.append(t)
    return rot_z, rot_y, trans


def move_box(copy_cuboid_ptx, box_ptx ,rot_z, rot_y, trans, iter):
    for i in reversed(range(iter)):
        t = trans[i]
        rot_mat_y = rot_y[i]
        rot_mat_z = rot_z[i]
        box_ptx = rot_mat_y.T @ (rot_mat_z.T @(box_ptx + t))

    plot_box_corners(copy_cuboid_ptx, box_ptx, i)
    return box_ptx


def box_camera_extrinsic(box_ptx, camera_position):
    box_ptx = box_ptx.T # 6 x 3
    box_to_camera_extrinsic = pnp_solve(camera_position, box_ptx)
    return box_to_camera_extrinsic


def projection(box_to_camera_extrinsic, camera_intrinsic, box_ptx):
    ones = np.ones((1,6))
    box_ptx = np.concatenate((box_ptx, ones), axis = 0)

    camera_model =  box_to_camera_extrinsic @ box_ptx 
    project_points = camera_intrinsic @ camera_model
    project_points_sensor =  project_points / project_points[2, :]
    return project_points_sensor


def plot_projection_points():
    filename = './asset/images/test_img.png'
    img = cv2.imread(filename)
    green = (0, 255, 0)

    re_project_ptx = np.load('./camera_infromation/reporject.npy')
    re_project_ptx = re_project_ptx[:2, :]
    print(re_project_ptx)

    for i in range(6) :
        cv2.line(img, (int(re_project_ptx[0, i]), int(re_project_ptx[1, i])), (int(re_project_ptx[0, i]), int(re_project_ptx[1, i])), green, 5)

    cv2.imshow('image', img)
    cv2.waitKey()



if __name__ == '__main__' :
    # Set Realsense box corners in pixel coordinate 
    camera_position = np.array([[211, 222], [262, 286], [167, 226], [329, 115], [373, 178], [272, 135]], dtype=float)

    # set box size [meter]
    # case 1  
    box_ptx = np.array([[0.0, -0.21, 0.0, 0.0, -0.21, 0.0],   
                        [0.0, 0.0, 0.456, 0.0, 0.0, 0.456,],
                        [0.0,0.0,0.0 ,0.41,0.39,0.39]])                              

    # case 2 
    # box_ptx = np.array([[0.0, 0.21, 0.21, 0.0, 0.0, 0.0],
    #                     [0.0, 0.0, 0.0, 0.0, 0.456, 0.456],
    #                     [0.0, 0.0, 0.39, 0.39, 0.0, 0.39]])   

    # read raw point clouds from VLP-16
    # case 1
    pcd = o3d.io.read_point_cloud ('dataset/1669082752.657106637.pcd', format = 'pcd')    
    # case 2 
    # pcd = o3d.io.read_point_cloud ('dataset2/1669530287.516644239.pcd', format = 'pcd')
    
    # 1. Search for plane candindates
    normal_vecs, planes_pts, inliers, outliers = search_for_plane_candidnates(pcd) 

    # 2. Select 3 planes
    perpendicular_plane_idx = find_perpendicular_planes(normal_vecs) # It will be L1, L2 and L3 plane

    # 3. Box fitting 
    cuboid_normals, cuboid_ptx = fit_orthogonal_planes(perpendicular_plane_idx, planes_pts, viz_mode = True)
    
    # 4. Box refinement
    copy_cuboid_ptx = copy.deepcopy(cuboid_ptx)
    iter = 14
    rot_z, rot_y, trans = box_refinement(cuboid_ptx, iter)
    box_ptx = move_box(copy_cuboid_ptx, box_ptx ,rot_z, rot_y, trans, iter)

    # 5. Find Box to Camera extrinsic parameterc 
    box_to_camera_extrinsic = box_camera_extrinsic(box_ptx, camera_position) # 3x4
    camera_intrinsic = np.load('./camera_infromation/intrinsic.npy')

    # 6. Projection box points to realsense
    project_points_sensor = projection(box_to_camera_extrinsic, camera_intrinsic, box_ptx)
    plot_projection_points()
