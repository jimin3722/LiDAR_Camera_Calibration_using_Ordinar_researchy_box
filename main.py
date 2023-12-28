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


def projection(box_to_camera_extrinsic, camera_intrinsic, box_ptx, cuboid_ptx):

    L1, L2, L3 = cuboid_ptx[0], cuboid_ptx[1], cuboid_ptx[2]
    cuboid_ptx_total = np.concatenate((L1, L2, L3), axis = 0)

    ones = np.ones((1, len(cuboid_ptx_total)))

    box_ptx = cuboid_ptx_total.T

    # ones = np.ones((1, 6))    
    box_ptx = np.concatenate((box_ptx, ones), axis = 0)

    # print(box_ptx)

    camera_model =  box_to_camera_extrinsic @ box_ptx 
    project_points = camera_intrinsic @ camera_model
    project_points_sensor =  project_points / project_points[2, :]
    # print(project_points_sensor)
    # print(len(project_points_sensor))
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


def find_intersection_and_lines(cuboid_equations):
    L1, L2, L3 = np.array(cuboid_equations[0]), np.array(cuboid_equations[1]), np.array(cuboid_equations[2])
    
    # 세 평면이 만나는 점 계산
    point_O = np.linalg.solve([L1[:3], L2[:3], L3[:3]], [-L1[3], -L2[3], -L3[3]])
    
    # 세 직선 모델 계산
    line1 = np.cross(L2[:3], L3[:3])
    line2 = np.cross(L3[:3], L1[:3])
    line3 = np.cross(L1[:3], L2[:3])

    lines = [line1, line2, line3]
    
    return point_O, lines


def find_dis(L_equ, L, _L):

    # Combine L and _L into a single list
    two_orthogonal_planes =  np.concatenate((L, _L),axis=0)
    distances = np.abs(np.dot(two_orthogonal_planes, L_equ[:3]) + L_equ[3]) / np.linalg.norm(L_equ[:3])
    distance = np.max(distances)

    return distance


def matching_length(d1,d2,d3,box_spec):

    d = [d1,d2,d3]

    sorted_l = sorted(box_spec)

    sorted_data = sorted(enumerate(d), key=lambda x: x[1])
    index_list = [x[0] for x in sorted_data]
    
    for i, v in enumerate(index_list):
        d[v] = sorted_l[i]

    return d


def find_point(d, l_equ, point_O):

    # Normalize the direction vector
    direction = l_equ / np.linalg.norm(l_equ)
    
    # Calculate the two points on the line
    point = point_O + direction * d
    _point = point_O - direction * d

    return point, _point


def check_valid_point(point, _point, L, _L, d):

    # Combine L and _L into a single list
    planes =  np.concatenate((L, _L),axis=0)
    
    # Calculate the average point of the combined planes
    planes_avg = np.mean(planes, axis=0)
    
    # Calculate the distances between the average point and the given points
    dist_point = np.linalg.norm(point - planes_avg)
    dist__point = np.linalg.norm(_point - planes_avg)
    
    # Compare the distances and determine the valid point
    if dist_point < dist__point:
        valid_point = point
    else:
        valid_point = _point
    
    return valid_point



def find_point_123(cuboid_ptx, point_O, lines, cuboid_equations, box_spec):

    l1_equ, l2_equ, l3_equ = lines[0], lines[1], lines[2]
    L1_equ, L2_equ, L3_equ = cuboid_equations[0], cuboid_equations[1], cuboid_equations[2]
    L1, L2, L3 = cuboid_ptx[0], cuboid_ptx[1], cuboid_ptx[2]

    d1 = find_dis(L1_equ, L2, L3)
    d2 = find_dis(L2_equ, L1, L3)
    d3 = find_dis(L3_equ, L2, L1)

    d1, d2, d3 = matching_length(d1,d2,d3,box_spec)

    point1, _point1 = find_point(d1, l1_equ, point_O)
    point2, _point2 = find_point(d2, l2_equ, point_O)
    point3, _point3 = find_point(d3, l3_equ, point_O)

    point1 = check_valid_point(point1, _point1, L2, L3, d1)
    point2 = check_valid_point(point2, _point2, L1, L3, d2)
    point3 = check_valid_point(point3, _point3, L2, L1, d3)

    point123 = [point1, point2, point3]

    return point123



if __name__ == '__main__' :
    # Set Realsense box corners in pixel coordinate 
    camera_position = np.array([[211, 222], [262, 286], [167, 226], [329, 115], [373, 178], [272, 135]], dtype=float)

    # set box size [meter]
    # case 1  
    box_ptx = np.array([[0.0, -0.21, 0.0, 0.0, -0.21, 0.0],   
                        [0.0, 0.0, 0.456, 0.0, 0.0, 0.456,],
                        [0.0,0.0,0.0 ,0.41,0.39,0.39]])       

    box_spec = [0.4, 0.22, 0.466]                       

    # case 2 
    # box_ptx = np.array([[0.0, 0.21, 0.21, 0.0, 0.0, 0.0],
    #                     [0.0, 0.0, 0.0, 0.0, 0.456, 0.456],
    #                     [0.0, 0.0, 0.39, 0.39, 0.0, 0.39]])   q

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
    cuboid_normals, cuboid_ptx, cuboid_equations  = fit_orthogonal_planes(perpendicular_plane_idx, planes_pts, viz_mode = True)
    
    point_O, lines= find_intersection_and_lines(cuboid_equations)

    point_123 = find_point_123(cuboid_ptx, point_O, lines, cuboid_equations, box_spec)

    points_123O = np.concatenate(([point_O], point_123), axis = 0)

    # rad1 = np.arccos( np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)) )
    # rad2 = np.arccos( np.dot(v3,v2) / (np.linalg.norm(v3)*np.linalg.norm(v2)) )
    # rad3 = np.arccos( np.dot(v1,v3) / (np.linalg.norm(v1)*np.linalg.norm(v3)) )

    # # 4. Box refinement
    # copy_cuboid_ptx = copy.deepcopy(cuboid_ptx)

    # plot_box_corners(copy_cuboid_ptx, np.array(points_123O), 14)

    # iter = 14
    # rot_z, rot_y, trans = box_refinement(cuboid_ptx, iter)
    # box_ptx = move_box(copy_cuboid_ptx, box_ptx ,rot_z, rot_y, trans, iter)

    # 5. Find Box to Camera extrinsic parameterc 
    box_to_camera_extrinsic = box_camera_extrinsic(box_ptx, camera_position) # 3x4
    camera_intrinsic = np.load('./camera_infromation/intrinsic.npy')

    # 6. Projection box points to realsense
    project_points_sensor = projection(box_to_camera_extrinsic, camera_intrinsic, box_ptx, copy_cuboid_ptx)
    
    
    filename = './asset/images/test_img.png'
    img = cv2.imread(filename)
    green = (0, 255, 0)
    re_project_ptx = project_points_sensor.T
    re_project_ptx = re_project_ptx[:, :2]

    for i in re_project_ptx :
        x, y = i
        cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
    
    cv2.imshow('image', img)

    cv2.waitKey()
    
    # plot_projection_points()
