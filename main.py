import open3d as o3d
import numpy as np
from utils import * 
from ransac import *

def search_for_plane_candidnates(pcd, iter):
    normal_vecs, planes_pts, inliers, outliers = find_plane(pcd, iter, viz_mode = True)
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


def projection(box_to_camera_extrinsic, camera_intrinsic, cuboid_ptx):

    # L1, L2, L3 = cuboid_ptx[0], cuboid_ptx[1], cuboid_ptx[2]
    # cuboid_ptx_total = np.concatenate((L1, L2, L3), axis = 0)

    # ones = np.ones((1, len(cuboid_ptx_total)))
    # box_ptx = cuboid_ptx_total.T

    ones = np.ones((1, len(cuboid_ptx)))

    box_ptx = np.concatenate((cuboid_ptx.T, ones), axis = 0)

    camera_model =  box_to_camera_extrinsic @ box_ptx 

    camera_model = camera_model / camera_model[2,:] 
    print(camera_model)
    
    project_points = camera_intrinsic @ camera_model 
    #project_points_sensor =  project_points / project_points[2, :]
    # print(project_points_sensor)
    # print(len(project_points_sensor))
    return project_points#project_points_sensor


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


# def find_dis(L_equ, L, _L):

#     # Combine L and _L into a single list
#     new_L_equ = L_equ

#     two_orthogonal_planes =  np.concatenate((L, _L),axis=0)
#     print("len(two_orthogonal_planes):",len(two_orthogonal_planes))
#     distances = np.abs(np.dot(two_orthogonal_planes, L_equ[:3]) + L_equ[3]) / np.linalg.norm(L_equ[:3])
#     distance = np.mean(distances)

#     return distance

def find_dis(L_equ, L, _L, step_size=0.01, max_distance=100):

    minus_flag = False
    
    # 초기 평면의 방정식
    a, b, c, d = L_equ

    # 초기 평면 상의 점들
    points_L = np.array(L)
    points__L = np.array(_L)

    def check_inliers(moved_plane, points_L, points__L):
        # 이동한 평면 상의 인라이어 개수 계산
        inliers_L = np.sum(np.abs(np.dot(points_L, moved_plane[:3]) + moved_plane[3]) / np.linalg.norm(moved_plane[:3]) < 0.03)
        inliers__L = np.sum(np.abs(np.dot(points__L, moved_plane[:3]) + moved_plane[3]) / np.linalg.norm(moved_plane[:3]) < 0.03)
        return inliers_L + inliers__L

    if check_inliers(np.array([a, b, c, d+0.15]), points_L, points__L) < 10:
        minus_flag = True


    # 이동 거리 초기화
    distance = 0

    while distance < max_distance:  
        
        # 평면 이동
        if minus_flag:
            d -= step_size
        else:
            d += step_size
        
        moved_plane = np.array([a, b, c, d])

        # 인라이어 개수가 일정 값 이하이면 이동한 거리 반환
        if check_inliers(moved_plane, points_L, points__L) < 30:  # 예시 값, 조정 가능
            print("distance :",distance)
            return distance

        # 거리 갱신
        distance += step_size

    # 최대 이동 거리에 도달하면 None 반환
    return None


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
    
    # Calculate the distances
    #  between the average point and the given points
    dist_point = np.linalg.norm(point - planes_avg)
    dist__point = np.linalg.norm(_point - planes_avg)
    
    # Compare the distances and determine the valid point
    if dist_point < dist__point:
        valid_point = point
    else:
        valid_point = _point
    
    return valid_point



def find_point_123(cuboid_ptx, point_O, lines, cuboid_equations, box_spec):

    means = np.array([np.mean(cuboid_ptx[0], axis = 0), np.mean(cuboid_ptx[1], axis = 0), np.mean(cuboid_ptx[2], axis = 0)])
    print(means)

    f = np.argmax(means[:,1], axis=0)
    means[f] = [-9899,-999,-9999]

    s = np.argmax(means[:,2], axis=0)
    
    
    print(f)
    print(s)

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


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(param['points']) < 6:
            param['points'].append([x, y])
            print(f"Clicked ({x}, {y})")
        if len(param['points']) == 6:
            print("All points clicked. Final points:")
            for i, (px, py) in enumerate(param['points']):
                print(f"Point {i+1}: ({px}, {py})")
            cv2.destroyAllWindows()
            param['completed'] = True


def get_camera_coordinates(cv_image):
    param = {'points': [], 'completed': False}  # param 변수를 정의합니다.
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", click_event, param)  # param 변수를 인자로 전달합니다.
    cv2.imshow("Image", cv_image)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if param['completed']:
            break
    return param['points']

## ros에서 사용
def lidar_cam_cali(my_points, cv_image):

    # 예시 이미지를 로드하고 함수 호출
    # camera_position = np.array(get_camera_coordinates(cv_image),  dtype=float)

 
    # Set Realsense box corners in pixel coordinate 
    camera_position = np.array([[617, 280], [665, 409], 
                                [679, 169], [493, 292], 
                                [712, 289], [572, 185]], dtype=float)


    my_points = np.array(my_points)[:,:3]

    #camera_position = find_corner()
    
    #박스 폭, 너비, 높이 
    box_spec = [0.295, 0.46, 0.335]
    #box_spec = [0.26, 0.40, 0.30]
    iter = 3

    normal_vecs, planes_pts, inliers, outliers = search_for_plane_candidnates(my_points, iter)               

    # 2. Select 3 planes
    perpendicular_plane_idx = find_perpendicular_planes(normal_vecs) # It will be L1, L2 and L3 plane

    # 3. Box fitting 
    cuboid_normals, cuboid_ptx, cuboid_equations  = fit_orthogonal_planes(perpendicular_plane_idx, planes_pts, viz_mode = True)

    
    point_O, lines= find_intersection_and_lines(cuboid_equations)

    point_123 = find_point_123(cuboid_ptx, point_O, lines, cuboid_equations, box_spec)

    point_1_2 = ((point_123[0] - point_O) + (point_123[1] - point_O)) + point_O
    point_1_3 = ((point_123[0] - point_O) + (point_123[2] - point_O)) + point_O
    point_3_2 = ((point_123[2] - point_O) + (point_123[1] - point_O)) + point_O

    points_123O = np.concatenate(([point_O], point_123, [point_1_2, point_1_3, point_3_2]), axis = 0)

    # min_z_idx = np.argmin(points_123O[:,2])

    #points_123O = np.delete(points_123O, (min_z_idx), axis = 0)
    points_123O = np.delete(points_123O, (5), axis = 0)

    # # 4. Box refinement
    copy_cuboid_ptx = copy.deepcopy(cuboid_ptx)
    
    L1, L2, L3 = copy_cuboid_ptx[0], copy_cuboid_ptx[1], copy_cuboid_ptx[2]
    cuboid_ptx_total = np.concatenate((L1, L2, L3), axis = 0)

    # plot_box_corners(copy_cuboid_ptx, np.array(points_123O), 14)
    # plot_box_corners(cuboid_ptx_total, np.array(points_123O), 14)
    plot_box_corners(my_points, np.array(points_123O), 14)

    # iter = 14
    # rot_z, rot_y, trans = box_refinement(cuboid_ptx, iter)
    # box_ptx = move_box(copy_cuboid_ptx, box_ptx ,rot_z, rot_y, trans, iter)
    # print(box_ptx)

    # 5. Find Box to Camera extrinsic parameterc 
    box_to_camera_extrinsic = box_camera_extrinsic(points_123O.T, camera_position) # 3x4
    #camera_intrinsic = np.load('./camera_infromation/intrinsic.npy')
    
    s = 1.005#0.995#0.975
    w = 1280
    h = 720
    fov = 60 * s

    fc_y = h/(2*np.tan(np.deg2rad(fov/2)))
    fc_x = fc_y #* 0.935
    
    cx = w/2
    cy = h/2
    
    camera_intrinsic = np.array([[fc_x,  0,   cx],
                                [  0,   fc_y, cy],
                                [  0,    0,    1]])
    

    # 6. Projection box points to realsense
    
    project_points_sensor = projection(box_to_camera_extrinsic, camera_intrinsic, my_points)


    #project_points_sensor = projection(box_to_camera_extrinsic, camera_intrinsic, cuboid_ptx_total)
    
    img = cv_image
    re_project_ptx = project_points_sensor.T
    re_project_ptx = re_project_ptx[:, :2]

    for i in re_project_ptx :
        x, y = i
        cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)

    project_points_sensor = projection(box_to_camera_extrinsic, camera_intrinsic, points_123O)
    
    img = cv_image
    green = (0, 255, 0)
    re_project_ptx = project_points_sensor.T
    re_project_ptx = re_project_ptx[:, :2]

    for i in re_project_ptx :
        x, y = i
        cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)

    cv2.imshow('image', img)


    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        

    # 이미지 표시 종료
    cv2.destroyAllWindows()


if __name__ == '__main__' :
    # Set Realsense box corners in pixel coordinate 
    camera_position = np.array([[211, 222], [262, 286], 
                                [167, 226], [329, 115], 
                                [373, 178], [272, 135]], dtype=float)
 
    # set box size [meter]
    # case 1  
    box_ptx = np.array([[0.0, -0.21, 0.0, 0.0, -0.21, 0.0],   
                        [0.0, 0.0, 0.456, 0.0, 0.0, 0.456,],
                        [0.0,0.0,0.0 ,0.41,0.39,0.39]])       

    box_spec = [0.40, 0.22, 0.466]                       

    # case 2 
    # box_ptx = np.array([[0.0, 0.21, 0.21, 0.0, 0.0, 0.0],
    #                     [0.0, 0.0, 0.0, 0.0, 0.456, 0.456],
    #                     [0.0, 0.0, 0.39, 0.39, 0.0, 0.39]])   q

    # read raw point clouds from VLP-16
    # case 1
    path = "/home/jimin/ros2_ws/src/lidar_camera_box/lidar_camera_box/LiDAR_Camera_Calibration_using_Ordinary_box/dataset"
    pcd_path = os.path.join(path, '1669082752.657106637.pcd')
    pcd = o3d.io.read_point_cloud (pcd_path, format = 'pcd')    
    # case 2 
    # pcd = o3d.io.read_point_cloud ('dataset2/1669530287.516644239.pcd', format = 'pcd')
    
    # 1. Search for plane candindates
    # pcd to numpy 
    pcd_np = np.asarray(pcd.points)
    iter = 3
    normal_vecs, planes_pts, inliers, outliers = search_for_plane_candidnates(pcd_np, iter) 

    # 2. Select 3 planes
    perpendicular_plane_idx = find_perpendicular_planes(normal_vecs) # It will be L1, L2 and L3 plane

    # 3. Box fitting 
    cuboid_normals, cuboid_ptx, cuboid_equations  = fit_orthogonal_planes(perpendicular_plane_idx, planes_pts, viz_mode = True)
    
    point_O, lines= find_intersection_and_lines(cuboid_equations)

    point_123 = find_point_123(cuboid_ptx, point_O, lines, cuboid_equations, box_spec)

    point_1_2 = ((point_123[0] - point_O) + (point_123[1] - point_O)) + point_O
    point_1_3 = ((point_123[0] - point_O) + (point_123[2] - point_O)) + point_O
    point_3_2 = ((point_123[2] - point_O) + (point_123[1] - point_O)) + point_O
    print("sdaf",np.dot((point_123[0] - point_O) , (point_123[1] - point_O)))

    points_123O = np.concatenate(([point_O], point_123, [point_1_2, point_1_3, point_3_2]), axis = 0)


    min_z_idx = np.argmin(points_123O[:,2])

    points_123O = np.delete(points_123O, (min_z_idx), axis = 0)

    # rad1 = np.arccos( np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)) )
    # rad2 = np.arccos( np.dot(v3,v2) / (np.linalg.norm(v3)*np.linalg.norm(v2)) )
    # rad3 = np.arccos( np.dot(v1,v3) / (np.linalg.norm(v1)*np.linalg.norm(v3)) )qq

    # # 4. Box refinement
    copy_cuboid_ptx = copy.deepcopy(cuboid_ptx)
    
    L1, L2, L3 = copy_cuboid_ptx[0], copy_cuboid_ptx[1], copy_cuboid_ptx[2]
    cuboid_ptx_total = np.concatenate((L1, L2, L3), axis = 0)

    # plot_box_corners(copy_cuboid_ptx, np.array(points_123O), 14)
    plot_box_corners(cuboid_ptx_total, np.array(points_123O), 14)

    # iter = 14
    # rot_z, rot_y, trans = box_refinement(cuboid_ptx, iter)
    # box_ptx = move_box(copy_cuboid_ptx, box_ptx ,rot_z, rot_y, trans, iter)
    # print(box_ptx)

    # 5. Find Box to Camera extrinsic parameterc 
    box_to_camera_extrinsic = box_camera_extrinsic(points_123O.T, camera_position) # 3x4
    #camera_intrinsic = np.load('./camera_infromation/intrinsic.npy')
    
    s = 0.96
    w = 640
    h = 480
    fov = 40.5 * s

    fc_x = h/(2*np.tan(np.deg2rad(fov/2)))
    fc_y = fc_x
    cx = w/2
    cy = h/2
    
    camera_intrinsic = np.array([[fc_x,  0,   cx],
                                [  0,   fc_y, cy],
                                [  0,    0,    1]])
    
    # 6. Projection box points to realsense
    
    project_points_sensor = projection(box_to_camera_extrinsic, camera_intrinsic, cuboid_ptx_total)

    # project_points_sensor = projection(box_to_camera_extrinsic, camera_intrinsic, box_ptx, np.asarray(pcd.points))

    path = "/home/jimin/ros2_ws/src/lidar_camera_box/lidar_camera_box/LiDAR_Camera_Calibration_using_Ordinary_box"
    img_path = os.path.join(path, 'asset/images/test_img.png')
    img = cv2.imread(img_path)

    green = (0, 255, 0)
    re_project_ptx = project_points_sensor.T
    re_project_ptx = re_project_ptx[:, :2]

    for i in re_project_ptx :
        x, y = i
        cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)
    
    cv2.imshow('image', img)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # 이미지 표시 종료
    cv2.destroyAllWindows()
    
    # plot_projection_points()
