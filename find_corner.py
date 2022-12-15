import cv2
import numpy as np

filename = 'test_img.png'
img = cv2.imread(filename)

click_points = []
green = (0, 255, 0)

def onMouse(event, x, y, flags, param) :
    if event == cv2.EVENT_LBUTTONDOWN :
        print('왼쪽 마우스 클릭 했을 때 좌표 : ', x, y)
        click_points.append([x, y])
    elif event == cv2.EVENT_LBUTTONUP :
        print('왼쪽 마우스 클릭 땠을 때 좌표 : ', x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        print('현재 이동하는 좌표 : ', x, y)
        if flags & cv2.EVENT_FLAG_LBUTTON :
            cv2.circle(img, (x,y), 5, (0,0,255), -1)
            cv2.imshow('image', img)



if __name__ == '__main__' :

    # cv2.imshow('image', img)
    # cv2.setMouseCallback('image', onMouse)
    # 


    re_project_ptx = np.load('./camera_infromation/reporject.npy')
    re_project_ptx = re_project_ptx[:2, :]
    print(re_project_ptx)

    for i in range(6) :
        cv2.line(img, (int(re_project_ptx[0, i]), int(re_project_ptx[1, i])), (int(re_project_ptx[0, i]), int(re_project_ptx[1, i])), green, 5)

    cv2.imshow('image', img)
    cv2.waitKey()

