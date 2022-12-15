import cv2
import numpy as np

filename = './asset/images/test_img.png'
img = cv2.imread(filename)
click_points = []

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
    # find box corers 
    cv2.setMouseCallback('image', onMouse)
