import numpy as np
import cv2 as cv
import glob

def mono_calibration(): 
    chessboardSize = (8, 6)
    frameSize = (640, 480)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2) * 26

    objpoints = []
    imgpoints = [] 

    images = glob.glob('check_pattern/*.jpg')

    for image in images:
        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

        if ret == True : 
            objpoints.append(objp)
            corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
        
            cv.drawChessboardCorners(img, chessboardSize, corners, ret)
            cv.imshow('img right', img)
            cv.waitKey(200)
        cv.destroyAllWindows()

    ret, mat, dist, rot, trans = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

    img = cv.imread('test_img.png')
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mat, dist, (w,h), 1, (w,h))
    undist_img = cv.undistort(img, mat, dist, None, newcameramtx)

    cv.imshow('undist ', undist_img)
    cv.waitKey(0)

    return newcameramtx, dist 

if __name__ == '__main__' :
    intrinsic, dist = mono_calibration() 

    # np.save('./camera_infromation/intrinsic', intrinsic)
    # np.save('./camera_infromation/distortion_coefficient', dist)

    