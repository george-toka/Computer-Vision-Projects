#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import glob
import time



## Set camera model
pinhole_calibration = True
fisheye_calibration = False

#base_folder = 'images_doorbell'
#base_folder = 'images_csi'
#base_folder = 'images_mobile_back_cam'
base_folder = '.'

file_entension = 'jpg'
#file_entension = 'png'

count = 0

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 0.001)

# For fisheye cameras
#calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

# For doorbell fisheye cameras
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW



# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
square_size = 2
objp = np.zeros((9*13,3), np.float32)

objp[:,:2] = np.mgrid[0:13,0:9].T.reshape(-1,2)
objp *= square_size

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(base_folder+'/calibration&distorted_imgs/*.'+file_entension)

print ('Images are read (#:' + str(len(images)) + ')')



for fname in images:    
    print (' Reading image =',count)
    img = cv2.imread(fname)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
    cv2.imshow('gray',gray)
    cv2.waitKey(100)

    if pinhole_calibration:
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (13,9),None)
#        ret, corners = cv2.findChessboardCorners(gray, (13,9),cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE)

    if fisheye_calibration:
        # Find the chess board corners for fisheye lens
        ret, corners = cv2.findChessboardCorners(gray, (13,9),cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (13,9), corners2,ret)

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img',img)
        cv2.imwrite(base_folder+'/chess_images/Image'+str(count)+'.'+file_entension,img)
        count += 1
        cv2.waitKey(100)

if pinhole_calibration:
    # For calibrating normal camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    print ('Pinhole Calibration results:')
    print ('mtx is:', mtx)
    print ('dist is:', dist)
    print ('rvecs is:', rvecs)
    print ('tvecs is:', tvecs)


if fisheye_calibration:
    # For Calibrating fisheye camera
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

    # Need to reshape the objpoints from Nx3 to Nx1x3 shape
    objpoints2 = np.expand_dims(np.asarray(objpoints), -2)


    rms, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(objpoints2,
                            imgpoints,
                            gray.shape[::-1],
                            K,
                            D,
                            rvecs,
                            tvecs,
                            calibration_flags,
                            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,
                            300, 1e-6))

    print ('\n')
    print ('mtx is:', mtx)
    print ('\n')
    print ('inverse mtx is:', np.linalg.inv(mtx))
    print ('\n')
    print ('dist are:', dist)
    print ('\n')
    print ('rvecs is:', rvecs)
    print ('\n')
    print ('tvecs is:', tvecs)
    print ('\n')


images = glob.glob(base_folder+'/calibration&distorted_imgs/*.'+file_entension)

l = 0
for fname2 in images:
    print ('undistorting images')
    img2 = cv2.imread(fname2)
    h,  w = img2.shape[:2]

    if pinhole_calibration:
        # Undistort for a normal camera
        newcameramtx, roi =cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
        cv2.namedWindow('undistorted', cv2.WINDOW_NORMAL)
        cv2.imshow('undistorted',dst)
        cv2.waitKey(100)
        time.sleep(2.0)

    if fisheye_calibration:
        # Undistort for a fisheye camera
        DIM = (int(w*1.1), int(h*1.1))
        DIM_new = DIM

        ### Use this K_new to keep more pixels from the original image
        K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, DIM_new, np.eye(3), balance=1.0)

        ### ... or use the K_new as K to get a cropped image
#        K_new = K

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_new, DIM, cv2.CV_16SC2)
        dst = cv2.remap(img2, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        cv2.namedWindow('undistorted', cv2.WINDOW_NORMAL)
        cv2.imshow('undistorted',dst)
        cv2.waitKey(100)
        time.sleep(2.0)


    cv2.imwrite(base_folder+'/undistorted_images/undistorted_Image'+str(l)+'.'+file_entension,dst)
    l += 1


if pinhole_calibration:
    #### For normal camera calibration only
    mean_error = 0
    tot_error = 0.0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        tot_error += error
    print ('tot_error is:', tot_error, 'len(objpoints is:', len(objpoints))
    print ("mean_error is: ", tot_error/len(objpoints))




cv2.destroyAllWindows()

