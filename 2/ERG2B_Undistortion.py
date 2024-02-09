#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import cv2



def pinhole_point_to_pixel (camera_matrix, distortion, pcd):

    k_1 = distortion[0]
    k_2 = distortion[1]
    p_1 = distortion[2]
    p_2 = distortion[3]
    k_3 = distortion[4]


    f_x = camera_matrix[0][0]
    f_y = camera_matrix[1][1]


    x_0 = camera_matrix[0][2]
    y_0 = camera_matrix[1][2]

    pix_x = np.zeros((pcd.shape[0], 1), dtype=np.float32)
    pix_y = np.zeros((pcd.shape[0], 1), dtype=np.float32)

    for idx,point in enumerate(pcd):

        # Look into : https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a
        # For a full explanation of the below equations

        # Compute normalized 3D point
        x = point[0]/point[2]
        y = point[1]/point[2]

        # Compute x-y distance
        r = np.sqrt(np.power(x,2) + np.power(y,2))

        # Compute the radial distortion correction term
        radial_correction_term = 1 + k_1*np.power(r,2) + k_2*np.power(r,4) + k_3*np.power(r,6)

        # Compute the tangental distortion dx, dy correction terms
        dx = 2*p_1*x*y  + p_2*(np.power(r,2) + 2*np.power(x,2))
        dy = p_1*(np.power(r,2) + + 2*np.power(y, 2)) + 2*p_2*x*y

        # Correct the normalized 3D point, according to radial and tangential correction terms
        x_corrected = x * radial_correction_term + dx
        y_corrected = y * radial_correction_term + dy

        # Convert the normalized 3D point, into pixel coordinates using the intrinsic calibration fx,fy, cx,cy
        x_corrected_pixel = x_corrected * f_x + x_0
        y_corrected_pixel = y_corrected * f_y + y_0

        pix_x[idx] = x_corrected_pixel
        pix_y[idx] = y_corrected_pixel


    return pix_x, pix_y

#1 for img0-img1, 2 for img2-img3 else my images
select = 1
img_select = 0

if(select==1):
    camera_matrix = np.array( [[2.12195577e+03, 0.00000000e+00, 9.52231504e+02],
                           [0.00000000e+00, 2.12404863e+03, 1.26061347e+03],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]] , dtype='f')

    distortion = np.array([ 3.19471785e-01, -1.46863989e+00, -7.27334102e-03,  4.17840306e-04,  2.39289521e+00], dtype='f')

    if(img_select==1):
    # Format of 3D point according to in X,Y,Z axes, in camera frame 
        pcd = np.array([[4, -2.5, 23],
               [4, -0.5,  23],
               [4, 1.5, 23],
               [4, 3.5, 23],
               [4, 5.5, 23],
               [4, 7.5, 23],
               [4, 9.5, 23],
               [4, 11.5, 23]])
        imgName = '0Z_-7X.jpg'
    else:
        pcd = np.array([[-3, -2.5, 23],
               [-3, -0.5,  23],
               [-3, 1.5, 23],
               [-3, 3.5, 23],
               [-3, 5.5, 23],
               [-3, 7.5, 23],
               [-3, 9.5, 23],
               [-3, 11.5, 23]])
        imgName = '0Z_0X.jpg'

elif(select==2):
    if(img_select==0):
        camera_matrix = np.array( [[2.12195577e+03, 0.00000000e+00, 9.52231504e+02],
                           [0.00000000e+00, 2.12404863e+03, 1.26061347e+03],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]] , dtype='f')

        distortion = np.array([ 3.19471785e-01, -1.46863989e+00, -7.27334102e-03,  4.17840306e-04,  2.39289521e+00], dtype='f')
 
    # Format of 3D point according to in X,Y,Z axes, in camera frame 
        pcd = np.array([[4, -2.5, 23],
               [4, -0.5,  23],
               [4, 1.5, 23],
               [4, 3.5, 23],
               [4, 5.5, 23],
               [4, 7.5, 23],
               [4, 9.5, 23],
               [4, 11.5, 23]])
        imgName = 'img2.png'
    else:
        camera_matrix = np.array( [[2.12195577e+03, 0.00000000e+00, 9.52231504e+02],
                           [0.00000000e+00, 2.12404863e+03, 1.26061347e+03],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]] , dtype='f')

        distortion = np.array([ 3.19471785e-01, -1.46863989e+00, -7.27334102e-03,  4.17840306e-04,  2.39289521e+00], dtype='f')

        pcd = np.array([[-3, -2.5, 23],
               [-3, -0.5,  23],
               [-3, 1.5, 23],
               [-3, 3.5, 23],
               [-3, 5.5, 23],
               [-3, 7.5, 23],
               [-3, 9.5, 23],
               [-3, 11.5, 23]])
        imgName = 'img3.png'

else:
    camera_matrix = np.array( [[2.12195577e+03, 0.00000000e+00, 9.52231504e+02],
                           [0.00000000e+00, 2.12404863e+03, 1.26061347e+03],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]] , dtype='f')

    distortion = np.array([ 3.19471785e-01, -1.46863989e+00, -7.27334102e-03,  4.17840306e-04,  2.39289521e+00], dtype='f')

    if(img_select==0):
    # Format of 3D point according to in X,Y,Z axes, in camera frame 
        pcd = np.array([[4, -2.5, 23],
               [4, -0.5,  23],
               [4, 1.5, 23],
               [4, 3.5, 23],
               [4, 5.5, 23],
               [4, 7.5, 23],
               [4, 9.5, 23],
               [4, 11.5, 23]])
        imgName = '0Z_-7X.jpg'
    else:
        pcd = np.array([[-3, -2.5, 23],
               [-3, -0.5,  23],
               [-3, 1.5, 23],
               [-3, 3.5, 23],
               [-3, 5.5, 23],
               [-3, 7.5, 23],
               [-3, 9.5, 23],
               [-3, 11.5, 23]])
        imgName = '0Z_0X.jpg'

pix_x , pix_y = pinhole_point_to_pixel (camera_matrix, distortion, pcd)

print ("pix_x is:", pix_x)
print ("pix_y is:", pix_y)

img = cv2.imread('point_to_pixel/'+imgName)

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img',img)
cv2.waitKey(100)


# Undistort for a normal camera
h,  w = img.shape[:2]
newcameramtx, roi =cv2.getOptimalNewCameraMatrix(camera_matrix,distortion,(w,h),1,(w,h))
undst = cv2.undistort(img, camera_matrix, distortion, None, newcameramtx)

cv2.imwrite('point_to_pixel/p2p_undist_'+imgName, undst)

cv2.namedWindow('undst', cv2.WINDOW_NORMAL)
cv2.imshow('undst',undst)
cv2.waitKey(100)


for idx,point in enumerate(pcd):
    print ("point is:", point)

    # Center coordinates
    center_coordinates = (pix_x[idx], pix_y[idx])

    # Radius of circle
    radius = 20

    # RED color in BGR
    color = (0, 0, 255)

    # Line thickness of 2 px
    thickness = 2

    img_circle = cv2.circle(img, center_coordinates, radius, color, thickness)

    undst_circle = cv2.circle(undst, center_coordinates, radius, color, thickness)

    cv2.namedWindow('img_circle', cv2.WINDOW_NORMAL)
    cv2.imshow('img_circle',img_circle)

    cv2.namedWindow('undst_circle', cv2.WINDOW_NORMAL)
    cv2.imshow('undst_circle',undst_circle)


    k = cv2.waitKey(0)







