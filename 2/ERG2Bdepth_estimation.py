#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import open3d as o3d



def crop_pcd (pcd, x_min, x_max, y_min, y_max, z_min, z_max):

    condition_x = np.logical_and(x_min < pcd[:, 0], pcd[:, 0] < x_max)
    condition_y = np.logical_and(y_min < pcd[:, 1], pcd[:, 1] < y_max)
    condition_z = np.logical_and(z_min < pcd[:, 2], pcd[:, 2] < z_max)

    condition = np.where(condition_x & condition_y & condition_z)
    pcd_cropped = pcd[condition]

    return pcd_cropped


def match2(d1, d2):
    n1 = d1.shape[0]
    n2 = d2.shape[0]

    matches = []
    for i in range(n1):
        fv = d1[i, :]
        diff = d2 - fv
        diff = np.abs(diff)
        distances = np.sum(diff, axis=1)

        i2 = np.argmin(distances)
        mindist2 = distances[i2]

        distances[i2] = np.inf

        i3 = np.argmin(distances)
        mindist3 = distances[i3]

        if mindist2 / mindist3 < 0.5:
            matches.append(cv2.DMatch(i, i2, mindist2))

    return matches


def compute_disparity_map(pixels_image1, pixels_image2, img1, subtract_camera_height, select_set=2, wait:bool = True):

    points_3d = np.zeros((pixels_image1.shape[0], 6), dtype=np.float32)

    # These values should be according to your camera calibration parameters and image properties
    if(select_set==0):
        B = 14.4 #Difference of camera position of image2, in relation to camera position of image1 along X axis, in cm
        c = 4396.869 #focal length (fx) from calibration matrix
        image_dimensions_x = 2880 # image dimension of x axis in image frame
        image_dimensions_y = 1980 # image dimension of y axis in image frame
    elif(select_set==1):
        B = 17.0 #Difference of camera position of image2, in relation to camera position of image1 along X axis, in cm
        c = 4844.97 #focal length (fx) from calibration matrix
        image_dimensions_x = 2792 # image dimension of x axis in image frame
        image_dimensions_y = 2008 # image dimension of y axis in image frame
    else:
        B = 7.0 #Difference of camera position of image2, in relation to camera position of image1 along X axis, in cm
        c = 2.12195577e+03 #focal length (fx) from calibration matrix
        image_dimensions_x = 1944 # image dimension of x axis in image frame
        image_dimensions_y = 2592 # image dimension of y axis in image frame
    
    camera_height = 13.5 #in cm (according to your mobile phone)

    for i in range(pixels_image1.shape[0]):

        pixel_x_img1 = pixels_image1[i][0] - image_dimensions_x / 2.0
        pixel_x_img2 = pixels_image2[i][0] - image_dimensions_x / 2.0

        pixel_y_img1 = pixels_image1[i][1] - image_dimensions_y / 2.0
        pixel_y_img2 = pixels_image2[i][1] - image_dimensions_y / 2.0

        # Equation for X coordinate
        points_3d[i][0] = pixel_x_img1 * B / (- (pixel_x_img2 - pixel_x_img1))

        # Equation for Y coordinate
        points_3d[i][1] = (((pixel_y_img1 + pixel_y_img2)) / 2.0 ) * B / (- (pixel_x_img2 - pixel_x_img1))

        if subtract_camera_height:
            points_3d[i][1] = camera_height - points_3d[i][1]

        # Equation for Z coordinate
        points_3d[i][2] = c * B / (- (pixel_x_img2 - pixel_x_img1))


        points_3d[i][3:] = img1[int(pixels_image1[i][1])][int(pixels_image1[i][0])]

        print ("pixels_image1[",i,"] is: ", pixels_image1[i])
        print ("pixels_image2[",i,"] is: ", pixels_image2[i])
        print ("points_3d[",i,"] is: ", points_3d[i])
        print ("\n")

        # Center coordinates
        center_coordinates = (int(pixels_image1[i][0]), int(pixels_image1[i][1]))

        # Radius of circle
        radius = 10

        # RED color in BGR
        color = (0, 0, 255)

        # Line thickness of 2 px
        thickness = 2

        img1_circle = 0
        img1_circle = cv2.circle(img1.copy(), center_coordinates, radius, color, thickness)

        cv2.namedWindow('img1_circle', cv2.WINDOW_NORMAL)
        cv2.imshow('img1_circle', img1_circle)
        if(wait): cv2.waitKey(0)

    return points_3d

## --------------------- START of MAIN ---------------------------##

img1 = cv2.imread('point_to_pixel/p2p_undist_0Z_-7X.jpg')
img2 = cv2.imread('point_to_pixel/p2p_undist_0Z_0X.jpg')

# Manual selection of points
## From unDistorted images -----------------------------------
                         # x  , y   in image frame (or u,v)
pixels_image1 = np.array([[1326.0074, 1026.1467],#16cm
                         [1324.6006, 1213.5503], #14cm
                         [1324.4600, 1399.7924], #12cm
                         [1325.4362, 1586.6232], #10cm
                         [1326.8696, 1774.7916], #8cm
                         [1327.9335, 1963.4122], #6cm
                         [1328.2523, 2151.0173], #4cm
                         [1328.7955, 2339.258]]) #2cm

pixels_image2 = np.array([[672.84564, 1027.0957],#16cm
                         [673.98740, 1213.9210], #14cm
                         [674.06330, 1399.5157], #12cm
                         [673.19415, 1585.8992], #10cm
                         [671.92000, 1773.9993], #8cm
                         [670.94200, 1962.8989], #6cm
                         [670.65906, 2150.7678], #4cm
                         [670.49740, 2338.4429]]) #2cm
## ----------------------------------------------------------

subtract_camera_height = True
points_3d = compute_disparity_map(pixels_image1, pixels_image2, img1, subtract_camera_height)

x_min = -100.0
x_max = 100.0
y_min = -100.0
y_max = 100.0
z_min = 0.0
z_max = 1000.0

points_3d_cropped = crop_pcd (points_3d, x_min, x_max, y_min, y_max, z_min, z_max)

pointcloud = points_3d_cropped[:,0:3]
colors_pointcloud = points_3d_cropped[:,3:]/255.0

print ("pointcloud is:", pointcloud)

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pointcloud)
pcd.colors = o3d.utility.Vector3dVector(colors_pointcloud)
print ('pcd is:', pcd)
o3d.visualization.draw_geometries([pcd, mesh_frame])



select_set=2
if(select_set==0):
    imgName_1 = 'img0.png'
    imgName_2 = 'img1.png'
elif(select_set==1):
    imgName_1 = 'img2.png'
    imgName_2 = 'img3.png'
else:
    imgName_1 = 'p2p_undist_0Z_-7X.jpg'
    imgName_2 = 'p2p_undist_0Z_0X.jpg'

## Detection of keypoints is performed automatically with SURF
surf = cv2.xfeatures2d_SURF.create()

img1 = cv2.imread('point_to_pixel/' + imgName_1)
kp1 = surf.detect(img1)
desc1 = surf.compute(img1, kp1)


img2 = cv2.imread('point_to_pixel/' + imgName_2)
kp2 = surf.detect(img2)
desc2 = surf.compute(img2, kp2)

matches = match2(desc1[1], desc2[1])

dimg = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)


img_pt1 = []
img_pt2 = []
for x in matches:
    img_pt1.append(kp1[x.queryIdx].pt)
    img_pt2.append(kp2[x.trainIdx].pt)
img_pt1 = np.array(img_pt1)
img_pt2 = np.array(img_pt2)

matched_pixels_image1 = np.zeros((len(matches), 2), dtype=np.float32)
matched_pixels_image2 = np.zeros((len(matches), 2), dtype=np.float32)

for i,match in enumerate(matches):
  p1 = kp1[match.queryIdx].pt
  p2 = kp2[match.trainIdx].pt

  matched_pixels_image1[i][0] = int(p1[0])
  matched_pixels_image1[i][1] = int(p1[1])

  matched_pixels_image2[i][0] = int(p2[0])
  matched_pixels_image2[i][1] = int(p2[1])


cv2.namedWindow('main3', cv2.WINDOW_NORMAL)
cv2.imshow('main3', dimg)
cv2.waitKey(0)

subtract_camera_height = False
points_3d = compute_disparity_map(matched_pixels_image1, matched_pixels_image2, img1, subtract_camera_height, select_set, wait=False)


x_min = -100.0
x_max = 100.0
y_min = -100.0
y_max = 100.0
z_min = 0.0
z_max = 1000.0

points_3d_cropped = crop_pcd (points_3d, x_min, x_max, y_min, y_max, z_min, z_max)

pointcloud = points_3d_cropped[:,0:3]
colors_pointcloud = points_3d_cropped[:,3:]/255.0
colors_pointcloud_tmp = np.copy(colors_pointcloud)
colors_pointcloud_tmp[:,0] = colors_pointcloud[:,2]
colors_pointcloud_tmp[:,1] = colors_pointcloud[:,1]
colors_pointcloud_tmp[:,2] = colors_pointcloud[:,0]

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pointcloud)
pcd.colors = o3d.utility.Vector3dVector(colors_pointcloud_tmp)

o3d.visualization.draw_geometries([pcd, mesh_frame])


