import numpy as np
import cv2 as cv

show_index = 0
def image_stitching(matches, img1, img2, kp1, kp2, SIFT:bool, offset=np.zeros(2, dtype = int)):
    img_pt1 = []
    img_pt2 = []
    for x in matches:
        img_pt1.append(kp1[x.queryIdx].pt)
        img_pt2.append(kp2[x.trainIdx].pt)
    img_pt1 = np.array(img_pt1)
    img_pt2 = np.array(img_pt2)
    
    M = cv.findHomography(img_pt2, img_pt1, cv.RANSAC)[0]
    
    border = 2000
    img3 = cv.warpPerspective(img2, M, (img1.shape[1]+border, img1.shape[0]+border))
    img3[0: img1.shape[0], 0: img1.shape[1]] = img1

    # crop the redundant right part of border
    threshold = int(border/4)
    count = 0
    
    for i in range(img1.shape[1], img3.shape[1]-1):
        if(img3[10][i] == 0 and img3[10][i+1] == 0):
            count = count + 1
        else:
            count = 0
        if(count == threshold):
            right_margin = i+1 - threshold - offset[1]
            img3 = img3[0:img3.shape[0],0:right_margin] 
            break
      
    count = 0
    # crop the redundant lower part of border
    for i in range(img1.shape[0], img3.shape[0]-1):
        if(img3[i][10] == 0 and img3[i+1][10] == 0):
            count = count + 1
        else:
            count = 0
        if(count == threshold):
            lower_margin = i+1 - threshold - offset[0]
            img3 = img3[0:lower_margin,0:img3.shape[1]] 
            break   
    
    # get info for the newly merged image
    if(SIFT):
        Feature_Detector = cv.xfeatures2d.SIFT_create(400)
    else:
        Feature_Detector = cv.xfeatures2d.SURF_create(400)

    kp3 = Feature_Detector.detect(img3)
    desc3 = Feature_Detector.compute(img3, kp3)

    return img3, kp3, desc3

def cross_checking(img1, img2, kp1, kp2, matches12, matches21, showMatches:bool):
    cross_matches = []
    for match1 in matches12:
        for match2 in matches21:
            if(match1.queryIdx == match2.trainIdx and match1.trainIdx == match2.queryIdx):
                cross_matches.append(match1)
    
    print('Number of cross matches: ' + str(len(cross_matches)))
    #Show Matches
    if(showMatches):
        global show_index 
        show_index = show_index + 1

        dimg = cv.drawMatches(img1, kp1, img2, kp2, cross_matches, None)
        cv.namedWindow('cross_match')
        cv.imshow('cross_match', dimg)
        cv.waitKey(0)

    return cross_matches

def feature_matching(img1, img2, kp1,kp2, d1, d2, showMatches:bool):
    n1 = d1.shape[0]
    n2 = d2.shape[0]

    # Matching Image Features Of 2 to 1
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
            matches.append(cv.DMatch(i, i2, mindist2))
    # Show Matches
    if(showMatches):
        global show_index 
        show_index = show_index + 1

        if (len(matches)>0):
            dimg = cv.drawMatches(img1, kp1, img2, kp2, matches, None)
            cv.namedWindow('match')
            cv.imshow('match', dimg)
            cv.waitKey(0)
    print('Number of matches: ' + str(len(matches)))
    return matches

def img_reader(src, BigImgFlag, show:bool, SIFT:bool):

    # Read + Get Key-Points & Their Descriptors
    img = cv.imread(src, cv.IMREAD_GRAYSCALE)
    img = img if(not BigImgFlag) else cv.resize(img, None, fx=0.5, fy=0.5)

    if(SIFT):
        Feature_Detector = cv.xfeatures2d.SIFT_create(400)
    else:
        Feature_Detector = cv.xfeatures2d.SURF_create(400)

    kp = Feature_Detector.detect(img)
    desc = Feature_Detector.compute(img, kp)

    # Show Original Image
    if(show):
        global show_index 
        show_index = show_index + 1
        
        cv.namedWindow('main ' + str(show_index))
        cv.imshow('main ' + str(show_index), img)
        cv.waitKey(0)

    return img,kp,desc


#main program
task_images = ['rio-01.png','rio-02.png','rio-03.png','rio-04.png']
my_images = ['image1.png', 'image2.png', 'image3.png', 'image4.png']

Flag = False #Turn to false for surf algorithm
BigImgFlag = True  #True when testing my_images 

# image read
img1,kp1,desc1 = img_reader("panorama/" + my_images[0], BigImgFlag, show=True, SIFT=Flag)
img2,kp2,desc2 = img_reader("panorama/" + my_images[1], BigImgFlag, show=True, SIFT=Flag)
img3,kp3,desc3 = img_reader("panorama/" + my_images[2], BigImgFlag, show=True, SIFT=Flag)
img4,kp4,desc4 = img_reader("panorama/" + my_images[3], BigImgFlag, show=True, SIFT=Flag)

# match & stitch 1-2
matches12 = feature_matching(img1, img2, kp1, kp2, desc1[1], desc2[1], showMatches = True)
matches21 = feature_matching(img2, img1, kp2, kp1, desc2[1], desc1[1], showMatches = True)
cross_matches12 = cross_checking(img1, img2, kp1, kp2, matches12, matches21, showMatches = True)
stitched12, kp12, desc12 = image_stitching(cross_matches12, img1, img2, kp1, kp2, SIFT = Flag, offset=np.array([0,50], dtype=int))

cv.namedWindow('main')
cv.imshow('main', stitched12)
cv.waitKey(0)

# match & stitch 3-4
matches34 = feature_matching(img3, img4, kp3, kp4, desc3[1], desc4[1], showMatches = True)
matches43 = feature_matching(img4, img3, kp4, kp3, desc4[1], desc3[1], showMatches = True)
cross_matches34 = cross_checking(img3, img4, kp3, kp4, matches34, matches43, showMatches = True)
stitched34, kp34, desc34 = image_stitching(cross_matches34, img3, img4, kp3, kp4, SIFT = Flag)

cv.namedWindow('main')
cv.imshow('main', stitched34)
cv.waitKey(0)

# match & stitch 12-34
matches12_34 = feature_matching(stitched12, stitched34, kp12, kp34, desc12[1], desc34[1], showMatches = True)
matches34_12 = feature_matching(stitched34, stitched12, kp34, kp12, desc34[1], desc12[1], showMatches = True)
cross_matches12_34 = cross_checking(stitched12, stitched34, kp12, kp34, matches12_34, matches34_12, showMatches = True)
stitched12_34 = image_stitching(cross_matches12_34, stitched12, stitched34, kp12, kp34, SIFT = Flag)[0]

cv.namedWindow('main')
cv.imshow('main', stitched12_34)
cv.waitKey(0)
