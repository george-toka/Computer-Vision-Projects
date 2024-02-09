import cv2 as cv
import numpy as np
import os

root = "mammals/test"

bow_descs = np.load('BOVW.npy').astype(np.float32)

img_paths = np.load('paths.npy')

# TRAINING
labels = []
classes = os.listdir(root)

for p in img_paths:
    for i in range(len(classes)):
        if classes[i] in p:
            labels.append(i)

labels = np.array(labels, np.int32)

knn = cv.ml.KNearest_create()
knn.train(bow_descs, cv.ml.ROW_SAMPLE, labels)

# TESTING
vocabulary = np.load('vocabulary.npy')
Class = -1 
hit_count = np.zeros(len(classes))
hit_ratio = np.zeros(len(classes))
n_of_tests = 0
for objectClass in classes:
    objectFolder = os.path.join(root, objectClass)
    tests = os.listdir(objectFolder)
    Class = Class + 1
    n_of_tests = n_of_tests + len(tests)
    for test in tests:
        test_path = os.path.join(objectFolder, test)

        sift = cv.xfeatures2d_SIFT.create()        

        descriptor_extractor = cv.BOWImgDescriptorExtractor(sift, cv.BFMatcher(cv.NORM_L2SQR))
        descriptor_extractor.setVocabulary(vocabulary)

        img = cv.imread(test_path)
        kp = sift.detect(img)
        bow_desc = descriptor_extractor.compute(img, kp)

        response, results, neighbours ,dist = knn.findNearest(bow_desc, 100)
        response = int (response)
        if classes.index(objectClass) == response:
            hit_count[Class] = hit_count[Class] + 1
            print("It is a " + objectClass + " (Hit)")
        else:
            print("It is a " + classes[response] + " (Miss)")

        cv.imshow('test object',img)
        #cv.waitKey()

    hit_ratio[Class] = hit_count[Class] / len(tests)
    print(hit_ratio[Class])

print(classes)
print(hit_ratio)
total_hr = sum(hit_count) / n_of_tests
print(total_hr)
pass