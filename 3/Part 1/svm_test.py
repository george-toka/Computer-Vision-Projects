import cv2 as cv
import numpy as np
import os 

sift = cv.xfeatures2d_SIFT.create()

vocabulary = np.load('vocabulary.npy')

root = "mammals/test"
classes = os.listdir(root)

Class = -1 
hit_count = np.zeros(len(classes))
hit_ratio = np.zeros(len(classes))
n_of_tests = 0
for objectClass in classes:
    Class = Class + 1
    objectFolder = os.path.join(root, objectClass)
    tests = os.listdir(objectFolder)
    n_of_tests = n_of_tests + len(tests)
    for test in tests:
        test_path = os.path.join(objectFolder, test)
        # Load SVM
        svm = cv.ml.SVM_create()
        svm = svm.load('svm')

        # Classification
        descriptor_extractor = cv.BOWImgDescriptorExtractor(sift, cv.BFMatcher(cv.NORM_L2SQR))
        descriptor_extractor.setVocabulary(vocabulary)

        img = cv.imread(test_path)
        kp = sift.detect(img)
        bow_desc = descriptor_extractor.compute(img, kp)

        response = svm.predict(bow_desc, flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
        i = int (response[1])

        if classes.index(objectClass) == i:
            hit_count[Class] = hit_count[Class] + 1
            print("It is a " + objectClass + " (Hit)")
        else:
            print("It is a " + classes[i] + " (Miss)")

        cv.imshow('test object',img)
        cv.waitKey()

    hit_ratio[Class] = hit_count[Class] / len(tests)
    print(hit_ratio[Class])

print(classes)
print(hit_ratio)
total_hr = sum(hit_count) / n_of_tests
print(total_hr)
pass