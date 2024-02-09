import cv2 as cv
import numpy as np
import os

sift = cv.xfeatures2d_SIFT.create()

def extract_local_features(path):
    img = cv.imread(path)
    kp = sift.detect(img)
    desc = sift.compute(img, kp)
    desc = desc[1]
    return desc

bow_descs = np.load('BOVW.npy').astype(np.float32)

img_paths = np.load('paths.npy')

# Train SVM
print('Training SVM...')
svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))

root = "mammals/test"

classes = os.listdir(root)
labels = []
for p in img_paths:
    for i in range(len(classes)):
        if classes[i] in p:
            labels.append(i)

labels = np.array(labels, np.int32)

svm.trainAuto(bow_descs, cv.ml.ROW_SAMPLE, labels)
svm.save('svm')