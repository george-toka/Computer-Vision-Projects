import cv2 as cv
import numpy as np


sift = cv.xfeatures2d_SIFT.create()

def extract_local_features(path):
    img = cv.imread(path)
    kp = sift.detect(img)
    desc = sift.compute(img, kp)
    desc = desc[1]
    return desc

def encode_bovw_descriptor(desc, vocabulary):
    bow_desc = np.zeros((1, vocabulary.shape[0]))
    for d in range(desc.shape[0]):
        distances = np.sum((desc[d, :] - vocabulary) ** 2, axis=1)
        mini = np.argmin(distances)
        bow_desc[0, mini] += 1

    # Normalization
    if np.sum(bow_desc) > 0:
        bow_desc = bow_desc / np.sum(bow_desc)
    return bow_desc

vocabulary = np.load('vocabulary.npy')

bow_descs = np.load('BOVW.npy')

img_paths = np.load('paths.npy')

# Search
query_img_path = "mammals/test/squirrel/squirrel-0244.jpg"  
query_img = cv.imread(query_img_path)

cv.namedWindow('query', cv.WINDOW_NORMAL)
cv.imshow('query', query_img)
cv.waitKey()

desc = extract_local_features(query_img_path)

bow_desc = encode_bovw_descriptor(desc, vocabulary)

distances = np.sum((bow_desc - bow_descs) ** 2, axis=1)
retrieved_ids = np.argsort(distances)
cv.namedWindow('results', cv.WINDOW_NORMAL)
for id in retrieved_ids.tolist():
    result_img = cv.imread(img_paths[id])
    cv.imshow('results', result_img)
    cv.waitKey(250)
pass