import os
import cv2 as cv
import numpy as np

train_folders = ['mammals/train']

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


print('Creating index...')

vocabulary = np.load('vocabulary.npy')

img_paths = []
# train_descs = np.zeros((0, 128))
bow_descs = np.zeros((0, vocabulary.shape[0]))
for folder in train_folders:
    subfolders = os.listdir(folder)
    for subfolder in subfolders:
        subfolder = os.path.join(folder, subfolder)
        files = os.listdir(subfolder)
        print(subfolder)
        for file in files:
            path = os.path.join(subfolder, file)
            desc = extract_local_features(path)
            if desc is None:
                continue
            bow_desc = encode_bovw_descriptor(desc, vocabulary)

            img_paths.append(path)
            bow_descs = np.concatenate((bow_descs, bow_desc), axis=0)

np.save('BOVW.npy', bow_descs)
np.save('paths', img_paths)