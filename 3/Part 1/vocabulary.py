import os
import cv2 as cv
import numpy as np

sift = cv.xfeatures2d_SIFT.create()

train_folders = ["mammals/train"]

def extract_local_features(path):
    img = cv.imread(path)
    kp = sift.detect(img)
    desc = sift.compute(img, kp)
    desc = desc[1]
    return desc


# Extract Database
print('Extracting features...')
train_descs = np.zeros((0, 128))
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
            train_descs = np.concatenate((train_descs, desc), axis=0)

# Create vocabulary
print('Creating vocabulary...')
term_crit = (cv.TERM_CRITERIA_EPS, 30, 0.1)
trainer = cv.BOWKMeansTrainer(70, term_crit, 1, cv.KMEANS_PP_CENTERS)
vocabulary = trainer.cluster(train_descs.astype(np.float32))

np.save('vocabulary.npy', vocabulary)