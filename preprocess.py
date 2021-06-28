import os
import PIL
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import h5py
import cv2


labels = []
for filename in range(1, 3065):
    with h5py.File('C:/Users/Acer/Desktop/BrainTumourClassification/Brain Images/{}.mat'.format(filename), 'r') as f:
        img = f['cjdata']['image']
        label = f['cjdata']['label'][0][0]
        labels.append(int(label))
        img = np.array(img, dtype=np.float32)
        img = img / 127.5 - 1
        plt.axis('off')
        if int(label) == 1:
            plt.imsave("C:/Users/Acer/Desktop/BrainTumourClassification/bt_images/meningioma/{}.jpg".format(filename), img, cmap='gray')
        elif int(label) == 2:
            plt.imsave("C:/Users/Acer/Desktop/BrainTumourClassification/bt_images/glioma/{}.jpg".format(filename), img, cmap='gray')
        elif int(label) == 3:
            plt.imsave("C:/Users/Acer/Desktop/BrainTumourClassification/bt_images/pituitary tumor/{}.jpg".format(filename), img, cmap='gray')




