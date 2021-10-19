import glob
from random import shuffle
import h5py
import numpy as np
import cv2
import math
import time
import matplotlib.pyplot as plt

destination_filepath = 'datasets/celeb_images1.h5'
with h5py.File(destination_filepath, "r") as f:
    print(list(f.keys()))

    x = f["input_data"][:]
    y = f["input_labels"][:]

    print('x shape =', x.shape, '| y shape =', y.shape)

index = 0
image = x[index]
#print(image)
#print(image.shape)
image = (x[index]).reshape(128,128,3)
plt.imshow(image)
print('y =', y[index])