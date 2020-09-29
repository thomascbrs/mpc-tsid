import pybullet as p  # PyBullet simulator
import numpy as np  # Numpy library
import cv2
import os
import matplotlib.pyplot as plt
from os.path import isfile, join

fps = 20
pathIn = 'recording/'

frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

#for sorting the file names properly
files.sort(key = lambda x: int(x[len('frame_'):len(x)-4]))

for i in range(len(files)):
    filename=pathIn + files[i]
    #reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    #inserting the frames into an image array
    frame_array.append(img)

adressVideo = 'test_1'+ '.avi'

out = cv2.VideoWriter(adressVideo,cv2.VideoWriter_fourcc(*'DIVX'), fps, size) 


for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()
