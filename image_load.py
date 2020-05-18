import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys
import os
from optparse import OptionParser
import numpy as np
from torch import optim
from PIL import Image
from torch.autograd import Function, Variable
import matplotlib.pyplot as plt
import matplotlib
from torchvision import transforms
import glob
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset
import cv2
import dlib

# get human face mask
predictor_model = './model/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()# dlib人脸检测器
predictor = dlib.shape_predictor(predictor_model)

def compute_landmark(img):
    points = np.zeros((68, 2))
    rects, scores, idx = detector.run(img, 2, 0)
    faces = dlib.full_object_detections()
    if len(rects) == 0:
        return False
    for rect in rects:
        faces.append(predictor(img, rect))
    i = 0
    for landmark in faces:
        for idx, point in enumerate(landmark.parts()):
            if i >= 68:
                return points
            points[i, 0] = point.x
            points[i, 1] = point.y
            i = i + 1
    return points

def preprocess_image(image_paths):
    img_list = []
    new_h, new_w = 200, 200
    for i in tqdm(range(len(image_paths))):
        img = Image.open(image_paths[i])
        img = np.array(img)
        img = cv2.resize(img, dsize=(new_h, new_w), interpolation=cv2.INTER_CUBIC)
        points = compute_landmark(img)
        if points is False:
            print("one image failed to landmark")
            continue
        else:
            img = np.array(img, np.float32) / 255.0
            img_list.append((img, points))
    return img_list

# save the data into pickle file and you can just reload this file, which can help you avoid reading the image
# file again in the future, since reading in image file from hard drive would take quite a long time
def pickle_store(file_name, save_data):
    fileObj = open(file_name, 'wb')
    pickle.dump(save_data, fileObj)
    fileObj.close()


# get all the image and mask path and number of images
train_image_paths = glob.glob("D:\\files\\project\\data\\human_faces\\testset\\*.jpg")
print('original image shape: {}'.format(np.array(Image.open(train_image_paths[0])).shape))
print('image type:'.format(Image.open(train_image_paths[0]).mode))
print('train len: {}'.format(len(train_image_paths)))

def load_image():
    if os.path.exists(train_img_save_path):
        with open(train_img_save_path, 'rb') as f:
            train_img = pickle.load(f)
        f.close()
    else:
        train_img = preprocess_image(train_image_paths)
        pickle_store(train_img_save_path, train_img)
    return train_img

train_img_save_path = 'data/train_img.pickle'
if os.path.exists(train_img_save_path):
    with open(train_img_save_path, 'rb') as f:
        train_img = pickle.load(f)
    f.close()
else:
    train_img = preprocess_image(train_image_paths)
    pickle_store(train_img_save_path, train_img)
print('train len: {}'.format(len(train_img)))









