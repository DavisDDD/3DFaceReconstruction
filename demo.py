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
import apex.amp as amp
import ResNet as Net
import image_load as imgLoad
import render as sr

from load_data import *
from reconstruct_mesh import *
from compute_bfm import *

if __name__=='__demo__':
    # read BFM model
    model_path1 = './data/obj/vd092_mesh.obj'
    model_path2 = './BFM/BFM_model_front.mat'
    model_save_path = './model/'
    input_path = './data/import/'
    output_path = './data/output/'
    mesh = sr.Mesh.from_obj(model_path1,
                            load_texture=True, texture_res=5, texture_type='vertex')
    face = mesh.faces

    if not os.path.isfile(model_path2):
        transferBFM09()
    Face_model = BFM()
    BFM_net = compute_bfm(torch.tensor(Face_model.idBase, dtype=torch.float16)
                          , torch.tensor(Face_model.exBase, dtype=torch.float16)
                          , torch.tensor(Face_model.meanshape, dtype=torch.float16)
                          , torch.tensor(Face_model.texBase, dtype=torch.float16)
                          , torch.tensor(Face_model.meantex, dtype=torch.float16))

    net = torch.load(model_save_path + 'net.pkl')

