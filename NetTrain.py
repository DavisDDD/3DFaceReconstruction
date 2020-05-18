import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys
import os
from optparse import OptionParser
import numpy as np
import torchvision
import torchvision as torchvision
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
import skimage.transform as trans

from load_data import *
from reconstruct_mesh import *
from compute_bfm import *
from image_load import *
from model.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# the dataset class
class Dataset(Dataset):
    def __init__(self, image_masks, transforms=None):
        self.image_masks = image_masks

    def __len__(self):  # return count of sample we have

        return len(self.image_masks)

    def __getitem__(self, index):
        image = self.image_masks[index][0]  # H, W, C
        image = np.transpose(image, axes=[2, 0, 1])  # C, H, W
        points = self.image_masks[index][1]
        sample = {'img': image, 'points':points}
        return sample





if __name__=='__main__':
    torch.set_grad_enabled(True)
    # create net model
    model_save_path = './model/'  # directory to same the model after each epoch.
    if os.path.exists(model_save_path+'net1.pkl'):
        net = torch.load(model_save_path+'net1.pkl')
    else:
        net = Net.ResNet([3, 4, 6, 3]).to(device)
    train_img = load_image()
    train_dataset = Dataset(train_img)
    epochs = 5  # e.g. 10, or more until dice converge
    batch_size = 16  # e.g. 16
    lr = 0.001  # e.g. 0.01


    optimizer = optim.SGD(net.parameters(), lr=lr)
    #optimizer = optim.Adam(net.parameters(), lr=lr)
    #net, optimizer = amp.initialize(net, optimizer, opt_level="O1")
    # what to use as loss function, not decided yet.
    criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss()


    if not os.path.isfile('./BFM/BFM_model_front.mat'):
        transferBFM09()
    Face_model = BFM()
    BFM_net = compute_bfm(torch.tensor(Face_model.idBase, dtype=torch.float, requires_grad=False)
                          , torch.tensor(Face_model.exBase, dtype=torch.float, requires_grad=False)
                          , torch.tensor(Face_model.meanshape, dtype=torch.float, requires_grad=False)
                          , torch.tensor(Face_model.texBase, dtype=torch.float, requires_grad=False)
                          , torch.tensor(Face_model.meantex, dtype=torch.float, requires_grad=False)
                          , torch.tensor(Face_model.tri, dtype=torch.int32, requires_grad=False))


    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, shuffle=True, num_workers=0)
        count = 0

        for i, b in enumerate(train_loader):
            # through net, get coeffs
            imgs = b['img'].to(device)
            BFM_coeff = net(imgs)
            true_points = b['points']

            # vertices, faces, textures, camera_distance, elevation, azimuth
            id_coeff = BFM_coeff[:, 0:80]
            ex_coeff = BFM_coeff[:, 80:144]
            tex_coeff = BFM_coeff[:, 144:224]
            camera_distance = BFM_coeff[:, 224]
            elevation = BFM_coeff[:, 224:226].sum(axis=1)
            azimuth = BFM_coeff[:, 226:228].sum(axis=1)
            # generate 2D face image
            True_Faces = imgs.to(device)
            Pred_face = torch.tensor(torch.zeros(len(True_Faces), 3, 200, 200))
            for n in range(len(True_Faces)):
                # generate 3D object
                vertices, textures, faces = BFM_net(id_coeff[n], ex_coeff[n], tex_coeff[n])

                # generate 2D image from 3D object
                camera_dist = 3.0
                ele = elevation[n] + 10
                angle = azimuth[n] + 180
                renderer = sr.SoftRenderer(camera_mode='look_at')
                renderer.transform.set_eyes_from_angles(camera_dist, ele, angle)
                pred_image = renderer.forward(vertices, faces, textures, texture_type='vertex')
                pred_images = pred_image[:, 0:3]

                # generate affine
                points2 = true_points[n].cpu().numpy()
                image = pred_images.clone().detach().cpu().numpy()[0, 0:3].transpose((1, 2, 0))
                image = (image * 255).astype(np.uint8)
                points1 = compute_landmark(image)
                if points1 is not False:
                    tr = trans.estimate_transform('affine', src=points1, dst=points2)
                    M = tr.params[0:2, :]
                    param = np.linalg.inv(tr.params)
                    theta = normalize_transforms(param[0:2, :], 200, 200)
                    theta = torch.Tensor(theta).unsqueeze(0).to(device)
                    grid = F.affine_grid(theta, pred_images.size()).to(device)
                    pred_images = F.grid_sample(pred_images, grid)
                    pred_images = pred_images.squeeze(0)
                    Pred_face[n] = pred_images
                else:
                    Pred_face[n] = pred_image[:, 0: 3]
                    image = Pred_face[n].clone().detach().cpu().numpy().transpose((1, 2, 0))
                    plt.figure(0)
                    plt.imshow(image)
                    plt.show()
                    print("error once: can't detect face")
            loss = criterion(Pred_face.to(device), True_Faces.to(device))
            count = count + 1
            if count % 10 == 0:
                print(loss)
            optimizer.zero_grad()
            #with amp.scale_loss(loss, optimizer) as scaled_loss:
            #    scaled_loss.backward()
            loss.backward()
            optimizer.step()
            if count % 100 == 0:
                torch.save(net, model_save_path + 'net.pkl')
                print("****************update net****************")

        image = Pred_face[0].detach().cpu().numpy().transpose((1, 2, 0))
        plt.figure(0)
        plt.imshow(image)
        plt.show()
        humanface = True_Faces[0].detach().cpu().numpy().transpose((1, 2, 0))
        plt.figure(1)
        plt.imshow(humanface)
        plt.show()
        torch.save(net, model_save_path + 'net.pkl')
        print("****************one epoch finished****************")








