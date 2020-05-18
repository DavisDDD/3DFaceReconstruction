import glob
import cv2
import matplotlib.pyplot as plt
import os

import torchvision
import tqdm
import numpy as np
import imageio
import argparse
import render as sr
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import ResNet as Net
import load_data as load
import dlib
import skimage.transform as trans

from compute_bfm import compute_bfm
from learning.WarpAffine2GridSample.utils import show_image, normalize_transforms, convert_image_np
from load_data import transferBFM09, BFM
import image_load as load

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '../data')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
predictor_model = './model/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()# dlib人脸检测器
predictor = dlib.shape_predictor(predictor_model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename-input', type=str,
        default=os.path.join('./data/obj/liu_mesh.obj'))
    parser.add_argument('-o', '--output-dir', type=str,
        default=os.path.join('./data/obj/vd092_mesh.obj'))
    args = parser.parse_args()

    # other settings
    camera_distance = 3.0
    elevation = -5
    azimuth = 0

    # load from Wavefront .obj file
    mesh = sr.Mesh.from_obj(args.filename_input,
                            load_texture=True, texture_res=5, texture_type='vertex')

    # create renderer with SoftRas
    renderer = sr.SoftRenderer(camera_mode='look_at')

    net = Net.ResNet([3, 4, 23, 3]).to(device)
    im = torch.rand(1, 3, 200, 200).to(device)
    out = net(im).to(device)
    BFM_coeff = out

    if not os.path.isfile('./BFM/BFM_model_front.mat'):
        transferBFM09()
    Face_model = BFM()
    BFM_net = compute_bfm(torch.tensor(Face_model.idBase, dtype=torch.float16)
                          , torch.tensor(Face_model.exBase, dtype=torch.float16)
                          , torch.tensor(Face_model.meanshape, dtype=torch.float16)
                          , torch.tensor(Face_model.texBase, dtype=torch.float16)
                          , torch.tensor(Face_model.meantex, dtype=torch.float16)
                          , torch.tensor(Face_model.tri, dtype=torch.int32))
    id_coeff = BFM_coeff[:, 0:80]
    ex_coeff = BFM_coeff[:, 80:144]
    tex_coeff = BFM_coeff[:, 144:224]
    print(id_coeff)
    vertices, textures, tri = BFM_net(id_coeff, ex_coeff, tex_coeff)
    # draw object from different view
    mesh.reset_()
    elevation = BFM_coeff[:, 226]
    # elevation = torch.sum(elevation)
    azimuth = -90

    renderer.transform.set_eyes_from_angles(camera_distance, 0, 180)
    # images = renderer.render_mesh(mesh)
    print(vertices)
    print(mesh.faces)
    faces = torch.tensor(Face_model.tri, dtype=torch.int32).to(device) - 1
    faces = faces.unsqueeze(0)
    print(faces)

    images = renderer.forward(mesh.vertices, mesh.faces, mesh.textures, texture_type='vertex')
    print(images)
    image = images.detach().cpu().numpy()[0,0:3].transpose((1, 2, 0))
    image = (image*255).astype(np.uint8)

    plt.figure(0)
    plt.imshow(image)
    plt.show()

    img_name1 = 'D:/files/project/data/human_faces/CACD2000/CACD2000/17_Jennifer_Lawrence_0013.jpg'
    img_name2 = 'D:/files/project/data/human_faces/CACD2000/CACD2000/17_Lily_Cole_0008.jpg'

    predictor_model = './model/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()  # dlib人脸检测器
    predictor = dlib.shape_predictor(predictor_model)

    img1 = cv2.imread(img_name1)
    image2 = cv2.imread(img_name2)
    image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    points1 = np.zeros((68, 2))
    points2 = np.zeros((68, 2))
    rects, scores, idx = detector.run(image, 2, 0)
    faces = dlib.full_object_detections()
    for rect in rects:
        faces.append(predictor(image, rect))
    i = 0
    for landmark in faces:
        for idx, point in enumerate(landmark.parts()):
            points1[i, 0] = point.x
            points1[i, 1] = point.y
            i = i + 1
    h, w, c = image.shape
    show_image(image, points1)

    rects, scores, idx = detector.run(image2, 2, 0)
    faces = dlib.full_object_detections()
    for rect in rects:
        faces.append(predictor(image2, rect))
    i = 0
    for landmark in faces:
        for idx, point in enumerate(landmark.parts()):
            points2[i, 0] = point.x
            points2[i, 1] = point.y
            i = i + 1
    h, w, c = image2.shape
    show_image(image2, points2)

    tr = trans.estimate_transform('affine', src=points1, dst=points2)
    M = tr.params[0:2, :]
    cv_img = cv2.warpAffine(image1, M, (image.shape[1], image.shape[0]))
    show_image(image2, points2)

    param = np.linalg.inv(tr.params)
    theta = normalize_transforms(param[0:2, :], w, h)

    to_tensor = torchvision.transforms.ToTensor()
    tensor_img = to_tensor(image).unsqueeze(0)
    theta = torch.Tensor(theta).unsqueeze(0)

    grid = F.affine_grid(theta, tensor_img.size())
    tensor_img = F.grid_sample(tensor_img, grid)
    tensor_img = tensor_img.squeeze(0)
    warp_img = convert_image_np(tensor_img)
    show_image(warp_img, points2)

    vertices = vertices[0].detach().cpu().numpy()
    faces = faces[0].detach().cpu().numpy() + 1
    textures = textures[0].detach().cpu().numpy()
    load.save_obj('./123.obj', vertices, faces, textures)


if __name__ == '__main__':
    main()


