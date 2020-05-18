import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class compute_bfm(nn.Module):
    def __init__(self, idBase, exBase, meanshape, texBase, meantex, tri):
        super(compute_bfm, self).__init__()
        # Define the layers here
        # Note: for conv, use a padding of (1,1) so that size is maintained
        self.texBase = texBase.to(device)
        self.meantex = meantex.to(device)
        self.idBase = idBase.to(device)
        self.exBase = exBase.to(device)
        self.meanshape = meanshape.to(device)
        self.tri = tri.to(device)

    def forward(self, id_coeff, ex_coeff, tex_coeff):
        # define forward operation using the layers we have defined
        faceshape = (self.idBase * id_coeff).sum(axis=1) + (self.exBase * ex_coeff).sum(axis=1) + self.meanshape
        facetexture = (self.texBase * tex_coeff).sum(axis=1) + self.meantex
        re_center = faceshape - self.meanshape
        shape = faceshape.reshape(35709, 3).unsqueeze(0)
        texture = facetexture.reshape(35709, 3).unsqueeze(0)
        float_texture = texture / 255.0
        faces = (self.tri - 1).unsqueeze(0)
        return shape, float_texture, faces




