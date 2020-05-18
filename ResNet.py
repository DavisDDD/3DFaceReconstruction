import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

__all__ = ['ResNet50', 'ResNet101','ResNet152']

# 定义单个卷积层：conv2d + BatchNormal + relu + Pool
def Conv1(in_ch, out_ch, stride=2):
    # nn.Sequential有序容器，输入会被依次执行
    return nn.Sequential(
        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=7, stride=stride, padding=3, bias=False).to(device),
        nn.BatchNorm2d(out_ch).to(device),
        nn.ReLU(inplace=True).to(device),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(device)
    )

# 定义一个结构块：bottleneck
class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, bias=False).to(device),
            nn.BatchNorm2d(out_ch).to(device),
            nn.ReLU(inplace=True).to(device),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=stride, padding=1, bias=False).to(device),
            nn.BatchNorm2d(out_ch).to(device),
            nn.ReLU(inplace=True).to(device),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch*self.expansion, kernel_size=1, stride=1, bias=False).to(device),
            nn.BatchNorm2d(out_ch*self.expansion).to(device),
        )

        # 如果为downsampling，输入x与输出net(x)的dimension不同，需要调整
        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch*self.expansion, kernel_size=1, stride=stride, bias=False).to(device),
                nn.BatchNorm2d(out_ch*self.expansion).to(device)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        # residual为残差
        if self.downsampling:
            residual = self.downsample(x)

        # out = x + net(x)
        out += residual
        out = self.relu(out)
        return out

class compute_bfm(nn.Module):
    def __init__(self, idBase, exBase, meanshape, texBase, meantex):
        super(compute_bfm, self).__init__()
        # Define the layers here
        # Note: for conv, use a padding of (1,1) so that size is maintained
        self.texBase = texBase.to(device)
        self.meantex = meantex.to(device)
        self.idBase = idBase.to(device)
        self.exBase = exBase.to(device)
        self.meanshape = meanshape.to(device)

    def forward(self, id_coeff, ex_coeff, tex_coeff):
        # define forward operation using the layers we have defined
        print(self.texBase.shape)
        print(tex_coeff.shape)
        print(self.meantex.shape)
        faceshape = (self.idBase * id_coeff).sum(axis=1) + (self.exBase * ex_coeff).sum(axis=1) + self.meanshape
        facetexture = (self.texBase * tex_coeff).sum(axis=1) + self.meantex
        re_center = faceshape - self.meanshape
        shape = faceshape.resize_(35709, 3).unsqueeze(0)
        texture = facetexture.resize_(35709, 3).unsqueeze(0)
        return shape, texture, re_center

class ResNet(nn.Module):
    def __init__(self, blocks, num_classes=230, expansion=4):
        super(ResNet, self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_ch=3, out_ch=64)

        self.layer1 = self.make_layer(in_ch=64, out_ch=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_ch=256, out_ch=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_ch=512, out_ch=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_ch=1024, out_ch=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, num_classes).to(device)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_ch, out_ch, block, stride):
        layers = []
        layers.append(Bottleneck(in_ch, out_ch, stride, downsampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(out_ch*self.expansion, out_ch))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResNet50():
    return ResNet([3, 4, 6, 3])

def ResNet101():
    return ResNet([3, 4, 23, 3])

def ResNet152():
    return ResNet([3, 8, 36, 3])


if __name__ =='__main__':
    model = ResNet50()
    print(model)
    input = torch.randn(5, 3, 200, 200).to(device)
    w = torch.randn(230).to(device)
    labels = torch.randn(5, 230)
    out = model(input)
    print(torch.max(out))


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    epoch = 0
    optimizer.zero_grad()

    loss = criterion(out.to(device), labels.to(device))
    epoch = epoch + loss
    print(loss)
    loss.backward()
    optimizer.step()


    out1 = model(input)

    torch.save(model, 'net.pkl')

    new = torch.load('net.pkl')
    out2 = new(input)
    print(out2.shape)