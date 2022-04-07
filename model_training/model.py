import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import pretrainedmodels

from modules import ConvBlock, DownBlock, UpBlock, HeadBlock


class KPR(nn.Module):
    """
    KeyPoints Regressor model
    """

    def __init__(self, pretrained, requires_grad):
        """Initialize the KeyPoints Regressor model from a ResNet-50 base

        Parameters
        ----------
        pretrained: bool
            to get the pretrained weights of the ResNet-50 on ImageNet
        requires_grad: bool
            to retrain the ResNet-50 weights
        """
        super(KPR, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained=None)


        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
            print('Training intermediate layer parameters...')
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
            print('Freezing intermediate layer parameters...')


        # new head for regression
        self.conv_block_1 = ConvBlock(2048, 512)
        self.conv_block_2 = ConvBlock(512, 256)

        self.linear = nn.Linear(256, 8)

    def forward(self, x):
        """Forward pass of the KeyPoints Regressor model

        Parameters
        ----------
        x: torch.Tensor
            input tensor to pass forward

        Returns
        -------
        torch.Tensor
            Resulting output tensor
        """
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        # x = self.bn_1(self.relu_1(self.conv_1(x)))

        x = self.conv_block_1(x)
        x = self.conv_block_2(x)

        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        x = self.linear(x)
        return x


class EBUnet(nn.Module):
    """
    Edge-Based UNet model
    """

    def __init__(self):
        """Initialize the Edge-Based UNet model
        """
        super(EBUnet, self).__init__()
        inp = [3, 32, 128]
        inter = [16, 64, 256]
        out = [32, 128, 512]
        # down blocks
        self.down_1 = DownBlock(input=inp[0], inter=inter[0], output=out[0])
        self.down_2 = DownBlock(input=inp[1], inter=inter[1], output=out[1])
        self.down_3 = DownBlock(input=inp[2], inter=inter[2], output=out[2])

        # up bocks
        self.up_2 = UpBlock(input=out[2] + out[1], inter=inter[2], output=out[1])
        self.up_1 = UpBlock(input=out[1] + out[0], inter=inter[1], output=out[0])

        # head
        self.head = HeadBlock(input=out[0], inter=inter[0], output=4)

    def forward(self, x, train=False):
        """Forward pass of the Edge-Based UNet model

        Parameters
        ----------
        x: torch.Tensor
            input tensor to pass forward

        Returns
        -------
        torch.Tensor
            Resulting output tensor
        """
        # down block with skip connection retention
        x = self.down_1(x)
        skip_1 = x
        x = self.down_2(x)
        skip_2 = x
        x = self.down_3(x)

        # up block with skip concatenation
        x = self.up_2(x, skip=skip_2)
        x = self.up_1(x, skip=skip_1)

        # head
        x = self.head(x)
        return x



if __name__ == '__main__':
    # get the models and the GPU device if available
    kpr = KPR(False, False)
    unet = EBUnet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0

    # print their summary
    print('Running on:', device)
    SIZE = (3, 448, 448) # KPR
    print('KPR')
    if device.type == 'cuda':
        summary(kpr.to(device), SIZE, batch_size=1)
    else:
        summary(kpr, SIZE, batch_size=1, device='cpu')

    SIZE = (3, 128, 128) # EB-Unet
    print('EB-Unet')
    if device.type == 'cuda':
        summary(unet.to(device), SIZE, batch_size=1)
    else:
        summary(unet, SIZE, batch_size=1, device='cpu')
