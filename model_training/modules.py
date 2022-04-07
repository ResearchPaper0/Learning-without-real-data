import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Convolutionnal block of the KPR
    """

    def __init__(self, input, output):
        """Initialize the ConvBlock

        Parameters
        ----------
        input: int
            number of input channels
        output: int
            number of output channels
        """
        super(ConvBlock,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input, output, kernel_size=1),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True),
            nn.Conv2d(output, output, kernel_size=3, padding=1),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Forward pass of the ConvBlock

        Parameters
        ----------
        x: torch.Tensor
            input tensor to pass forward

        Returns
        -------
        torch.Tensor
            Resulting output tensor
        """
        x = self.conv(x)
        return x


class DownBlock(nn.Module):
    """
    Down block of the EB-Unet
    """

    def __init__(self, input, output, inter=None):
        """Initialize the ConvBlock

        Parameters
        ----------
        input: int
            number of input channels
        output: int
            number of output channels
        inter: int
            number of internal channels. Same as output if not specified
        """
        if inter is None:
            inter = output
        super(DownBlock,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input, inter, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, output, kernel_size=3, padding=1),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True)
        )
        # residual
        self.avg_pool = nn.AvgPool2d(2)
        self.conv_1x1 = nn.Conv2d(input, output, kernel_size=1)

    def forward(self,x):
        """Forward pass of the DownBlock

        Parameters
        ----------
        x: torch.Tensor
            input tensor to pass forward

        Returns
        -------
        torch.Tensor
            Resulting output tensor
        """
        _x = self.conv_1x1(self.avg_pool(x))
        x = self.conv(x)
        return x + _x

class UpBlock(nn.Module):
    """
    Up block of the EB-Unet
    """

    def __init__(self, input, output, inter=None):
        """Initialize the UpBlock

        Parameters
        ----------
        input: int
            number of input channels
        output: int
            number of output channels
        inter: int
            number of internal channels. Same as output if not specified
        """
        if inter is None:
            inter = output
        super(UpBlock,self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2)

        self.conv = nn.Sequential(
            nn.Conv2d(input, inter, kernel_size=3, padding=1),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, output, kernel_size=3, padding=1),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True)
        )
        self.conv_1x1 = nn.Conv2d(input, output, kernel_size=1)

    def forward(self, x, skip):
        """Forward pass of the UpBlock

        Parameters
        ----------
        x: torch.Tensor
            input tensor to pass forward

        Returns
        -------
        torch.Tensor
            Resulting output tensor
        """
        x = self.up_sample(x)
        x = torch.cat([x, skip], dim=1)
        _x = self.conv_1x1(x)
        x = self.conv(x)
        return x + _x

class MySoftmax(nn.Module):
    """
    Channel-wise softmax activation for the EB-Unet
    """

    def forward(self, x):
        """Forward pass of the xhannel-wise softmax activation function

        Parameters
        ----------
        x: torch.Tensor
            input tensor to pass forward

        Returns
        -------
        torch.Tensor
            Resulting output tensor
        """
        B, C, H, W = x.shape
        x = torch.stack([
                torch.stack([
                    F.softmax(x[b, c].view(-1), dim=0).view(H, W)
                    for c in range(C)
                ], 0)
                for b in range(B)
            ], 0)

        return x

class HeadBlock(nn.Module):
    """
    Head block of the EB-Unet
    """

    def __init__(self, input, output, inter=None):
        """Initialize the HeadBlock

        Parameters
        ----------
        input: int
            number of input channels
        output: int
            number of output channels
        inter: int
            number of internal channels. Same as output if not specified
        """
        if inter is None:
            inter = output
        super(HeadBlock,self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2)

        self.conv = nn.Sequential(
            nn.Conv2d(input, inter, kernel_size=3, padding=1),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, output, kernel_size=3, padding=1),
            MySoftmax()
        )

    def forward(self, x):
        """Forward pass of the HeadBlock

        Parameters
        ----------
        x: torch.Tensor
            input tensor to pass forward

        Returns
        -------
        torch.Tensor
            Resulting output tensor
        """
        x = self.up_sample(x)
        x = self.conv(x)
        return x
