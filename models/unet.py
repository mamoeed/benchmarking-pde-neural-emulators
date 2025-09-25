import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(torch.nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet architecture based on Ronneberger et al. 
    
    Args:
        in_channels (int): num of input channels
        out_channels (int): num of output channels  
        depth (int): number of down and up sampling levels. Default: 4
        base_channels (int): num of channels in the first layer. Default: 64
        dropout_prob (float): dropout at end of contracting path
    """
    
    def __init__(self, in_channels, out_channels, depth=4, base_channels=64, dropout_prob=0.0):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.base_channels = base_channels
        self.dropout_prob = dropout_prob

        # Initial convolution
        self.inc = DoubleConv(in_channels, base_channels)
        
        # Encoder (downsampling path)
        self.downs = nn.ModuleList()
        channels = base_channels
        for i in range(depth):
            self.downs.append(Down(channels, channels * 2))
            channels *= 2
        
        # Dropout at the end of contracting path (as per original paper)
        if dropout_prob > 0:
            self.dropout = nn.Dropout2d(dropout_prob)
        else:
            self.dropout = None
            
        # Decoder (upsampling path)
        self.ups = nn.ModuleList()
        for i in range(depth):
            self.ups.append(Up(channels, channels // 2))
            channels //= 2
            
        # Output convolution
        self.outc = OutConv(base_channels, out_channels)

    def forward(self, x):
        # Store skip connections
        skip_connections = []
        
        # Initial convolution
        x = self.inc(x)
        skip_connections.append(x)
        
        # Encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
        
        # Apply dropout at the end of contracting path (bottleneck)
        if self.dropout is not None:
            x = self.dropout(x)
        
        # Remove the last skip connection (bottom of U)
        skip_connections.pop()
        
        # Decoder path
        for up in self.ups:
            skip = skip_connections.pop()
            x = up(x, skip)
        
        # Final output
        logits = self.outc(x)
        return logits


def create_UNet(size="M", **kwargs):
    size_configs = {
        "S": {"depth": 2, "base_channels": 5, "dropout_prob": 0.1},
        "M": {"depth": 3, "base_channels": 8, "dropout_prob": 0.1},
        "L": {"depth": 4, "base_channels": 12, "dropout_prob": 0.1},
    }
    return UNet(**size_configs[size], **kwargs)

