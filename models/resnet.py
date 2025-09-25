import torch 

class ResidualBlock(torch.nn.Module):
    """Basic residual block for the middle layers where channels = width"""
    def __init__(self, width: int):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            width, width, kernel_size=3, padding=1, padding_mode="circular"
        )
        self.conv2 = torch.nn.Conv2d(
            width, width, kernel_size=3, padding=1, padding_mode="circular"
        )
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        identity = x  # Save input for skip connection
        
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        
        # Add skip connection: F(x) + x
        out = out + identity
        out = self.relu(out)
        
        return out

class ResNet(torch.nn.Module):
    def __init__(self, width: int, depth: int, inp_depth=1):
        super().__init__()
        
        # Lift: inp_depth -> width (no skip connection here due to dimension change)
        self.conv_lift = torch.nn.Conv2d(
            inp_depth, width, kernel_size=3, padding=1, padding_mode="circular"
        )
        self.relu = torch.nn.ReLU()
        
        # Middle layers as residual blocks (width -> width)
        self.residual_blocks = torch.nn.ModuleList()
        num_residual_blocks = (depth - 1) // 2  # Group pairs of layers into blocks
        
        for _ in range(num_residual_blocks):
            self.residual_blocks.append(ResidualBlock(width))
        
        # Handle remaining layers if depth-1 is odd
        remaining_layers = (depth - 1) % 2
        if remaining_layers > 0:
            self.extra_conv = torch.nn.Conv2d(
                width, width, kernel_size=3, padding=1, padding_mode="circular"
            )
        else:
            self.extra_conv = None
            
        # Projection: width -> inp_depth (no skip connection due to dimension change)
        self.conv_proj = torch.nn.Conv2d(
            width, inp_depth, kernel_size=3, padding=1, padding_mode="circular"
        )
    
    def forward(self, x):
        # Lift
        x = self.relu(self.conv_lift(x))
        
        # Pass through residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Extra layer if needed
        if self.extra_conv is not None:
            x = self.relu(self.extra_conv(x))
            
        # Projection
        x = self.conv_proj(x)
        
        return x



def create_ResNet(size="M", **kwargs):
    size_configs = {
        "S": {"width": 18, "depth": 5},
        "M": {"width": 36, "depth": 10},
        "L": {"width": 70, "depth": 25},
    }
    return ResNet(**size_configs[size], **kwargs)