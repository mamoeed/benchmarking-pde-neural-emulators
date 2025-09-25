import torch 

class CNN(torch.nn.Module):
    def __init__(self, width: int, depth: int, inp_depth=1):
        super().__init__()
        module_list = []

        conv_lift = torch.nn.Conv2d(
            inp_depth, width, kernel_size=3, padding=1, padding_mode="circular"
        )
        module_list.append(conv_lift)
        module_list.append(torch.nn.ReLU())

        for _ in range(depth - 1):
            conv = torch.nn.Conv2d(
                width, width, kernel_size=3, padding=1, padding_mode="circular"
            )
            module_list.append(conv)
            module_list.append(torch.nn.ReLU())

        conv_proj = torch.nn.Conv2d(
            width, inp_depth, kernel_size=3, padding=1, padding_mode="circular"
        )
        module_list.append(conv_proj)

        self.sequential_modules = torch.nn.Sequential(*module_list)

    def forward(self, x):
        return self.sequential_modules(x)

def create_CNN(size="M", **kwargs):
    size_configs = {
        "S": {"width": 18, "depth": 5},
        "M": {"width": 36, "depth": 10},
        "L": {"width": 70, "depth": 25},
    }
    return CNN(**size_configs[size], **kwargs)