import torch
from neuralop.models import FNO

def create_FNO(size="M", **kwargs):
    size_configs = {
        "S": {
            "n_modes":(10,10),
            "hidden_channels":12,  # width
            "n_layers": 3, # num of fourier layers (spectral conv layers + skip + nonlin)
        },
        "M": {
            "n_modes":(16,16),
            "hidden_channels":20,  # width
            "n_layers": 4, # num of fourier layers (spectral conv layers + skip + nonlin)
        },
        "L": {
            "n_modes":(20,20),
            "hidden_channels":42,  # width
            "n_layers": 6, # num of fourier layers (spectral conv layers + skip + nonlin)
        },
    }
    return FNO(**size_configs[size], **kwargs, factorization='tucker', rank=0.42)