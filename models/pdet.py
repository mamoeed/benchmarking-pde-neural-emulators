import torch
from pdetransformer.core.mixed_channels.pde_transformer import PDEImpl


def create_PDET(size="M", **kwargs):
    size_configs = {
        "S": {
                "sample_size": 128, 
                "patch_size": 4, 
                "window_size": 10, 
                "down_factor": 4, 
                "hidden_size" : 4, 
                "max_hidden_size" : 4,
                "depth" : [1, 1, 1], 
                "num_heads" : 2, 
                "mlp_ratio":2,
                "class_dropout_prob" : 0.1,
                "num_classes":1,
                "periodic":True,
                "carrier_token_active": False
        },
        "M": {
                "sample_size": 128, 
                "patch_size": 4, 
                "window_size": 10, 
                "down_factor": 4, 
                "hidden_size" : 12, 
                "max_hidden_size" : 48,
                "depth" : [1, 5, 1], 
                "num_heads" : 4, 
                "mlp_ratio":2,
                "class_dropout_prob" : 0.1,
                "num_classes":1,
                "periodic":True,
                "carrier_token_active": False
        },
        "L": {
                "sample_size": 128, 
                "patch_size": 4, 
                "window_size": 10, 
                "down_factor": 4, 
                "hidden_size" : 16, 
                "max_hidden_size" : 80,
                "depth" : [1, 4, 8, 4, 1], 
                "num_heads" : 16, 
                "mlp_ratio":4,
                "class_dropout_prob" : 0.1,
                "num_classes":1,
                "periodic":True,
                "carrier_token_active": False
        },
    }
    return PDEImpl(**size_configs[size], **kwargs)