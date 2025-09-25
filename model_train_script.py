import os 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.autonotebook import tqdm
import numpy as np
import math
import pandas as pd
import seaborn as sb
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR
import pickle
import argparse

from pdetransformer.core.mixed_channels.pde_transformer import PDEImpl
from models.cnn import CNN
from models.resnet import ResNet
from models.vit import VisionTransformer
from neuralop.models import FNO
from models.unet import UNet

from train_function import train_model

def set_seeds(seed=42):
    np.random.seed(seed)         
    torch.manual_seed(seed)     
    torch.cuda.manual_seed(seed) 

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" 



def main():
    parser = argparse.ArgumentParser(description='Run model with different configurations')
    parser.add_argument('--pde_model', type=str, required=True, 
                       help='PDE model type ("advection2D_hard", "burgers2D", "kuramoto2D", "fisher2D", "advdiff2D", "kolmflow2D", "grayscott2D")')
    parser.add_argument('--model_arch', type=str, required=True,
                       help='Model architecture ("CNN", "ResNet", "ViT", "FNO", "UNet", "PDET")')
    parser.add_argument('--model_size', type=str, required=True,
                       help='Model size ("S", "M", "L")')
    parser.add_argument('--num_epochs', type=int, required=True,
                       help='Number of epochs ')
    parser.add_argument('--batch_size', type=int, required=True,
                       help='training batch size')
    parser.add_argument('--seeds', type=str, required=True,
                       help='List of seed values (ex input: 41,42,43)')
    
    args = parser.parse_args()

    seeds = [int(seed.strip()) for seed in args.seeds.split(',')]
    
    # Access the arguments as strings/int
    pde_model = args.pde_model
    model_arch = args.model_arch
    model_size = args.model_size
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Running with pde_model: {pde_model}, model_arch: {model_arch}, model_size: {model_size}, num_epochs: {num_epochs}, batch_size: {batch_size}, seeds: {seeds}, device: {device}")

    train_model(pde_model, model_arch, model_size, num_epochs, batch_size, seeds, device)

main()

