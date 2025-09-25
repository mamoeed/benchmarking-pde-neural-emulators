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
import copy
import argparse


from pdetransformer.core.mixed_channels.pde_transformer import PDEImpl
from models.pdet import create_PDET
from models.cnn import CNN, create_CNN
from models.resnet import ResNet, create_ResNet
from models.vit import VisionTransformer, create_VisionTransformer
from neuralop.models import FNO
from models.fno import create_FNO
from models.unet import UNet, create_UNet


def set_seeds(seed=42):
    np.random.seed(seed)         
    torch.manual_seed(seed)     
    torch.cuda.manual_seed(seed) 

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" 

def train_model(pde_model, model_arch, model_size, num_epochs, batch_size, seeds):
    # num_epochs = 2
    # batch_size = 5
    
    # load data (already saved)
    train_data = torch.from_numpy(np.load(f"datasets/{pde_model}/{pde_model}_train_data_128x128_100ic_50t.npy"))
    val_data = torch.from_numpy(np.load(f"datasets/{pde_model}/{pde_model}_val_data_128x128_10ic_50t.npy"))
    test_set = torch.from_numpy(np.load(f"datasets/{pde_model}/{pde_model}_test_data_128x128_30ic_200t.npy"))
    # dimensions: (num_samples , time_horizon , 1, spatial_points, spatial_points)

    
    # train_data = train_data[:10,:,:,:,:]
    # val_data = val_data
    # test_set = test_set[:3,:,:,:,:]
    print('train data:',train_data.shape, '\nval data',val_data.shape, '\ntest data',  test_set.shape)

    # preparing sequential data for training by matching each value at i with value i+1. Adding one more dimension

    substacked_data = torch.stack(
        [
            torch.stack([td[i : i + 2] for i in range(td.shape[0] - 1)])
            for td in train_data
        ]
    )
    print(substacked_data.shape)
    train_windows = substacked_data.flatten(0, 1)
    print(train_windows.shape)
    
    # split into inputs (x) and targets (y)
    inputs_df, targets_df = train_windows[:, 0], train_windows[:, 1]
    
    print("train_X:",inputs_df.shape, "train_Y:", targets_df.shape)
    
    val_substacked_data = torch.stack(
        [
            torch.stack([td[i : i + 2] for i in range(td.shape[0] - 1)])
            for td in val_data
        ]
    )
    print(val_substacked_data.shape)
    val_windows = val_substacked_data.flatten(0, 1)
    print(val_windows.shape)
    
    # split into inputs (x) and targets (y)
    val_inputs_df, val_targets_df = val_windows[:, 0], val_windows[:, 1]
    
    print("val_X", val_inputs_df.shape, "val_Y", val_targets_df.shape)


    model_dict = {}
    if pde_model in ["burgers2D",  "grayscott2D"]:
        input_channels = 2
    else:
        input_channels = 1

    lrs = [1e-3]
    # batch_sizes = [20,40,64]
    # optimizers = []
    for seed in seeds:
        for l in lrs:
            set_seeds(seed)
            if model_arch == "PDET":
                model = create_PDET(model_size,in_channels = input_channels,out_channels = input_channels).to("cuda")
            elif model_arch == "CNN":
                model = create_CNN(model_size, inp_depth=input_channels).to("cuda")
            elif model_arch == "ResNet":
                model = create_ResNet(model_size, inp_depth=input_channels).to("cuda")
            elif model_arch == "UNet":
                model = create_UNet(model_size,in_channels=input_channels, out_channels=input_channels).to("cuda")
            elif model_arch == "ViT":
                model = create_VisionTransformer(model_size,in_channels = input_channels, out_channels = input_channels).to("cuda")
            elif model_arch == "FNO":
                model = create_FNO(model_size,in_channels = input_channels, out_channels = input_channels).to("cuda")
            else:
                print("model not defined")
            
            print(model_arch,"initialised with parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
            print('SEED:', seed)
        
    
            dataloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(inputs_df, targets_df),
                batch_size=batch_size,
                shuffle=True,
            )
            val_dataloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(inputs_df, targets_df),
                batch_size=batch_size,
                shuffle=True,
            )
            
            optimizer = torch.optim.Adam(model.parameters(), lr=l)
            loss_history = []
            best_loss = float('inf')
            best_model_state = None
        
            for epoch in tqdm(range(int(num_epochs))):
                model.train()
                for inputs, targets in dataloader:
                    inputs, targets = inputs.to("cuda"), targets.to("cuda")
                    
                    optimizer.zero_grad()
                    if model_arch == "PDET":
                        outputs = model(inputs, None, None)
                    else:
                        outputs = model(inputs)
                    loss = torch.nn.functional.mse_loss(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    loss_history.append(loss.item())
                # print(loss.item())
    
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    total_val_loss = 0
                    total_data_size = 0
                    # calculate loss somehow:
                    for inputs, targets in val_dataloader:
                        inputs, targets = inputs.to("cuda"), targets.to("cuda")
                        if model_arch == "PDET":
                            outputs = model(inputs, None, None)
                        else:
                            outputs = model(inputs)
                        total_val_loss += torch.nn.functional.mse_loss(outputs, targets)
                        total_data_size += inputs.shape[0]
                        
                    val_loss = total_val_loss/total_data_size
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_model_state = copy.deepcopy(model.state_dict())
                        print(f"New best model at epoch {epoch+1} with loss: {val_loss}")
    
            # load best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            model_dict[seed] = {'loss_history': np.array(loss_history)}
            
            initial = test_set[:, 0, :, :]  # Shape: (30, H, W)
            model_rollouts = []  # Will store rollouts for each of the 30 items
            model.eval()
            
            # Process each item in the first dimension separately
            for i in range(test_set.size(0)):  # Loop through all 30 items
                current_state = initial[i:i+1]  # Shape: (1, H, W), keep batch dimension
                single_rollout = []
                
                for _ in range(100):
                    with torch.no_grad():
                        # current_state = torch.cat((current_state,torch.zeros_like(current_state)),dim=1)
                        if model_arch == "PDET":
                            current_state = model(current_state.to("cuda"),None,None)[:,:,:,:]
                        else:
                            current_state = model(current_state.to("cuda"))[:,:,:,:]
                    single_rollout.append(current_state.cpu())
                
                # Stack the 100 timesteps for this item
                single_rollout = torch.stack(single_rollout, dim=1)  # Shape: (1, 200, H, W)
                model_rollouts.append(single_rollout)
            
            # Combine all rollouts
            model_dict[seed]['model_rollout'] = torch.cat(model_rollouts, dim=0) # Shape: (30, 100, H, W)

            print('saved model rollouts with shape:',model_dict[seed]['model_rollout'].shape)
            
            # Calculate mean nRMSE (normalized RMSE)
            model_dict[seed]['mean_rollout_nRMSE'] = torch.sum(
                torch.pow(
                    torch.sum(torch.pow(model_dict[seed]['model_rollout'] - test_set[:,1:101,:,:], 2), dim=(2,3,4)) / 
                    torch.sum(torch.pow(test_set[:,1:101,:,:], 2), dim=(2,3,4)), 0.5
                ), dim=0
            ) * (1/test_set.shape[0])


    directory = f"results/{pde_model}_model_seed_experiments/{model_size}/{model_arch}/"
    os.makedirs(directory, exist_ok=True)
    
    with open(f"{directory}data.pkl", 'wb') as f:
        pickle.dump(model_dict, f)

    
