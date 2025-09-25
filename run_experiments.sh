#!/bin/bash

# RUN all experiments

#######################################################
#################### MEDIUM SIZE ######################
#######################################################
# advection2D_hard

python model_train_script.py --pde_model="advection2D_hard" --model_arch="CNN" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_advection2D_hard_CNN.txt && echo "$(date): Command completed" >> logs/M_advection2D_hard_CNN.txt

python model_train_script.py --pde_model="advection2D_hard" --model_arch="ResNet" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_advection2D_hard_ResNet.txt && echo "$(date): Command completed" >> logs/M_advection2D_hard_ResNet.txt

python model_train_script.py --pde_model="advection2D_hard" --model_arch="ViT" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_advection2D_hard_ViT.txt && echo "$(date): Command completed" >> logs/M_advection2D_hard_ViT.txt

python model_train_script.py --pde_model="advection2D_hard" --model_arch="FNO" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_advection2D_hard_FNO.txt && echo "$(date): Command completed" >> logs/M_advection2D_hard_FNO.txt

python model_train_script.py --pde_model="advection2D_hard" --model_arch="UNet" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_advection2D_hard_UNet.txt && echo "$(date): Command completed" >> logs/M_advection2D_hard_UNet.txt

python model_train_script.py --pde_model="advection2D_hard" --model_arch="PDET" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_advection2D_hard_PDET.txt && echo "$(date): Command completed" >> logs/M_advection2D_hard_PDET.txt



# kuramoto2D

python model_train_script.py --pde_model="kuramoto2D" --model_arch="CNN" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_kuramoto2D_CNN.txt && echo "$(date): Command completed" >> logs/M_kuramoto2D_CNN.txt

python model_train_script.py --pde_model="kuramoto2D" --model_arch="ResNet" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_kuramoto2D_ResNet.txt && echo "$(date): Command completed" >> logs/M_kuramoto2D_ResNet.txt

python model_train_script.py --pde_model="kuramoto2D" --model_arch="ViT" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_kuramoto2D_ViT.txt && echo "$(date): Command completed" >> logs/M_kuramoto2D_ViT.txt

python model_train_script.py --pde_model="kuramoto2D" --model_arch="FNO" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_kuramoto2D_FNO.txt && echo "$(date): Command completed" >> logs/M_kuramoto2D_FNO.txt

python model_train_script.py --pde_model="kuramoto2D" --model_arch="UNet" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_kuramoto2D_UNet.txt && echo "$(date): Command completed" >> logs/M_kuramoto2D_UNet.txt

python model_train_script.py --pde_model="kuramoto2D" --model_arch="PDET" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_kuramoto2D_PDET.txt && echo "$(date): Command completed" >> logs/M_kuramoto2D_PDET.txt


# fisher2D

python model_train_script.py --pde_model="fisher2D" --model_arch="CNN" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_fisher2D_CNN.txt && echo "$(date): Command completed" >> logs/M_fisher2D_CNN.txt

python model_train_script.py --pde_model="fisher2D" --model_arch="ResNet" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_fisher2D_ResNet.txt && echo "$(date): Command completed" >> logs/M_fisher2D_ResNet.txt

python model_train_script.py --pde_model="fisher2D" --model_arch="ViT" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_fisher2D_ViT.txt && echo "$(date): Command completed" >> logs/M_fisher2D_ViT.txt

python model_train_script.py --pde_model="fisher2D" --model_arch="FNO" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_fisher2D_FNO.txt && echo "$(date): Command completed" >> logs/M_fisher2D_FNO.txt

python model_train_script.py --pde_model="fisher2D" --model_arch="UNet" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_fisher2D_UNet.txt && echo "$(date): Command completed" >> logs/M_fisher2D_UNet.txt

python model_train_script.py --pde_model="fisher2D" --model_arch="PDET" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_fisher2D_PDET.txt && echo "$(date): Command completed" >> logs/M_fisher2D_PDET.txt


# kolmflow2D

python model_train_script.py --pde_model="kolmflow2D" --model_arch="CNN" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_kolmflow2D_CNN.txt && echo "$(date): Command completed" >> logs/M_kolmflow2D_CNN.txt

python model_train_script.py --pde_model="kolmflow2D" --model_arch="ResNet" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_kolmflow2D_ResNet.txt && echo "$(date): Command completed" >> logs/M_kolmflow2D_ResNet.txt

python model_train_script.py --pde_model="kolmflow2D" --model_arch="ViT" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_kolmflow2D_ViT.txt && echo "$(date): Command completed" >> logs/M_kolmflow2D_ViT.txt

python model_train_script.py --pde_model="kolmflow2D" --model_arch="FNO" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_kolmflow2D_FNO.txt && echo "$(date): Command completed" >> logs/M_kolmflow2D_FNO.txt

python model_train_script.py --pde_model="kolmflow2D" --model_arch="UNet" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_kolmflow2D_UNet.txt && echo "$(date): Command completed" >> logs/M_kolmflow2D_UNet.txt

python model_train_script.py --pde_model="kolmflow2D" --model_arch="PDET" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_kolmflow2D_PDET.txt && echo "$(date): Command completed" >> logs/M_kolmflow2D_PDET.txt



# grayscott2D

python model_train_script.py --pde_model="grayscott2D" --model_arch="CNN" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_grayscott2D_CNN.txt && echo "$(date): Command completed" >> logs/M_grayscott2D_CNN.txt

python model_train_script.py --pde_model="grayscott2D" --model_arch="ResNet" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_grayscott2D_ResNet.txt && echo "$(date): Command completed" >> logs/M_grayscott2D_ResNet.txt

python model_train_script.py --pde_model="grayscott2D" --model_arch="ViT" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_grayscott2D_ViT.txt && echo "$(date): Command completed" >> logs/M_grayscott2D_ViT.txt

python model_train_script.py --pde_model="grayscott2D" --model_arch="FNO" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_grayscott2D_FNO.txt && echo "$(date): Command completed" >> logs/M_grayscott2D_FNO.txt

python model_train_script.py --pde_model="grayscott2D" --model_arch="UNet" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_grayscott2D_UNet.txt && echo "$(date): Command completed" >> logs/M_grayscott2D_UNet.txt

python model_train_script.py --pde_model="grayscott2D" --model_arch="PDET" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_grayscott2D_PDET.txt && echo "$(date): Command completed" >> logs/M_grayscott2D_PDET.txt


# burgers2D

python model_train_script.py --pde_model="burgers2D" --model_arch="CNN" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_burgers2D_CNN.txt && echo "$(date): Command completed" >> logs/M_burgers2D_CNN.txt

python model_train_script.py --pde_model="burgers2D" --model_arch="ResNet" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_burgers2D_ResNet.txt && echo "$(date): Command completed" >> logs/M_burgers2D_ResNet.txt

python model_train_script.py --pde_model="burgers2D" --model_arch="ViT" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_burgers2D_ViT.txt && echo "$(date): Command completed" >> logs/M_burgers2D_ViT.txt

python model_train_script.py --pde_model="burgers2D" --model_arch="FNO" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_burgers2D_FNO.txt && echo "$(date): Command completed" >> logs/M_burgers2D_FNO.txt

python model_train_script.py --pde_model="burgers2D" --model_arch="UNet" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_burgers2D_UNet.txt && echo "$(date): Command completed" >> logs/M_burgers2D_UNet.txt

python model_train_script.py --pde_model="burgers2D" --model_arch="PDET" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/M_burgers2D_PDET.txt && echo "$(date): Command completed" >> logs/M_burgers2D_PDET.txt









################################################################
######################## SMALL SIZE ############################
################################################################
# advection2D_hard

python model_train_script.py --pde_model="advection2D_hard" --model_arch="CNN" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_advection2D_hard_CNN.txt && echo "$(date): Command completed" >> logs/S_advection2D_hard_CNN.txt

python model_train_script.py --pde_model="advection2D_hard" --model_arch="ResNet" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_advection2D_hard_ResNet.txt && echo "$(date): Command completed" >> logs/S_advection2D_hard_ResNet.txt

python model_train_script.py --pde_model="advection2D_hard" --model_arch="ViT" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_advection2D_hard_ViT.txt && echo "$(date): Command completed" >> logs/S_advection2D_hard_ViT.txt

python model_train_script.py --pde_model="advection2D_hard" --model_arch="FNO" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_advection2D_hard_FNO.txt && echo "$(date): Command completed" >> logs/S_advection2D_hard_FNO.txt

python model_train_script.py --pde_model="advection2D_hard" --model_arch="UNet" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_advection2D_hard_UNet.txt && echo "$(date): Command completed" >> logs/S_advection2D_hard_UNet.txt

python model_train_script.py --pde_model="advection2D_hard" --model_arch="PDET" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_advection2D_hard_PDET.txt && echo "$(date): Command completed" >> logs/S_advection2D_hard_PDET.txt



# kuramoto2D

python model_train_script.py --pde_model="kuramoto2D" --model_arch="CNN" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_kuramoto2D_CNN.txt && echo "$(date): Command completed" >> logs/S_kuramoto2D_CNN.txt

python model_train_script.py --pde_model="kuramoto2D" --model_arch="ResNet" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_kuramoto2D_ResNet.txt && echo "$(date): Command completed" >> logs/S_kuramoto2D_ResNet.txt

python model_train_script.py --pde_model="kuramoto2D" --model_arch="ViT" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_kuramoto2D_ViT.txt && echo "$(date): Command completed" >> logs/S_kuramoto2D_ViT.txt

python model_train_script.py --pde_model="kuramoto2D" --model_arch="FNO" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_kuramoto2D_FNO.txt && echo "$(date): Command completed" >> logs/S_kuramoto2D_FNO.txt

python model_train_script.py --pde_model="kuramoto2D" --model_arch="UNet" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_kuramoto2D_UNet.txt && echo "$(date): Command completed" >> logs/S_kuramoto2D_UNet.txt

python model_train_script.py --pde_model="kuramoto2D" --model_arch="PDET" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_kuramoto2D_PDET.txt && echo "$(date): Command completed" >> logs/S_kuramoto2D_PDET.txt


# fisher2D

python model_train_script.py --pde_model="fisher2D" --model_arch="CNN" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_fisher2D_CNN.txt && echo "$(date): Command completed" >> logs/S_fisher2D_CNN.txt

python model_train_script.py --pde_model="fisher2D" --model_arch="ResNet" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_fisher2D_ResNet.txt && echo "$(date): Command completed" >> logs/S_fisher2D_ResNet.txt

python model_train_script.py --pde_model="fisher2D" --model_arch="ViT" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_fisher2D_ViT.txt && echo "$(date): Command completed" >> logs/S_fisher2D_ViT.txt

python model_train_script.py --pde_model="fisher2D" --model_arch="FNO" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_fisher2D_FNO.txt && echo "$(date): Command completed" >> logs/S_fisher2D_FNO.txt

python model_train_script.py --pde_model="fisher2D" --model_arch="UNet" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_fisher2D_UNet.txt && echo "$(date): Command completed" >> logs/S_fisher2D_UNet.txt

python model_train_script.py --pde_model="fisher2D" --model_arch="PDET" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_fisher2D_PDET.txt && echo "$(date): Command completed" >> logs/S_fisher2D_PDET.txt


# kolmflow2D

python model_train_script.py --pde_model="kolmflow2D" --model_arch="CNN" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_kolmflow2D_CNN.txt && echo "$(date): Command completed" >> logs/S_kolmflow2D_CNN.txt

python model_train_script.py --pde_model="kolmflow2D" --model_arch="ResNet" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_kolmflow2D_ResNet.txt && echo "$(date): Command completed" >> logs/S_kolmflow2D_ResNet.txt

python model_train_script.py --pde_model="kolmflow2D" --model_arch="ViT" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_kolmflow2D_ViT.txt && echo "$(date): Command completed" >> logs/S_kolmflow2D_ViT.txt

python model_train_script.py --pde_model="kolmflow2D" --model_arch="FNO" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_kolmflow2D_FNO.txt && echo "$(date): Command completed" >> logs/S_kolmflow2D_FNO.txt

python model_train_script.py --pde_model="kolmflow2D" --model_arch="UNet" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_kolmflow2D_UNet.txt && echo "$(date): Command completed" >> logs/S_kolmflow2D_UNet.txt

python model_train_script.py --pde_model="kolmflow2D" --model_arch="PDET" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_kolmflow2D_PDET.txt && echo "$(date): Command completed" >> logs/S_kolmflow2D_PDET.txt



# grayscott2D

python model_train_script.py --pde_model="grayscott2D" --model_arch="CNN" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_grayscott2D_CNN.txt && echo "$(date): Command completed" >> logs/S_grayscott2D_CNN.txt

python model_train_script.py --pde_model="grayscott2D" --model_arch="ResNet" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_grayscott2D_ResNet.txt && echo "$(date): Command completed" >> logs/S_grayscott2D_ResNet.txt

python model_train_script.py --pde_model="grayscott2D" --model_arch="ViT" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_grayscott2D_ViT.txt && echo "$(date): Command completed" >> logs/S_grayscott2D_ViT.txt

python model_train_script.py --pde_model="grayscott2D" --model_arch="FNO" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_grayscott2D_FNO.txt && echo "$(date): Command completed" >> logs/S_grayscott2D_FNO.txt

python model_train_script.py --pde_model="grayscott2D" --model_arch="UNet" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_grayscott2D_UNet.txt && echo "$(date): Command completed" >> logs/S_grayscott2D_UNet.txt

python model_train_script.py --pde_model="grayscott2D" --model_arch="PDET" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_grayscott2D_PDET.txt && echo "$(date): Command completed" >> logs/S_grayscott2D_PDET.txt


# burgers2D

python model_train_script.py --pde_model="burgers2D" --model_arch="CNN" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_burgers2D_CNN.txt && echo "$(date): Command completed" >> logs/S_burgers2D_CNN.txt

python model_train_script.py --pde_model="burgers2D" --model_arch="ResNet" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_burgers2D_ResNet.txt && echo "$(date): Command completed" >> logs/S_burgers2D_ResNet.txt

python model_train_script.py --pde_model="burgers2D" --model_arch="ViT" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_burgers2D_ViT.txt && echo "$(date): Command completed" >> logs/S_burgers2D_ViT.txt

python model_train_script.py --pde_model="burgers2D" --model_arch="FNO" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_burgers2D_FNO.txt && echo "$(date): Command completed" >> logs/S_burgers2D_FNO.txt

python model_train_script.py --pde_model="burgers2D" --model_arch="UNet" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_burgers2D_UNet.txt && echo "$(date): Command completed" >> logs/S_burgers2D_UNet.txt

python model_train_script.py --pde_model="burgers2D" --model_arch="PDET" --model_size="S" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/S_burgers2D_PDET.txt && echo "$(date): Command completed" >> logs/S_burgers2D_PDET.txt













####################################################
################### LARGE SIZE #####################
####################################################
# advection2D_hard

python model_train_script.py --pde_model="advection2D_hard" --model_arch="CNN" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_advection2D_hard_CNN.txt && echo "$(date): Command completed" >> logs/L_advection2D_hard_CNN.txt

python model_train_script.py --pde_model="advection2D_hard" --model_arch="ResNet" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_advection2D_hard_ResNet.txt && echo "$(date): Command completed" >> logs/L_advection2D_hard_ResNet.txt

python model_train_script.py --pde_model="advection2D_hard" --model_arch="ViT" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_advection2D_hard_ViT.txt && echo "$(date): Command completed" >> logs/L_advection2D_hard_ViT.txt

python model_train_script.py --pde_model="advection2D_hard" --model_arch="FNO" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_advection2D_hard_FNO.txt && echo "$(date): Command completed" >> logs/L_advection2D_hard_FNO.txt

python model_train_script.py --pde_model="advection2D_hard" --model_arch="UNet" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_advection2D_hard_UNet.txt && echo "$(date): Command completed" >> logs/L_advection2D_hard_UNet.txt

python model_train_script.py --pde_model="advection2D_hard" --model_arch="PDET" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_advection2D_hard_PDET.txt && echo "$(date): Command completed" >> logs/L_advection2D_hard_PDET.txt



# kuramoto2D

python model_train_script.py --pde_model="kuramoto2D" --model_arch="CNN" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_kuramoto2D_CNN.txt && echo "$(date): Command completed" >> logs/L_kuramoto2D_CNN.txt

python model_train_script.py --pde_model="kuramoto2D" --model_arch="ResNet" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_kuramoto2D_ResNet.txt && echo "$(date): Command completed" >> logs/L_kuramoto2D_ResNet.txt

python model_train_script.py --pde_model="kuramoto2D" --model_arch="ViT" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_kuramoto2D_ViT.txt && echo "$(date): Command completed" >> logs/L_kuramoto2D_ViT.txt

python model_train_script.py --pde_model="kuramoto2D" --model_arch="FNO" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_kuramoto2D_FNO.txt && echo "$(date): Command completed" >> logs/L_kuramoto2D_FNO.txt

python model_train_script.py --pde_model="kuramoto2D" --model_arch="UNet" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_kuramoto2D_UNet.txt && echo "$(date): Command completed" >> logs/L_kuramoto2D_UNet.txt

python model_train_script.py --pde_model="kuramoto2D" --model_arch="PDET" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_kuramoto2D_PDET.txt && echo "$(date): Command completed" >> logs/L_kuramoto2D_PDET.txt


# fisher2D

python model_train_script.py --pde_model="fisher2D" --model_arch="CNN" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_fisher2D_CNN.txt && echo "$(date): Command completed" >> logs/L_fisher2D_CNN.txt

python model_train_script.py --pde_model="fisher2D" --model_arch="ResNet" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_fisher2D_ResNet.txt && echo "$(date): Command completed" >> logs/L_fisher2D_ResNet.txt

python model_train_script.py --pde_model="fisher2D" --model_arch="ViT" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_fisher2D_ViT.txt && echo "$(date): Command completed" >> logs/L_fisher2D_ViT.txt

python model_train_script.py --pde_model="fisher2D" --model_arch="FNO" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_fisher2D_FNO.txt && echo "$(date): Command completed" >> logs/L_fisher2D_FNO.txt

python model_train_script.py --pde_model="fisher2D" --model_arch="UNet" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_fisher2D_UNet.txt && echo "$(date): Command completed" >> logs/L_fisher2D_UNet.txt

python model_train_script.py --pde_model="fisher2D" --model_arch="PDET" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_fisher2D_PDET.txt && echo "$(date): Command completed" >> logs/L_fisher2D_PDET.txt


# kolmflow2D

python model_train_script.py --pde_model="kolmflow2D" --model_arch="CNN" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_kolmflow2D_CNN.txt && echo "$(date): Command completed" >> logs/L_kolmflow2D_CNN.txt

python model_train_script.py --pde_model="kolmflow2D" --model_arch="ResNet" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_kolmflow2D_ResNet.txt && echo "$(date): Command completed" >> logs/L_kolmflow2D_ResNet.txt

python model_train_script.py --pde_model="kolmflow2D" --model_arch="ViT" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_kolmflow2D_ViT.txt && echo "$(date): Command completed" >> logs/L_kolmflow2D_ViT.txt

python model_train_script.py --pde_model="kolmflow2D" --model_arch="FNO" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_kolmflow2D_FNO.txt && echo "$(date): Command completed" >> logs/L_kolmflow2D_FNO.txt

python model_train_script.py --pde_model="kolmflow2D" --model_arch="UNet" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_kolmflow2D_UNet.txt && echo "$(date): Command completed" >> logs/L_kolmflow2D_UNet.txt

python model_train_script.py --pde_model="kolmflow2D" --model_arch="PDET" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_kolmflow2D_PDET.txt && echo "$(date): Command completed" >> logs/L_kolmflow2D_PDET.txt



# grayscott2D

python model_train_script.py --pde_model="grayscott2D" --model_arch="CNN" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_grayscott2D_CNN.txt && echo "$(date): Command completed" >> logs/L_grayscott2D_CNN.txt

python model_train_script.py --pde_model="grayscott2D" --model_arch="ResNet" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_grayscott2D_ResNet.txt && echo "$(date): Command completed" >> logs/L_grayscott2D_ResNet.txt

python model_train_script.py --pde_model="grayscott2D" --model_arch="ViT" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_grayscott2D_ViT.txt && echo "$(date): Command completed" >> logs/L_grayscott2D_ViT.txt

python model_train_script.py --pde_model="grayscott2D" --model_arch="FNO" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_grayscott2D_FNO.txt && echo "$(date): Command completed" >> logs/L_grayscott2D_FNO.txt

python model_train_script.py --pde_model="grayscott2D" --model_arch="UNet" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_grayscott2D_UNet.txt && echo "$(date): Command completed" >> logs/L_grayscott2D_UNet.txt

python model_train_script.py --pde_model="grayscott2D" --model_arch="PDET" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_grayscott2D_PDET.txt && echo "$(date): Command completed" >> logs/L_grayscott2D_PDET.txt



# burgers2D

python model_train_script.py --pde_model="burgers2D" --model_arch="CNN" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_burgers2D_CNN.txt && echo "$(date): Command completed" >> logs/L_burgers2D_CNN.txt

python model_train_script.py --pde_model="burgers2D" --model_arch="ResNet" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_burgers2D_ResNet.txt && echo "$(date): Command completed" >> logs/L_burgers2D_ResNet.txt

python model_train_script.py --pde_model="burgers2D" --model_arch="ViT" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_burgers2D_ViT.txt && echo "$(date): Command completed" >> logs/L_burgers2D_ViT.txt

python model_train_script.py --pde_model="burgers2D" --model_arch="FNO" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_burgers2D_FNO.txt && echo "$(date): Command completed" >> logs/L_burgers2D_FNO.txt

python model_train_script.py --pde_model="burgers2D" --model_arch="UNet" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_burgers2D_UNet.txt && echo "$(date): Command completed" >> logs/L_burgers2D_UNet.txt

python model_train_script.py --pde_model="burgers2D" --model_arch="PDET" --model_size="L" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46  2>&1 | tee logs/L_burgers2D_PDET.txt && echo "$(date): Command completed" >> logs/L_burgers2D_PDET.txt


