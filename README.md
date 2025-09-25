Benchmarking of machine learning models for PDE simulations on regular grids. 

The repository contains training code for the following models:

1. CNN
2. ResNet
3. Vision Transformer (ViT)
4. Fourier Neural Operator (FNO)
5. U-Net
6. PDE-Transformer

Recommended python version is 3.12.3. All dependencies are listed in `requirements.txt` file and can be installed using pip with the following command: 

```
pip install -r requirements.txt
```

Model training can be run using the following command:
```
python model_train_script.py --pde_model="advection2D_hard" --model_arch="CNN" --model_size="M" --num_epochs=20 --batch_size=5 --seeds=42,43,44,45,46
```
pde_model: PDE model type ("advection2D_hard", "burgers2D", "kuramoto2D", "fisher2D", "advdiff2D", "kolmflow2D", "grayscott2D")
model_arch: Model architecture ("CNN", "ResNet", "ViT", "FNO", "UNet", "PDET")
model_size: Model size based on parameter size ("S", "M", "L")
num_epochs: Total number of epochs for each train run
batch_size: Train/Val batch size
seeds: List of model seed value for model initialisation and training
