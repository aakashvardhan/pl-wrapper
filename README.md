# PyTorch Lightning Wrapper

## Overview
This wrapper is aimed at being used in most deep learning projects that require pytorch lightning. In this case, we used CIFAR-10 Dataset which consists of 60000 images in 10 classes at shape (32,32,3).

## Requirements
- pillow
- gradio (Required for HuggingFace)
- numpy
- torch
- torchsummary
- torch-lr-finder
- grad-cam
- pytorch-lightning
- albumentations

## Why Use PyTorch Lightning?

PyTorch Lightning offers several advantages for deep learning development:

### Simplified Training Loop
- Abstracts boilerplate code, letting you focus on model architecture and hyperparameters.

### Readability and Maintainability
- Organizes code for easier reading and maintenance, beneficial for large projects.

### Reproducibility
- Standardized training loop for consistent and reproducible results.

### Automatic Hardware Acceleration
- Auto-handles GPU or TPU acceleration.

### Distributed Training Support
- Easy integration with frameworks like PyTorch Distributed Data Parallel (DDP).

### Experiment Logging
- Integrates with platforms like TensorBoard for metrics tracking.

### Multi-Framework Support
- Compatible with PyTorch, PyTorch Lightning Bolts, and more.

### Active Community
- Large, contributing community and a range of pre-built components.

## [Data](https://github.com/aakashvardhan/pl-wrapper/tree/main/data)

### [DataLoader](https://github.com/aakashvardhan/pl-wrapper/blob/main/data/cifar10_datamodule.py)

#### Overview
`CIFAR10DataModule` is a custom PyTorch Lightning DataModule for handling the CIFAR-10 dataset. It manages data loading and preprocessing.

#### Key Features

##### Hyperparameters
- `data_dir`: Directory where CIFAR-10 dataset is stored.

##### Properties
- `num_classes`: Returns the number of classes in the CIFAR-10 dataset.
- `class_names`: Returns the names of classes.

#### Methods

##### `prepare_data()`
Downloads the CIFAR-10 dataset.

##### `setup(stage=None)`
Sets up the training and validation datasets.

##### `train_dataloader()`
Returns a DataLoader for the training set.

##### `val_dataloader()`
Returns a DataLoader for the validation set.

##### `test_dataloader()`
Returns a DataLoader for the test set.

#### Example Usage
```python
from data import CIFAR10DataModule

datamodule = CIFAR10DataModule(data_dir="./data")
datamodule.prepare_data()
datamodule.setup('fit')
train_loader = datamodule.train_dataloader()
```
## [model.py](https://github.com/aakashvardhan/pl-wrapper/blob/main/models/model.py)

### Overview
`LitModel` is a PyTorch Lightning Module that wraps around a ResNet18 model for CIFAR-10 classification.

### Initialization

#### `__init__(BATCH_SIZE, best_lr, epochs=5, num_classes=10)`
Initializes the model with the following parameters:
- `BATCH_SIZE`: Batch size for training.
- `best_lr`: Best learning rate discovered.
- `epochs`: Number of training epochs (default is 5).
- `num_classes`: Number of target classes (default is 10).

### Methods

#### `forward(x)`
Runs forward pass on the input `x`.

#### `training_step(batch, batch_idx)`
Computes the training loss and accuracy and logs them.

#### `evaluate(batch, stage=None)`
Utility function for computing loss and accuracy during validation and testing.

#### `validation_step(batch, batch_idx)`
Computes the validation loss and accuracy and logs them.

#### `test_step(batch, batch_idx)`
Computes the test loss and accuracy and logs them.

#### `configure_optimizers()`
Configures the SGD optimizer and OneCycleLR scheduler.

#### `on_train_start()`
Prints a summary of the model architecture at the start of training.

### Example Usage
```python
from models import LitModel

model = LitModel(BATCH_SIZE=256, best_lr=0.03, epochs=5)
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model)
```





## Model Summary

```
  | Name  | Type   | Params
---------------------------------
0 | model | ResNet | 11.2 M
---------------------------------
11.2 M    Trainable params
0         Non-trainable params
11.2 M    Total params
44.696    Total estimated model params size (MB)
```

### ResNet Model Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
            Conv2d-3           [-1, 64, 32, 32]          36,864
       BatchNorm2d-4           [-1, 64, 32, 32]             128
            Conv2d-5           [-1, 64, 32, 32]          36,864
       BatchNorm2d-6           [-1, 64, 32, 32]             128
        BasicBlock-7           [-1, 64, 32, 32]               0
            Conv2d-8           [-1, 64, 32, 32]          36,864
       BatchNorm2d-9           [-1, 64, 32, 32]             128
           Conv2d-10           [-1, 64, 32, 32]          36,864
      BatchNorm2d-11           [-1, 64, 32, 32]             128
       BasicBlock-12           [-1, 64, 32, 32]               0
           Conv2d-13          [-1, 128, 16, 16]          73,728
      BatchNorm2d-14          [-1, 128, 16, 16]             256
           Conv2d-15          [-1, 128, 16, 16]         147,456
      BatchNorm2d-16          [-1, 128, 16, 16]             256
           Conv2d-17          [-1, 128, 16, 16]           8,192
      BatchNorm2d-18          [-1, 128, 16, 16]             256
       BasicBlock-19          [-1, 128, 16, 16]               0
           Conv2d-20          [-1, 128, 16, 16]         147,456
      BatchNorm2d-21          [-1, 128, 16, 16]             256
           Conv2d-22          [-1, 128, 16, 16]         147,456
      BatchNorm2d-23          [-1, 128, 16, 16]             256
       BasicBlock-24          [-1, 128, 16, 16]               0
           Conv2d-25            [-1, 256, 8, 8]         294,912
      BatchNorm2d-26            [-1, 256, 8, 8]             512
           Conv2d-27            [-1, 256, 8, 8]         589,824
      BatchNorm2d-28            [-1, 256, 8, 8]             512
           Conv2d-29            [-1, 256, 8, 8]          32,768
      BatchNorm2d-30            [-1, 256, 8, 8]             512
       BasicBlock-31            [-1, 256, 8, 8]               0
           Conv2d-32            [-1, 256, 8, 8]         589,824
      BatchNorm2d-33            [-1, 256, 8, 8]             512
           Conv2d-34            [-1, 256, 8, 8]         589,824
      BatchNorm2d-35            [-1, 256, 8, 8]             512
       BasicBlock-36            [-1, 256, 8, 8]               0
           Conv2d-37            [-1, 512, 4, 4]       1,179,648
      BatchNorm2d-38            [-1, 512, 4, 4]           1,024
           Conv2d-39            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-40            [-1, 512, 4, 4]           1,024
           Conv2d-41            [-1, 512, 4, 4]         131,072
      BatchNorm2d-42            [-1, 512, 4, 4]           1,024
       BasicBlock-43            [-1, 512, 4, 4]               0
           Conv2d-44            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-45            [-1, 512, 4, 4]           1,024
           Conv2d-46            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-47            [-1, 512, 4, 4]           1,024
       BasicBlock-48            [-1, 512, 4, 4]               0
           Linear-49                   [-1, 10]           5,130
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 11.25
Params size (MB): 42.63
Estimated Total Size (MB): 53.89
----------------------------------------------------------------

```

### Training & Testing logs

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│      test_acc_epoch       │    0.9350000023841858     │
│      test_loss_epoch      │    0.22751745581626892    │
└───────────────────────────┴───────────────────────────┘
```

#### Training Loss & Acc Graph

![image](https://github.com/aakashvardhan/pl-wrapper/blob/main/train_test_loss_acc/train_graph.png)

#### Testing Loss & Acc Graph

![image](https://github.com/aakashvardhan/pl-wrapper/blob/main/train_test_loss_acc/test_graph.png)

## 10 Misclassified Images with the use of GradCam

![image](https://github.com/aakashvardhan/pl-wrapper/blob/main/gradcam.png)

