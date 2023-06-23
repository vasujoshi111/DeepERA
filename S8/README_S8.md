# DeepERA
Learn: Deep learning, Pytorch, Computer Vision, NLP

## S8
**Objective: Train a model with less than 50K parameters for 20 epochs and get accuacy more than 70% and get intuition of Batch normalization, group normalization and layer normalization.**

We have 3 models here and in each model same architecture followed only change in normalization betwen the feature.<br>

Optimizer used: SGD<br>
Epochs: 20<br>
Lr Scheduler: Step LR<br>
Dataset: CIFAR10<br>
Batch Size: 128

#### 1. BNModel
This model uses batch normalization.<br>

---------------------------------------------------------------------
          Layer (type)               Output Shape         Param
---------------------------------------------------------------------
            Conv2d-1           [-1, 16, 32, 32]             432
              ReLU-2           [-1, 16, 32, 32]               0
       BatchNorm2d-3           [-1, 16, 32, 32]              32
           Dropout-4           [-1, 16, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]           4,608
              ReLU-6           [-1, 32, 32, 32]               0
       BatchNorm2d-7           [-1, 32, 32, 32]              64
           Dropout-8           [-1, 32, 32, 32]               0
            Conv2d-9           [-1, 16, 32, 32]             512
        MaxPool2d-10           [-1, 16, 16, 16]               0
           Conv2d-11           [-1, 16, 16, 16]           2,304
             ReLU-12           [-1, 16, 16, 16]               0
      BatchNorm2d-13           [-1, 16, 16, 16]              32
          Dropout-14           [-1, 16, 16, 16]               0
           Conv2d-15           [-1, 32, 16, 16]           4,608
             ReLU-16           [-1, 32, 16, 16]               0
      BatchNorm2d-17           [-1, 32, 16, 16]              64
          Dropout-18           [-1, 32, 16, 16]               0
           Conv2d-19           [-1, 32, 16, 16]           9,216
             ReLU-20           [-1, 32, 16, 16]               0
      BatchNorm2d-21           [-1, 32, 16, 16]              64
          Dropout-22           [-1, 32, 16, 16]               0
           Conv2d-23           [-1, 16, 16, 16]             512
        MaxPool2d-24             [-1, 16, 8, 8]               0
           Conv2d-25             [-1, 32, 8, 8]           4,608
             ReLU-26             [-1, 32, 8, 8]               0
      BatchNorm2d-27             [-1, 32, 8, 8]              64
          Dropout-28             [-1, 32, 8, 8]               0
           Conv2d-29             [-1, 32, 8, 8]           9,216
             ReLU-30             [-1, 32, 8, 8]               0
      BatchNorm2d-31             [-1, 32, 8, 8]              64
          Dropout-32             [-1, 32, 8, 8]               0
           Conv2d-33             [-1, 32, 8, 8]           9,216
             ReLU-34             [-1, 32, 8, 8]               0
      BatchNorm2d-35             [-1, 32, 8, 8]              64
          Dropout-36             [-1, 32, 8, 8]               0
     AdaptiveAvgPool2d-37        [-1, 32, 1, 1]               0
           Conv2d-38             [-1, 10, 1, 1]             320

--------------------------------------------------------------------

Number of parameters used: 46000<br>
Training accuracy: 76.29<br>
Test accuracy: 75.20<br>


#### 2. Group Normalization
This model uses group normalization with group size of 4.

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1           [-1, 16, 32, 32]             432
              ReLU-2           [-1, 16, 32, 32]               0
         GroupNorm-3           [-1, 16, 32, 32]              32
           Dropout-4           [-1, 16, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]           4,608
              ReLU-6           [-1, 32, 32, 32]               0
         GroupNorm-7           [-1, 32, 32, 32]              64
           Dropout-8           [-1, 32, 32, 32]               0
            Conv2d-9           [-1, 16, 32, 32]             512
        MaxPool2d-10           [-1, 16, 16, 16]               0
           Conv2d-11           [-1, 16, 16, 16]           2,304
             ReLU-12           [-1, 16, 16, 16]               0
        GroupNorm-13           [-1, 16, 16, 16]              32
          Dropout-14           [-1, 16, 16, 16]               0
           Conv2d-15           [-1, 32, 16, 16]           4,608
             ReLU-16           [-1, 32, 16, 16]               0
        GroupNorm-17           [-1, 32, 16, 16]              64
          Dropout-18           [-1, 32, 16, 16]               0
           Conv2d-19           [-1, 32, 16, 16]           9,216
             ReLU-20           [-1, 32, 16, 16]               0
        GroupNorm-21           [-1, 32, 16, 16]              64
          Dropout-22           [-1, 32, 16, 16]               0
           Conv2d-23           [-1, 16, 16, 16]             512
        MaxPool2d-24             [-1, 16, 8, 8]               0
           Conv2d-25             [-1, 32, 8, 8]           4,608
             ReLU-26             [-1, 32, 8, 8]               0
        GroupNorm-27             [-1, 32, 8, 8]              64
          Dropout-28             [-1, 32, 8, 8]               0
           Conv2d-29             [-1, 32, 8, 8]           9,216
             ReLU-30             [-1, 32, 8, 8]               0
        GroupNorm-31             [-1, 32, 8, 8]              64
          Dropout-32             [-1, 32, 8, 8]               0
           Conv2d-33             [-1, 32, 8, 8]           9,216
             ReLU-34             [-1, 32, 8, 8]               0
        GroupNorm-35             [-1, 32, 8, 8]              64
          Dropout-36             [-1, 32, 8, 8]               0
    AdaptiveAvgPool2d-37         [-1, 32, 1, 1]               0
           Conv2d-38             [-1, 10, 1, 1]             320

----------------------------------------------------------------

Number of parameters used: 46000<br>
Training accuracy: 75.42<br>
Test accuracy: 74.34<br>
The accuracy is less then batch normalization.

#### 3. Layer Normlization
This model uses layer normalization.

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1           [-1, 16, 32, 32]             432
              ReLU-2           [-1, 16, 32, 32]               0
         GroupNorm-3           [-1, 16, 32, 32]              32
           Dropout-4           [-1, 16, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]           4,608
              ReLU-6           [-1, 32, 32, 32]               0
         GroupNorm-7           [-1, 32, 32, 32]              64
           Dropout-8           [-1, 32, 32, 32]               0
            Conv2d-9           [-1, 16, 32, 32]             512
        MaxPool2d-10           [-1, 16, 16, 16]               0
           Conv2d-11           [-1, 16, 16, 16]           2,304
             ReLU-12           [-1, 16, 16, 16]               0
        GroupNorm-13           [-1, 16, 16, 16]              32
          Dropout-14           [-1, 16, 16, 16]               0
           Conv2d-15           [-1, 32, 16, 16]           4,608
             ReLU-16           [-1, 32, 16, 16]               0
        GroupNorm-17           [-1, 32, 16, 16]              64
          Dropout-18           [-1, 32, 16, 16]               0
           Conv2d-19           [-1, 32, 16, 16]           9,216
             ReLU-20           [-1, 32, 16, 16]               0
        GroupNorm-21           [-1, 32, 16, 16]              64
          Dropout-22           [-1, 32, 16, 16]               0
           Conv2d-23           [-1, 16, 16, 16]             512
        MaxPool2d-24             [-1, 16, 8, 8]               0
           Conv2d-25             [-1, 32, 8, 8]           4,608
             ReLU-26             [-1, 32, 8, 8]               0
        GroupNorm-27             [-1, 32, 8, 8]              64
          Dropout-28             [-1, 32, 8, 8]               0
           Conv2d-29             [-1, 32, 8, 8]           9,216
             ReLU-30             [-1, 32, 8, 8]               0
        GroupNorm-31             [-1, 32, 8, 8]              64
          Dropout-32             [-1, 32, 8, 8]               0
           Conv2d-33             [-1, 32, 8, 8]           9,216
             ReLU-34             [-1, 32, 8, 8]               0
        GroupNorm-35             [-1, 32, 8, 8]              64
          Dropout-36             [-1, 32, 8, 8]               0
    AdaptiveAvgPool2d-37         [-1, 32, 1, 1]               0
           Conv2d-38             [-1, 10, 1, 1]             320

----------------------------------------------------------------

Number of parameters used: 46000<br>
Training accuracy: 74.06<br>
Test accuracy: 72.47<br>
The accuracy is less than both Batch normalization and group normalization.