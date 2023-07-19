# DeepERA
Learn: Deep learning, Pytorch, Computer Vision, NLP

## S8
**Objective: Train a model with less than 200K parameters for as many as epochs and get accuacy more than 85% and get intuition of different convolutions.**

Optimizer used: SGD<br>
Lr Scheduler: ONeCycle LR<br>
Dataset: CIFAR10<br>
Batch Size: 128

#### 1. Model

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1           [-1, 32, 32, 32]             864
              ReLU-2           [-1, 32, 32, 32]               0
       BatchNorm2d-3           [-1, 32, 32, 32]              64
           Dropout-4           [-1, 32, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]           9,216
              ReLU-6           [-1, 32, 32, 32]               0
       BatchNorm2d-7           [-1, 32, 32, 32]              64
           Dropout-8           [-1, 32, 32, 32]               0
            Conv2d-9           [-1, 32, 28, 28]           9,216
             ReLU-10           [-1, 32, 28, 28]               0
      BatchNorm2d-11           [-1, 32, 28, 28]              64
          Dropout-12           [-1, 32, 28, 28]               0
           Conv2d-13           [-1, 32, 28, 28]           9,216
             ReLU-14           [-1, 32, 28, 28]               0
      BatchNorm2d-15           [-1, 32, 28, 28]              64
          Dropout-16           [-1, 32, 28, 28]               0
           Conv2d-17           [-1, 32, 28, 28]             288
             ReLU-18           [-1, 32, 28, 28]               0
      BatchNorm2d-19           [-1, 32, 28, 28]              64
          Dropout-20           [-1, 32, 28, 28]               0
           Conv2d-21           [-1, 32, 28, 28]           1,024
             ReLU-22           [-1, 32, 28, 28]               0
      BatchNorm2d-23           [-1, 32, 28, 28]              64
          Dropout-24           [-1, 32, 28, 28]               0
           Conv2d-25           [-1, 32, 28, 28]           9,216
             ReLU-26           [-1, 32, 28, 28]               0
      BatchNorm2d-27           [-1, 32, 28, 28]              64
          Dropout-28           [-1, 32, 28, 28]               0
           Conv2d-29           [-1, 64, 28, 28]          18,432
             ReLU-30           [-1, 64, 28, 28]               0
      BatchNorm2d-31           [-1, 64, 28, 28]             128
          Dropout-32           [-1, 64, 28, 28]               0
           Conv2d-33           [-1, 32, 13, 13]          18,432
             ReLU-34           [-1, 32, 13, 13]               0
      BatchNorm2d-35           [-1, 32, 13, 13]              64
          Dropout-36           [-1, 32, 13, 13]               0
           Conv2d-37           [-1, 64, 13, 13]          18,432
             ReLU-38           [-1, 64, 13, 13]               0
      BatchNorm2d-39           [-1, 64, 13, 13]             128
          Dropout-40           [-1, 64, 13, 13]               0
           Conv2d-41           [-1, 64, 13, 13]          36,864
             ReLU-42           [-1, 64, 13, 13]               0
      BatchNorm2d-43           [-1, 64, 13, 13]             128
          Dropout-44           [-1, 64, 13, 13]               0
           Conv2d-45           [-1, 64, 11, 11]          36,864
             ReLU-46           [-1, 64, 11, 11]               0
      BatchNorm2d-47           [-1, 64, 11, 11]             128
          Dropout-48           [-1, 64, 11, 11]               0
      AdaptiveAvgPool2d-49       [-1, 64, 1, 1]               0
           Conv2d-50             [-1, 10, 1, 1]             640

----------------------------------------------------------------


Number of parameters used: 169,728<br>
Training accuracy: 76.98<br>
Test accuracy: 85.86<br>

Able to achieve 85%. The model is overfitting.
