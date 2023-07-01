# DeepERA
Learn: Deep learning, Pytorch, Computer Vision, NLP

## S8
**Objective: Train a model with less than 200K parameters for as many as epochs and get accuacy more than 85% and get intuition of different convolutions.**

Optimizer used: SGD<br>
Lr Scheduler: Step LR<br>
Dataset: CIFAR10<br>
Batch Size: 128

#### 1. Model

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1           [-1, 16, 32, 32]             432
              ReLU-2           [-1, 16, 32, 32]               0
       BatchNorm2d-3           [-1, 16, 32, 32]              32
           Dropout-4           [-1, 16, 32, 32]               0
            Conv2d-5           [-1, 16, 32, 32]           2,304
              ReLU-6           [-1, 16, 32, 32]               0
       BatchNorm2d-7           [-1, 16, 32, 32]              32
           Dropout-8           [-1, 16, 32, 32]               0
            Conv2d-9           [-1, 32, 28, 28]           4,608
             ReLU-10           [-1, 32, 28, 28]               0
      BatchNorm2d-11           [-1, 32, 28, 28]              64
          Dropout-12           [-1, 32, 28, 28]               0
           Conv2d-13           [-1, 32, 28, 28]           9,216
             ReLU-14           [-1, 32, 28, 28]               0
      BatchNorm2d-15           [-1, 32, 28, 28]              64
          Dropout-16           [-1, 32, 28, 28]               0
           Conv2d-17           [-1, 64, 28, 28]          18,432
             ReLU-18           [-1, 64, 28, 28]               0
      BatchNorm2d-19           [-1, 64, 28, 28]             128
          Dropout-20           [-1, 64, 28, 28]               0
           Conv2d-21           [-1, 32, 28, 28]           2,048
             ReLU-22           [-1, 32, 28, 28]               0
      BatchNorm2d-23           [-1, 32, 28, 28]              64
          Dropout-24           [-1, 32, 28, 28]               0
           Conv2d-25           [-1, 32, 28, 28]           9,216
             ReLU-26           [-1, 32, 28, 28]               0
      BatchNorm2d-27           [-1, 32, 28, 28]              64
          Dropout-28           [-1, 32, 28, 28]               0
           Conv2d-29           [-1, 32, 28, 28]           9,216
             ReLU-30           [-1, 32, 28, 28]               0
      BatchNorm2d-31           [-1, 32, 28, 28]              64
          Dropout-32           [-1, 32, 28, 28]               0
           Conv2d-33           [-1, 64, 24, 24]          18,432
             ReLU-34           [-1, 64, 24, 24]               0
      BatchNorm2d-35           [-1, 64, 24, 24]             128
          Dropout-36           [-1, 64, 24, 24]               0
           Conv2d-37           [-1, 64, 24, 24]          36,864
             ReLU-38           [-1, 64, 24, 24]               0
      BatchNorm2d-39           [-1, 64, 24, 24]             128
          Dropout-40           [-1, 64, 24, 24]               0
           Conv2d-41          [-1, 128, 24, 24]          73,728
             ReLU-42          [-1, 128, 24, 24]               0
      BatchNorm2d-43          [-1, 128, 24, 24]             256
          Dropout-44          [-1, 128, 24, 24]               0
           Conv2d-45           [-1, 64, 24, 24]           8,192
             ReLU-46           [-1, 64, 24, 24]               0
      BatchNorm2d-47           [-1, 64, 24, 24]             128
          Dropout-48           [-1, 64, 24, 24]               0
      AdaptiveAvgPool2d-49       [-1, 64, 1, 1]               0
           Conv2d-50             [-1, 10, 1, 1]             640

--------------------------------------------------------------------

Number of parameters used: 194,480<br>
Training accuracy: 82.48<br>
Test accuracy: 80.45<br>

Able to achieve 80%. The model is still overfitting.
