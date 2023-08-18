# DeepERA
Learn: Deep learning, Pytorch, Computer Vision, NLP, Pytorch Lightening

## S12
**Objective: Train a model with 24 epochs using one cycle policy**

Optimizer used: ADAM<br>
Lr Scheduler: ONeCycle LR<br>
Dataset: CIFAR10<br>
Batch Size: 512

#### 1. Model

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1           [-1, 64, 32, 32]           1,728
              ReLU-2           [-1, 64, 32, 32]               0
       BatchNorm2d-3           [-1, 64, 32, 32]             128
           Dropout-4           [-1, 64, 32, 32]               0
            Conv2d-5          [-1, 128, 32, 32]          73,728
         MaxPool2d-6          [-1, 128, 16, 16]               0
       BatchNorm2d-7          [-1, 128, 16, 16]             256
              ReLU-8          [-1, 128, 16, 16]               0
           Dropout-9          [-1, 128, 16, 16]               0
           Conv2d-10          [-1, 128, 16, 16]         147,456
      BatchNorm2d-11          [-1, 128, 16, 16]             256
             ReLU-12          [-1, 128, 16, 16]               0
           Conv2d-13          [-1, 128, 16, 16]         147,456
      BatchNorm2d-14          [-1, 128, 16, 16]             256
             ReLU-15          [-1, 128, 16, 16]               0
    ResidualBlock-16          [-1, 128, 16, 16]               0
           Conv2d-17          [-1, 256, 16, 16]         294,912
        MaxPool2d-18            [-1, 256, 8, 8]               0
      BatchNorm2d-19            [-1, 256, 8, 8]             512
             ReLU-20            [-1, 256, 8, 8]               0
          Dropout-21            [-1, 256, 8, 8]               0
           Conv2d-22            [-1, 512, 8, 8]       1,179,648
        MaxPool2d-23            [-1, 512, 4, 4]               0
      BatchNorm2d-24            [-1, 512, 4, 4]           1,024
             ReLU-25            [-1, 512, 4, 4]               0
          Dropout-26            [-1, 512, 4, 4]               0
           Conv2d-27            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-28            [-1, 512, 4, 4]           1,024
             ReLU-29            [-1, 512, 4, 4]               0
           Conv2d-30            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-31            [-1, 512, 4, 4]           1,024
             ReLU-32            [-1, 512, 4, 4]               0
    ResidualBlock-33            [-1, 512, 4, 4]               0
        MaxPool2d-34            [-1, 512, 1, 1]               0
           Linear-35                   [-1, 10]           5,130
           
----------------------------------------------------------------


Number of parameters used: 6,573,130<br>

Able to achieve 87%.
Please download the notebook here: ![ Notebook ](./S12.ipynb)

If you want to try your own images, please click here: ![ HuggingFace ](https://huggingface.co/spaces/Vasudevakrishna/eraS12)
