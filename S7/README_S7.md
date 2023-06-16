# DeepERA
Learn: Deep learning, Pytorch, Computer Vision, NLP

## S7

### MNIST Training:

Fixed hyperparameters:  batch Size: 128<br>
                        Optimizer Used: SGD<br>

#### ModelName: Model Init
**Target:**   
To Initialize all the setup like dataloaders, batchsize etc. and get the flow of the complete working code.<br>
**Results:**  
1. Training accuracy: 99.64<br>
2. Test accuracy: 99.36<br>
3. Total number of parameters: 593,200<br>
**Analysis:** 
1. Number of parameters are very high.<br>
2. Receptive field is 16.<br>
3. There is small underfitting of data.<br>
4. Accuracy is good.<br>

Please download the notebook here: ![ModelInit](./S7_Model_Init.ipynb)
<br>
**Model Architecture:**<br> 

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1           [-1, 32, 26, 26]             320
            Conv2d-2           [-1, 64, 24, 24]          18,496
            Conv2d-3          [-1, 128, 10, 10]          73,856
            Conv2d-4            [-1, 256, 8, 8]         295,168
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510     
----------------------------------------------------------------


#### ModelName: Net1
Target:   To reduce the number of parameters and get the good accuracy. Use of batch normalization layers, dropouts layers.<br>
**Results:** 
1. Training accuracy: 99.46<br>
2. Test accuracy: 99.52<br>
3. Total number of parameters: 19,470<br>
**Analysis:** 
1. Number of parameters are reduced.<br>
2. Receptive field is 11.<br>
3. Training accuracy is low as compare to test accuracy.<br>

Please download the notebook here: ![Model_1](./S7_Model_1.ipynb)
<br>

**Model Architecture:**

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
            Conv2d-1           [-1, 16, 26, 26]             160
              ReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
            Conv2d-4           [-1, 16, 24, 24]           2,320
              ReLU-5           [-1, 16, 24, 24]               0
       BatchNorm2d-6           [-1, 16, 24, 24]              32
         MaxPool2d-7           [-1, 16, 12, 12]               0
              ReLU-8           [-1, 16, 12, 12]               0
           Dropout-9           [-1, 16, 12, 12]               0
           Conv2d-10           [-1, 32, 10, 10]           4,640
             ReLU-11           [-1, 32, 10, 10]               0
      BatchNorm2d-15             [-1, 32, 8, 8]              64
        MaxPool2d-16             [-1, 32, 4, 4]               0
             ReLU-17             [-1, 32, 4, 4]               0
          Dropout-18             [-1, 32, 4, 4]               0
           Conv2d-19             [-1, 10, 2, 2]           2,890
      BatchNorm2d-20             [-1, 10, 2, 2]              20
          Dropout-21             [-1, 10, 2, 2]               0
        AvgPool2d-22             [-1, 10, 1, 1]               0
----------------------------------------------------------------



#### ModelName: Net2
**Target:**  
To reduce the number of parameters and get the accuracy upto 99.4 within 20 epochs.<br>
**Results:**  
1. Training accuracy: 99<br>
2. Test accuracy: 99.43<br>
3. Total number of parameters: 13,808<br>
**Analysis:** 
1. Number of parameters are reduced.
2. Receptive field is 16.
3. Pretty good model in terms of test accuracy.

Please download the notebook here: ![Model_2](./S7_Model_2.ipynb)
<br>

**Model Architecture:**

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------

            Conv2d-1           [-1, 16, 26, 26]             144
              ReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
           Dropout-4           [-1, 16, 26, 26]               0
            Conv2d-5           [-1, 32, 24, 24]           4,608
              ReLU-6           [-1, 32, 24, 24]               0
       BatchNorm2d-7           [-1, 32, 24, 24]              64
           Dropout-8           [-1, 32, 24, 24]               0
            Conv2d-9           [-1, 10, 24, 24]             320
        MaxPool2d-10           [-1, 10, 12, 12]               0
           Conv2d-11           [-1, 16, 10, 10]           1,440
             ReLU-12           [-1, 16, 10, 10]               0
      BatchNorm2d-13           [-1, 16, 10, 10]              32
          Dropout-14           [-1, 16, 10, 10]               0
           Conv2d-15             [-1, 16, 8, 8]           2,304
             ReLU-16             [-1, 16, 8, 8]               0
      BatchNorm2d-17             [-1, 16, 8, 8]              32
          Dropout-18             [-1, 16, 8, 8]               0
           Conv2d-19             [-1, 16, 6, 6]           2,304
             ReLU-20             [-1, 16, 6, 6]               0
      BatchNorm2d-21             [-1, 16, 6, 6]              32
          Dropout-22             [-1, 16, 6, 6]               0
           Conv2d-23             [-1, 16, 6, 6]           2,304
             ReLU-24             [-1, 16, 6, 6]               0
      BatchNorm2d-25             [-1, 16, 6, 6]              32
          Dropout-26             [-1, 16, 6, 6]               0
        AvgPool2d-27             [-1, 16, 1, 1]               0
           Conv2d-28             [-1, 10, 1, 1]             160
----------------------------------------------------------------


#### ModelName: Final Model
*Experiments1:* <br>
**Target:**   
To fix the skeleton of the architecture and furether reduce the number of parameters below the 8k and get the good accuracy.<br>
**Results:**  
1. Training accuracy: 98.78<br>
2. Test accuracy: 99.38(Only one time)<br>
3. Total number of parameters: 7,958<br>
**Analysis:** 
1. Number of parameters are reduced and architecture is fixed.
2. Receptive field is 12.
3. The model accuracy is good and training accuracy is less. Need for some improvements there.
4. Test accuracy is doesn't meet our objective.

Please download the notebook here: ![ModelExp1](./S7_FinalModelExps.ipynb)
<br>

**Model Architecture:**

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1           [-1, 10, 26, 26]              90
              ReLU-2           [-1, 10, 26, 26]               0
       BatchNorm2d-3           [-1, 10, 26, 26]              20
           Dropout-4           [-1, 10, 26, 26]               0
            Conv2d-5           [-1, 20, 24, 24]           1,800
              ReLU-6           [-1, 20, 24, 24]               0
       BatchNorm2d-7           [-1, 20, 24, 24]              40
           Dropout-8           [-1, 20, 24, 24]               0
            Conv2d-9           [-1, 10, 24, 24]             200
        MaxPool2d-10           [-1, 10, 12, 12]               0
           Conv2d-11           [-1, 10, 10, 10]             900
             ReLU-12           [-1, 10, 10, 10]               0
      BatchNorm2d-13           [-1, 10, 10, 10]              20
          Dropout-14           [-1, 10, 10, 10]               0
           Conv2d-15             [-1, 10, 8, 8]             900
             ReLU-16             [-1, 10, 8, 8]               0
      BatchNorm2d-17             [-1, 10, 8, 8]              20
          Dropout-18             [-1, 10, 8, 8]               0
           Conv2d-19             [-1, 16, 6, 6]           1,440
             ReLU-20             [-1, 16, 6, 6]               0
      BatchNorm2d-21             [-1, 16, 6, 6]              32
          Dropout-22             [-1, 16, 6, 6]               0
           Conv2d-23             [-1, 16, 6, 6]           2,304
             ReLU-24             [-1, 16, 6, 6]               0
      BatchNorm2d-25             [-1, 16, 6, 6]              32
          Dropout-26             [-1, 16, 6, 6]               0
        AvgPool2d-27             [-1, 16, 1, 1]               0
           Conv2d-28             [-1, 10, 1, 1]             160
----------------------------------------------------------------


*Experiments2:* <br>
**Target:**   
Change the learining rate to high to get better accuracy in less epochs.<br>

**Results:** 
1. Training accuracy: 98.83<br>
2. Test accuracy: 99.28<br>
3. Total number of parameters: 7,958<br>

**Analysis:**
1. Still there is a gap between training and test accuracy.<br>
2. Converging rate is faster as compare to previous model.<br>
3. Test accuracy became plattue.<br>

Please download the notebook here: ![ModelExp2](./S7_FinalModelExps.ipynb)
<br>

*Experiments3:* <br>
**Target:**   
Add lr schedular to get whenever the validation or test accuracy becomes plattue.<br>

**Results:**  
1. Training accuracy: 98.93<br>
2. Test accuracy: 99.46<br>
3. Total number of parameters: 7,958<br>

**Analysis:**
1.  Met the test accuracy and it is consistantly more then 99.40%.
2.  Still training accracy is less due to making training data hard.
Please download the notebook here: ![FinalModel](./S7_FinalModel.ipynb)
<br>