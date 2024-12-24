# LeNet-5 digit-classifier reimplementation with custom classes

## contents
- [Features](#features/about)
- [Results](#results)
- [How to Use](#how-to-use)

## features/about
this is a reimplementation of LeCunn's 1998 influential paper "Gradient Based Learning Applied to Document Recognition" which introduced LeNet-5, a low-parameter, CNN-based neural network to classify handwritten digits 0-9.

hyperparameter values, loss choice, optimizer choice, initialization, architecture etc. all follow the paper. my goal was to implement the paper as closely as possible and using as little of the pytorch library as I could (i.e. not using nn.conv2d, nn.avgpool, nn.BCE, nn.optim, etc).

## results
before training:

![image](https://github.com/user-attachments/assets/88bb8314-6cfc-4ae2-89bb-8a9e44505977)


after training:

![image](https://github.com/user-attachments/assets/91b42650-4e7f-4e7e-a672-407cdf0683d4)



## how to use
Your usage instructions here
