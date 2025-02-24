[![pwc](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gradient-based-learning-applied-to-document/handwritten-digit-recognition-on-digits-1)](https://paperswithcode.com/sota/handwritten-digit-recognition-on-digits-1?p=gradient-based-learning-applied-to-document)

# LeNet-5 implementation

this project implements LeCunn's 1998 paper ["gradient based learning applied to document recognition"](http://vision.stanford.edu/cs598_spring07/papers/lecun98.pdf). the model performed a 0.96 on the f1 macro metric with minimal training epochs. code and models are available.

also on https://paperswithcode.com/paper/gradient-based-learning-applied-to-document

## features:
- custom implementations of:
  - a convolution layer class which supports sparse connections between channels and shared weights making it more flexible than pytorch's nn.conv2d!
  - average pooling, loss function, and optimizer using pytorch tensors operations 
    - maximum a posteriori loss and sdlm optimizer
- follows original paper's specifications strictly for architecture, hyperparameters, initialization, and even the stylized 10x8x12 bitmap the 0-9 digits used as weights in the rbf layer
  [https://ibb.co/d6ktzc0](https://ibb.co/d6KTZC0)
- trained with 60,000 samples and validated with 10,000 on mnist dataset

## performance after 2 epochs of training:
- macro f1 score: 0.964 
- per-digit f1 scores:
  | Digit | F1 Score |
  |-------|----------|
  | 0     | 0.980    |
  | 1     | 0.990    |
  | 2     | 0.960    |
  | 3     | 0.955    |
  | 4     | 0.979    |
  | 5     | 0.979    |
  | 6     | 0.955    |
  | 7     | 0.955    |
  | 8     | 0.938    |
  | 9     | 0.950    |
  
- all digit-f1 scores above 0.93!

training loss over time:
![loss-over-time](https://github.com/user-attachments/assets/c120031b-8aae-4a7b-987b-22330ea578dc)

before/after predictions:  
![pre-training predictions](https://github.com/user-attachments/assets/88bb8314-6cfc-4ae2-89bb-8a9e44505977)
![post-training predictions](https://github.com/user-attachments/assets/91b42650-4e7f-4e7e-a672-407cdf0683d4)

implementation details:
- input: 32x32 grayscale images
- architecture: 
 - conv5x5 (6 maps) -> avgpool2x2 
 - conv5x5 (16 maps) -> avgpool2x2
 - fc (120) -> fc (84) -> rbf output
