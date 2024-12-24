# LeNet-5 digit-classifier reimplementation with custom classes

## contents
- [features](#features/about)
- [results](#results)

## features/about
this is a reimplementation of LeCunn's 1998 influential paper ["Gradient Based Learning Applied to Document Recognition"](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) which introduced LeNet-5, a low-parameter, CNN-based neural network to classify handwritten digits 0-9.

hyperparameter values, loss choice, optimizer choice, initialization, architecture etc. all follow the paper. my goal was to implement the paper as closely as possible and using as little of the pytorch library as I could (i.e. not using nn.conv2d, nn.avgpool, nn.BCE, nn.optim, etc).

this model was trained on 60,000 samples from MNIST and validated with a seperate set of 10,000 samples. 

## results

### Model Performance
After training for 2 epochs on the MNIST dataset (60,000 training samples), the model achieved:
- Macro F1 Score: 0.964
- Per-digit F1 Scores:
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

#### Highlights
- fast convergence - achieved strong performance in just 2 epochs (state of the art classifiers score 0.99)
- consistent F1 scores above 0.93 across all digits

### Training Visualization
#### Loss Over Time (Epoch 1)
![loss-over-time](https://github.com/user-attachments/assets/c120031b-8aae-4a7b-987b-22330ea578dc)


### Model Predictions
#### Before Training
![Pre-training Predictions](https://github.com/user-attachments/assets/88bb8314-6cfc-4ae2-89bb-8a9e44505977)

#### After Training
![Post-training Predictions](https://github.com/user-attachments/assets/91b42650-4e7f-4e7e-a672-407cdf0683d4)


