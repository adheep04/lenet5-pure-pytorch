# LeNet-5 digit-classifier reimplementation with custom classes

## contents
- [features](#features/about)
- [results](#results)

## features/about
this is a reimplementation of LeCunn's 1998 influential paper "Gradient Based Learning Applied to Document Recognition" which introduced LeNet-5, a low-parameter, CNN-based neural network to classify handwritten digits 0-9.

hyperparameter values, loss choice, optimizer choice, initialization, architecture etc. all follow the paper. my goal was to implement the paper as closely as possible and using as little of the pytorch library as I could (i.e. not using nn.conv2d, nn.avgpool, nn.BCE, nn.optim, etc).

this model was trained on 60,000 samples from MNIST and validated with a seperate set of 10,000 samples. 

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

### Key Highlights
- Fast Convergence: Achieved strong performance in just 2 epochs
- Balanced Performance: Consistent F1 scores above 0.93 across all digits
- Efficient Training: Results obtained using [your training specifications - batch size, optimizer, etc.]

## results

The model's final accuracy tested on the validation set after 1 epoch of training was 97%. 

epoch 1 training loss over time:

![image](https://github.com/user-attachments/assets/a7df16c1-fab8-4168-a692-64190a0d7b47)

following validation samples are from

before training:

![image](https://github.com/user-attachments/assets/88bb8314-6cfc-4ae2-89bb-8a9e44505977)


after training:

![image](https://github.com/user-attachments/assets/91b42650-4e7f-4e7e-a672-407cdf0683d4)


