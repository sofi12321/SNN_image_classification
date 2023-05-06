# SNN image classification
Comparing Spiking Neural Network with Convolutional Neural Network for Image Classification using snnTorch

## Project Description

Artificial neural networks, including deep neural networks, are widely used for solving machine learning problems. However, these networks are not biologically plausible. Spiking neural networks (SNNs), on the other hand, use different methods of encoding and processing data, making their behavior more similar to brain activity. This motivated us to explore SNNs in more detail and compare their performance with convolutional neural networks (CNNs) in solving image classification problems.

## Repository Contents

- Jupyter notebook with code for implementing and evaluating the SNN and CNN models on the three datasets
- README file with project details and description

## Used datasets

- SOCOFing -  biometric fingerprints and their syn-
thetic alterations 
- EMNIST - handwritten character
digits derived from the NIST Special Database 19 and con-
verted to a 28x28 pixel image format and dataset structure
that directly matches the MNIST dataset 
- Fashion-MNIST - Zalando’s article images

![Datasets](./image.jpg)

## Key methodology components

- Rate-encoding
- LIF neuron model
- Data preprocessing
- Train/validation/test with snnTorch
- CrossEntropy loss
- Adam optimizer
- SNN and CNN comparison


## Model architecture

- SNN

| Layer | Parameters |
|-------|-------|
| Linear | input = 9409, output = 1024  |
| Leaky  | beta = 0.95 |
| Linear | input = 1024, output = 10    |
| Leaky  | beta = 0.95 |


- CNN

| Layer | Parameters |
|-------|-------|
| Conv2d | input channels = 1, output channels = 16, kernel size = (3, 3), stride = (1, 1), activation = ReLU  |
| MaxPool2d  | kernel size = 2, stride = 2, padding = 0 |
| Conv2d | input channels = 16, output channels = 32, kernel size = (3, 3), stride = (1, 1), activation = ReLU    |
| MaxPool2d  | kernel size = 2, stride = 2, padding = 0 |
| Linear | input = 15488, output = 1024, activation = ReLU  |
| Linear | input = 1024, output = 10, activation = log softmax  |

## Visualization of output neuron

![Datasets](./vizualization.jpg)


## Results

| Dataset       | Model | Accuracy | Precision | Recall | F1-score |
|-------------  |-------|----------|-----------|--------|----------|
| SOCOFing      | SNN   | 0.98     | 0.98      | 0.98   | 0.98     |
| SOCOFing      | CNN   | 0.83     | 0.84      | 0.82   | 0.83     |
| EMNIST        | SNN   | 0.99     | 0.99      | 0.99   | 0.99     |
| EMNIST        | CNN   | 0.99     | 0.99      | 0.99   | 0.99     |
| Fashion-MNIST | SNN   | 0.86     | 0.86      | 0.87   | 0.86     |
| Fashion-MNIST | CNN   | 0.86     | 0.86      | 0.86   | 0.86     |

The training time for 1 epoch was more than 1.5 times longer for the SNN model.

## Requirements

- Python 3.x
- snnTorch
- PyTorch
- NumPy
- Matplotlib
- Pandas
- Kaggle API

## Usage

- Clone the repository to your local machine.
- Install the required libraries.
- Obtain a Kaggle API token to download the SOCOFing dataset.
- Run the Jupyter notebook and follow the instructions to train and evaluate the SNN and CNN models on the three datasets.

## Authors

- [Danila Shulepin](https://github.com/D4ni1a)
- [Sofi Zaitseva](https://github.com/sofi12321)
- [Arsen Mutalapov](https://github.com/system205)

## License

This project is licensed under the MIT License - see the LICENSE file for details.