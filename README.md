# Programming Assignment 1

CS640 Assignment 1

Sahil Gupta

02/15/2019



## Introduction

Here I will present my report for the programming assignment 1. I will include a copy of my jupyter notebook and python file in the github repository where this page is hosted as well. 



## Problem Definition

In this assignment, you need to build a neural network by yourself and implement the backpropagation algorithm to train it. 



## Implementation

#### Activation Functions

Below are my choices for the activation functions and the reason for choosing them: 

1. sigmoid function as activation function because its the most common function which can give the weighted sum between 0 and 1with some bias in it for it to be inactive and only fire when the threshold is reached.
2. softmax function because of its capability of converting output as probabilities.



#### Neural Network Specification: 

We will implement a simple 2 layer neural network with the following spec:

1. *Layer 1:*
   1. Input: Number of features
   2. Output: Number of Nodes in the hidden layer
   3. Activation Function:  Sigmoid Function
2. *Layer 2:* 
   1. Input: Number of Nodes in the hidden layer
   2. Output: Number of dimensions in the output 
   3. Activation Function:  None because the output here is passed through softmax function

#### Tasks: 

Below are the tasks performed: 

1. Seperating non-linear data
2. Seperating linear data and 
3. Recognizing the digits



#### Limitations

- This neural network will introduce some non-linearity in the analysis but it will not be scalable. 
- The network will only be able to understand the very simple non-linear functions. As we are using only 1 hidden layer, the network can introuce only 1 layer of non-linearity in the network.
-  To calculate more complex function, we can select a network with more number of nodes in hidden layer, but a better way to go about it to introduce more layers in the network.



#### Helper Functions

1. sigmoid - This function gives the sigmoid of the input array 

2. sigmoid_derivative - This functions returns the derivative of the sigmoid function

3. softmax - This function gives the softmax of the input array

4. softmax_hoty - This function gives input softmax vector and returns the most probable class

5. k_fold - We will be using K-Fold from the Sklearn library and this function will take the input matrix i.e., provided data X, Y along with neural_network class object and returns sum of the test error. 

   

python
def sigmoid(t):
    return 1.0/(1.0+np.exp(-1.0*t))

def sigmoid_derivative(s):
    return s * (1.0 - s)

def softmax(z):
    exp_z = np.exp(z)
    softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return softmax_scores

def softmax_hoty(softmax_scores):
    return np.argmax(softmax_scores, axis=1)

def k_fold(X,y,k,nn):
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    # returns the number of splitting iterations in the cross-validator
    kf.get_n_splits(X) 
    test_error = []
    train_error = []
    for train_index, test_index in kf.split(X):
         #print('TRAIN:', train_index, 'TEST:', test_index)
         X_train, X_test = X[train_index], X[test_index]
         y_train, y_test = y[train_index], y[test_index]
         train_error = (nn.fit(X_train, y_train, num_epochs = 500))
         y_pred = nn.predict(X_test)
         test_error.append(nn.compute_cost(y_pred, y_test, X.shape[0]))
    #plt.plot(train_error)
    plt.plot(train_error)
    return (np.sum(test_error)/(k+1))




#### Neural Network Implementation



## Linear Data



## Non-Linear Data



## Learning Rate



## Regularlization 

Any type of constrained optimization is regularization procedure. We could add a penality in the performance function which would indicate the complexity of a function.



- L1 Regularization
- L2 Regularization
- Dropout Regularization



#### L2 Regularization Implementation:

We are adding a term *lambdaweight* to the weight on every term. This is used because for L2 regularization, we add lambda/2* ( weight ) ^ 2 to the performace function. Derivative of this function is lambda * weight.



## Digit Recognition

We have taken a neural network with more than 10 nodes in the layer. If we tke less than 10, the network will have to share computations, which may lead to poor performance.



## Experiments and Exploration

We will understand hyperparameters in our network by analyzing the test error and the train error in the network. In this case, the hyperparameters are as follows: 

1. Number of nodes in the hidden layer
2. Learning Rate and 
3. No. of Epochs

## Results

List your experimental results. Provide examples of input images and output images. If relevant, you may provide images showing any intermediate steps. If your work involves videos, do not submit the videos but only links to them.

## Discussion



The accuracy of this model is 94% and this accuracy rate common among neural networks with one hidden layer. 

## Credits and Bibliography
