# Programming Assignment 1

CS640 Assignment 1

Sahil Gupta

02/15/2019



## Introduction

Here I will present my report for the programming assignment 1. I will include a copy of my jupyter notebook and python file in the github repository where this page is hosted as well. 



## Problem Definition

In this assignment, you need to build a neural network by yourself and implement the backpropagation algorithm to train it. 

Please find the notebook here - https://github.com/sahilBU/cs640/blob/master/Logistic_Regression.ipynb



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

   

```python
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
```



#### Neural Network Implementation

Below are the functions used for implementation:

Forward Propagation: Input is the feature vector. Below are the mathematical equations for it
hidden=X.W1+B1
 
output=hidden.W2+B2
 
Back Propagation: Back propagation here is done with the help of gradient descent. Input here is the activation cache from forward propagation. X is the feature vector and y is the output vector. We backpropagate the derivatives backward. Also here we use the cross entropy loss and softmax function in the output layer. Below are the equations -
outputError=output−oneHotY
 
outputDelta=outputError
 
hiddenError=outputDelta∗(W2.T)
 
hiddenDelta=hiddenError∗∂hidden∂sigmoid
 
Updated Values:

W2=W2−α[(hidden.T).(outputDelta)/d]
 
W1=W1−α[(X.T).hiddenDelta)/d]
 
B2=B2−α∗[∑(outputDelta)/d]
 
B1=B1−α∗[∑(hiddenDelta)/d]
 
Compute_Cost: Computes cost of the data set. This is the cost function which we hope to minimize after k epochs
fit: Computer the train error after completeing k epoch
predict: This is used after the neural network has been trained.


``` python

class NN(object):

    def __init__(self, input_dimension, output_dimension, nodes, alpha=0.1, num_epochs=1000):
        # weights
        self.input_weight = np.random.randn(input_dimension, nodes) / np.sqrt(input_dimension)
        self.hidden_weight = np.random.randn(nodes, output_dimension) / np.sqrt(nodes)

        # bias
        self.input_bias = np.zeros((1, nodes))
        self.output_bias = np.zeros((1, output_dimension))
        self.alpha = alpha
        self.epochs = num_epochs

    def hyperparameters(self, alpha, epochs):
        self.alpha = alpha
        self.epochs = epochs

    def forward_propagation(self, X):
        # dot product of X (input) and first set
        self.hidden = reLU(np.dot(X, self.input_weight) + self.input_bias)
        # dot product of hidden layer and second set
        self.output = softmax(np.dot(self.hidden, self.hidden_weight) + self.output_bias)
        return self.output

    def backward_propagation(self, X, y):
        d = X.shape[0]
        one_hot_y = np.zeros_like(self.output)
        for i in range(y.shape[0]):
            one_hot_y[i, y[i]] = 1

        self.o_error = self.output - one_hot_y
        self.o_delta = self.o_error

        # error: how much hidden layer weights contributed to output error
        self.hid_error = self.o_delta.dot(self.hidden_weight.T)

        # applying derivative of reLu to hidden error
        self.hid_delta = self.hid_error * reLU_derivative(self.hidden)

        w2 = self.hidden.T.dot( self.o_delta) / d
        b2 = np.sum(self.o_delta, axis=0, keepdims=True) / d
        w1 = X.T.dot( self.hid_delta) / d
        b1 = np.sum(self.hid_delta, axis=0, keepdims=True) / d

        # Return updated gradients
        values = { "w1": w1,
                    "b1": b1,
                   "w2": w2,
                "b2": b2}
        return values

    # Updates the weights after calculating gradient in the self propagation step
    def update_weight(self, grads):
        self.input_bias -= self.alpha * grads["b1"]
        self.output_bias -= self.alpha * grads["b2"]
        self.input_weight -= self.alpha * grads["w1"]
        self.hidden_weight -= self.alpha * grads["w2"]
        
   

  
    def compute_cost(self, oz, y):
        m = y.shape[0]
        return np.sum(1 - (oz == y)) / m

    # Fits the neural network using the training dataset
    # Returns the training error for every 10th epoch
    def fit(self, X_train, y_train):
        train_error = [0.5]
        for i in range(self.epochs):
            output = self.forward_propagation(X_train)
            grads = self.backward_propagation(X=X_train, y=y_train)
            self.update_weight(grads)
            if (i % 10 == 0):
                # hot_y = softmax_to_y(output)
                train_error += [self.compute_cost(output, y_train)]
        return train_error

    # Fits the neural network using the training dataset,
    # calculates train as well as test error rate alongside
    def fit_test_train(self, X_train, y_train, X_test, y_test):
        train_error = []
        test_error = []
        for i in range(self.epochs):
            output_train = self.forward_propagation(X_train)
            grads = self.backward_propagation(X=X_train, y=y_train)
            self.update_weight(grads)
            if (i % 10 == 0):
                train_error += [self.compute_cost(output_train, y_train)]
                output_test = self.forward_propagation(X_test)
                test_error += [self.compute_cost(output_test, y_test)]
        error = [None] * 2
        error[0] = train_error
        error[1] = test_error
        return error

    def predict(self, X_test):
        output = self.forward_propagation(X_test)
        return softmax_to_y(output)
```



## Linear 

The plots that the code shows below show that this is a linearly separable case. A lot of epochs are not needed to come up with suitable weights and having higher epochs may lead to the problem of overfitting

```python
def linear():
    X = np.genfromtxt('DATA/data_linearX.csv', delimiter=',')
    y = np.genfromtxt('DATA/data_lineary.csv', delimiter=',').astype(np.int64)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    plt.show()

    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2)
    input_dim = int(X.shape[1])
    output_dim = int(y.max() + 1)
    nodes = 10

    nn = NN(input_dim, nodes, output_dim, alpha=0.05, num_epochs=500)
    train_err = nn.fit(X, y)

    y_pred = nn.predict(X_test)

    err = error_rate(y_pred, y_test)
    print("Error in test set is ", err * 100, "%")

    plt.plot(train_err)
    plt.title("Cross Entropy with respect to Epochs")
    plt.xlabel("Number of Epochs (factor of 10)")
    plt.ylabel("Cross Entropy")
    plt.show()

    # Even though the cross entropy is not minimized,
    # our system is able to distinguish the red points from the blue points easily
    # Plotting decision boundary
    plot_decision_boundary(nn, X_test, y_test)

    # Confusion matrix
    print("Confusion Matrix \n")
    print(confusion_matrix(y_pred, y_test))
    print(classification_report(y_pred, y_test))

    X = np.genfromtxt('DATA/data_linearX.csv', delimiter=',')
    y = np.genfromtxt('DATA/data_lineary.csv', delimiter=',').astype(np.int64)
    test_error = k_fold(X, y, 5, nn)
    print("Average test error is :", test_error)
linear()
```
 





## NonLinear Data



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
