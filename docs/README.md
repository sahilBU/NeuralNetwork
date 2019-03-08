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

![L1](https://user-images.githubusercontent.com/43014978/54051654-548bac00-41b0-11e9-867d-bcfd067dfc4e.png)

![L2](https://user-images.githubusercontent.com/43014978/54051733-8270f080-41b0-11e9-8ed3-cc17312e9dc7.png)

![L3](https://user-images.githubusercontent.com/43014978/54051734-843ab400-41b0-11e9-91e0-d016089968fc.png)


Confusion Matrix 

[[204   0]
 [  1 195]]
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       204
           1       1.00      0.99      1.00       196

   micro avg       1.00      1.00      1.00       400
   macro avg       1.00      1.00      1.00       400
weighted avg       1.00      1.00      1.00       400

Average test error is : 0.0004166666666666667

 





## NonLinear Data

```python

def non_linear():
    X = np.genfromtxt('DATA/data_nonlinearX.csv', delimiter=',')
    y = np.genfromtxt('DATA/data_nonlineary.csv', delimiter=',').astype(np.int64)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    plt.show()
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2)
    input_dim = int(X.shape[1])
    output_dim = int(y.max() + 1)
    nodes = 30

    nn = NN(input_dim, nodes, output_dim, alpha=0.5, num_epochs=2500)
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

    # Cross Validation score for the linear dataset
    X = np.genfromtxt('DATA/data_nonlinearX.csv', delimiter=',')
    y = np.genfromtxt('DATA/data_nonlineary.csv', delimiter=',').astype(np.int64)
    test_error = k_fold(X, y, 5, nn)
    print("Average test error is :", test_error)

    # Script for checking the test and train error
    X = np.genfromtxt('DATA/data_nonlinearX.csv', delimiter=',')
    y = np.genfromtxt('DATA/data_nonlineary.csv', delimiter=',').astype(np.int64)
    input_dim = int(X.shape[1])
    output_dim = int(y.max() + 1)

    nodes = 30

    nn = NN(input_dim, nodes, output_dim, alpha=0.05, num_epochs=500)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    err = nn.fit_test_train(X_train, y_train, X_test, y_test)

    train_err = err[0]
    test_err = err[1]

    plt.plot(train_err)
    plt.plot(test_err)
    plt.legend(('Train', 'Test'))
    plt.show()
non_linear()

```
![NL1](https://user-images.githubusercontent.com/43014978/54052088-776a9000-41b1-11e9-925d-aa43c0b4d449.png)

Error in test set is  2.0 %

![NL2](https://user-images.githubusercontent.com/43014978/54052094-7c2f4400-41b1-11e9-82c3-8e6c86af78ab.png)

![NL3](https://user-images.githubusercontent.com/43014978/54052101-818c8e80-41b1-11e9-93e0-9ba74e43accf.png)

Confusion Matrix 

[[200   4]
 [  4 192]]
              precision    recall  f1-score   support

           0       0.98      0.98      0.98       204
           1       0.98      0.98      0.98       196

   micro avg       0.98      0.98      0.98       400
   macro avg       0.98      0.98      0.98       400
weighted avg       0.98      0.98      0.98       400

Average test error is : 0.012499999999999999

## Learning Rate

What effect does the learning rate have on how your neural network is trained? Illustrate your answer by training your model using different learning rates. Use a script to generate output statistics and visualize them. (5pts)

Learning rate controls the weights of our network with respect the loss gradient. The speed at which we travel the slope is determined by the learning rate. If we choose the lower value of learning rate, we travel the slope slowly.

Relationship between learning rate and weights of our network - new_weight = existing_weight — learning_rate * gradient

We need to choose the optimum learning rate for our system to perform best.

For learning rate .001, we get error - 0.899888765294772 For learning rate .05, we get error - 0.0967741935483871 For learning rate .01, we get error - 0.39154616240266965 For learning rate .5, we get error - 0.07119021134593993 For learning rate 1, we get error - 0.12791991101223582 For learning rate 5, we get error - 0.8987764182424917 For learning rate 10, we get error - 0.899888765294772 For learning rate 15, we get error - 0.8987764182424917

We start with lower value our error rate is high, as we increase our learning rate to 1 our error rate gets to minimum. As we keep increasing the value of our rate, the error rate increases. So we have to find the optimum error rate.


   ```python
lrs=[.001,.05,.01,.5,1,5,10,15]
errors=[]
for i in lrs:
    print('Learning Rate: -'+str(i))
    X = np.genfromtxt('DATA/Digit_X_train.csv', delimiter=',')
    y = np.genfromtxt('DATA/Digit_y_train.csv', delimiter=',').astype(np.int64)
    m = X.shape[0] 
    input_dim = int(X.shape[1])
    output_dim = int(y.max() + 1)
    nodes = 20
    nn=NN(input_dim,  nodes , output_dim,alpha=i)
    train_err = nn.fit(X,y)
    X_test = np.genfromtxt('DATA/Digit_X_test.csv', delimiter=',')
    y_test = np.genfromtxt('DATA/Digit_y_test.csv', delimiter=',').astype(np.int64)

    y_pred = nn.predict(X_test)

    error_rate1 = nn.compute_cost(y_pred,y_test)
    errors.append(error_rate1)
    print("Error Rate:- "+str(error_rate1))
    plt.plot(train_err)
    plt.title('Train Error')
    plt.xlabel('No. of epochs')
    plt.ylabel("Error")
    plt.show()
 
 ```
    
    ![LR1](https://user-images.githubusercontent.com/43014978/54052316-18f1e180-41b2-11e9-838f-1a2a934ca502.png)

Learning Rate: -0.05
Error Rate:- 0.135706340378198

![LR2](https://user-images.githubusercontent.com/43014978/54052322-1becd200-41b2-11e9-9c75-965d4951c1f4.png)

Learning Rate: -0.01
Error Rate:- 0.489432703003337


![LR3](https://user-images.githubusercontent.com/43014978/54052328-1e4f2c00-41b2-11e9-99ef-8ad11e446ae7.png)

Learning Rate: -0.5
Error Rate:- 0.0778642936596218


![LR4](https://user-images.githubusercontent.com/43014978/54052330-2018ef80-41b2-11e9-8add-7bee9f7f5444.png)

Learning Rate: -1
Error Rate:- 0.08120133481646273

![LR6](https://user-images.githubusercontent.com/43014978/54052341-260ed080-41b2-11e9-8326-5c06f80b8810.png)

Learning Rate: -10
Error Rate:- 0.8987764182424917

![LR7](https://user-images.githubusercontent.com/43014978/54052348-28712a80-41b2-11e9-915f-ea44aed3e802.png)


Learning Rate: -15
Error Rate:- 0.8987764182424917
    




![LR](https://user-images.githubusercontent.com/43014978/54052470-7f76ff80-41b2-11e9-842f-71930981fc15.png)



## Regularlization 





Any type of constrained optimization is regularization procedure. We could add a penality in the performance function which would indicate the complexity of a function.



- L1 Regularization
- L2 Regularization
- Dropout Regularization



#### L2 Regularization Implementation:

We are adding a term *lambdaweight* to the weight on every term. This is used because for L2 regularization, we add lambda/2* ( weight ) ^ 2 to the performace function. Derivative of this function is lambda * weight.

```python
class regularizationL2(NN):

    def __init__(self, input_dimension, nodes, output_dimension, alpha=0.5, num_epochs=1000, reg_para=0.001):
        self.reg = reg_para
        super().__init__(input_dimension, nodes, output_dimension, alpha, num_epochs)

    def hyperparameter(self, alpha, num_epochs, reg_para):
        self.alpha = alpha
        self.num_epochs = num_epochs
        self.reg = reg_para

    def update_weight(self, grads):
        # Adding derivative of regularization term
        self.input_bias -= self.reg * self.input_bias
        self.output_bias -= self.reg * self.output_bias
        self.input_weight -= self.reg * self.input_weight
        self.hidden_weight -= self.reg * self.hidden_weight

        self.input_bias -= self.alpha * grads["b1"]
        self.output_bias -= self.alpha * grads["b2"]
        self.input_weight -= self.alpha * grads["w1"]
        self.hidden_weight -= self.alpha * grads["w2"]
        
def l2Regularization():
    X = np.genfromtxt('DATA/data_nonlinearX.csv', delimiter=',')
    y = np.genfromtxt('DATA/data_nonlineary.csv', delimiter=',').astype(np.int64)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    plt.show()
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2)
    input_dim = int(X.shape[1])
    output_dim = int(y.max() + 1)
    nodes = 30
    nn = regularizationL2(input_dim, nodes, output_dim, alpha=0.1, num_epochs=2500)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    err = nn.fit_test_train(X_train, y_train, X_test, y_test)

    train_err = err[0]
    test_err = err[1]

    
l2Regularization()

```
![L2](https://user-images.githubusercontent.com/43014978/54052720-25c30500-41b3-11e9-8388-51fd78d189db.png)



## Digit Recognition

We have taken a neural network with more than 10 nodes in the layer. If we tke less than 10, the network will have to share computations, which may lead to poor performance.

```python

def digitTraining():
    X = np.genfromtxt('DATA/Digit_X_train.csv', delimiter=',')
    y = np.genfromtxt('DATA/Digit_y_train.csv', delimiter=',').astype(np.int64)
    input_dim = int(X.shape[1])
    output_dim = int(y.max() + 1)
    nodes = 30

    nn = NN(input_dim, nodes, output_dim, alpha=0.05, num_epochs=2500)
    train_err = nn.fit(X, y)
    train_err.pop(0)

    X_test = np.genfromtxt('DATA/Digit_X_test.csv', delimiter=',')
    y_test = np.genfromtxt('DATA/Digit_y_test.csv', delimiter=',').astype(np.int64)

    y_pred = nn.predict(X_test)
    
    print(X_test.shape)
    print(y_test.shape)

    err = error_rate(y_pred, y_test)
    print(err)

    

    nn = regularizationL2(input_dim, nodes, output_dim, alpha=0.1, num_epochs=2500)
    err = nn.fit_test_train(X, y, X_test, y_test)

    train_err = err[0]
    test_err = err[1]
    
    print(confusion_matrix(y_pred,y_test))
digitTraining()

```
(899, 64)
(899,)
0.067853170189099
[[84  1  0  0  0  0  0  0  0  0]
 [ 0 77  0  1  2  0  1  0  5  0]
 [ 0  0 83  2  0  0  0  0  1  0]
 [ 0  0  3 82  0  0  0  1  1  1]
 [ 1  1  0  0 87  0  0  1  0  0]
 [ 1  0  0  4  0 88  0  0  4  2]
 [ 2  1  0  0  1  1 89  0  0  0]
 [ 0  0  0  0  0  0  0 83  0  0]
 [ 0  0  0  2  0  0  1  0 77  1]
 [ 0 11  0  0  2  2  0  4  0 88]]



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
