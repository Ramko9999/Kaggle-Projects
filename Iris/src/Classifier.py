import os
import numpy as np
import scipy.optimize as optimize
from math import e, pow

def getData(features, labels):
    os.chdir("/Train")
    X = np.genfromtxt(features)
    X = X[:, 1:]
    X = X[:, : np.shape(X)[1] - 1]
    #appending the bias units
    bias = np.ones((np.shape(X)[0], 1))
    X = np.hstack((bias, X))
    Y = np.genfromtxt(labels)
    return X, Y


#3 layered neural network
class NeuralNetwork():
    # X_train and Y_train should be the training data, nothing else
    '''
    I haven't created the weights using any kind of initialization function
    '''
    def __init__(self, X_train, Y_train, hidden_size, reg):
        self.X = X_train #120 x 5
        self.reg = reg
        self.Y = Y_train #120 x 3
        self.input_size = np.shape(X_train)[0] #120
        self.hidden_size = hidden_size #3
        self.output_size = np.shape(Y_train)[1] #3
        length_of_weights = np.shape(X_train)[1]*(hidden_size) + (hidden_size+1)*self.output_size
        self.weights = np.random.randn(length_of_weights, 1)

    #unregularized cost function
    def getCost(self, weights):
        a3 = self.predict(weights)
        m = np.shape(self.X)[0]

        return np.sum(np.sum(self.crossEntropy(a3)))/m


    #feed forward part of the neural network
    def predict(self, weights):
        theta1, theta2 = self.unroll(weights)
        z2 = np.dot(self.X, theta1)
        a2 = self.sigmoid(z2)
        #appending the bias units to our new inputs
        bias_units = np.ones((np.shape(a2)[0], 1))
        a2 = np.hstack((bias_units, a2))
        z3 = np.dot(a2, theta2)
        a3 = self.sigmoid(z3)
        return a3

    #find the gradients
    def find_gradients(self, weights):
        theta1, theta2 = self.unroll(weights) #
        z2 = np.dot(self.X, theta1) #120 x k
        a2 = self.sigmoid(z2) #120 x k
        # appending the bias units to our new inputs
        a2 = np.hstack((np.ones((np.shape(a2)[0], 1)), a2))
        z3 = np.dot(a2, theta2)
        a3 = self.sigmoid(z3) #120 x 3
        error3 = a3 - self.Y #120 x 3
        z2 = np.hstack((np.ones((np.shape(z2)[0], 1)), z2))  #120 x k+1
        delta_z2 = self.delta_sigmoid(z2)
        error2 = np.multiply(np.matmul(error3, theta2.T), delta_z2) #120 x k+1
        m = np.shape(self.X)[0]
        grad_2 = np.dot(a2.T, error3)/m # 3 x 3
        grad_1 = np.dot(self.X.T, error2)/m # 5 x 4
        grad_1 = grad_1[:, 1:] #5 x 3
        total_grad = self.roll(grad_1, grad_2)

        return total_grad.flatten()

    #used to confirm that gradients are calculated correctly
    def gradient_check(self):
        Epsilon = pow(10, -4)
        approx_gradients= []
        for i in range(0, np.shape(self.weights)[0]):
            theta_plus = self.weights.tolist()
            theta_minus = self.weights.tolist()
            theta_plus[i][0] += Epsilon
            theta_minus[i][0] -= Epsilon
            calc_grad = (self.getCost(np.array(theta_plus)) - self.getCost(np.array(theta_minus)))/(2*Epsilon)
            approx_gradients.append(calc_grad)
        approx_gradients = np.array(approx_gradients)
        approx_gradients = np.reshape(approx_gradients, newshape=(approx_gradients.shape[0], 1))

        return approx_gradients

    def predict_for_testing(self, weights, features, labels):
        X, Y = getData(features, labels)
        theta1, theta2 = self.unroll(weights)
        z2 = np.dot(X, theta1)
        a2 = self.sigmoid(z2)
        bias_units = np.ones((np.shape(a2)[0], 1))
        a2 = np.hstack((bias_units, a2))
        z3 = np.dot(a2, theta2)
        a3 = self.sigmoid(z3)
        counter = 0
        for i in range(np.shape(a3)[0]):
            max_pred = np.argmax(a3[i], 0)
            max_ans = np.argmax(Y[i], 0)
            if(max_pred == max_ans):
                counter+= 1
            print("ans:", max_ans, "pred:", max_pred)
        print("Accuracy:", counter/np.shape(a3)[0] * 100)
        return a3


    def unroll(self, weights):
        #unrolling theta1
        stored_weights = np.array(weights.tolist())[0:np.shape(self.X)[1] * self.hidden_size]
        theta1 = np.reshape(stored_weights, newshape=(np.shape(self.X)[1], self.hidden_size))
        #unrolling theta2
        stored_weights = np.array(weights.tolist())[np.shape(self.X)[1] * self.hidden_size:]
        theta2 = np.reshape(stored_weights, newshape=(self.hidden_size + 1, self.output_size))
        return theta1, theta2

    #used to roll both thetas
    def roll(self, theta1, theta2):
        rolled_t1 = np.reshape(theta1, newshape=(np.shape(theta1)[0] * np.shape(theta1)[1], 1))
        rolled_t2 = np.reshape(theta2, newshape = (np.shape(theta2)[0] * np.shape(theta2)[1], 1))
        return np.vstack((rolled_t1, rolled_t2))

    def train(self, iter):
        result = optimize.fmin_cg(f=self.getCost, x0 = self.weights.flatten(), fprime= self.find_gradients, maxiter= iter)
        self.weights = result

    '''
    The following below will be the activation functions with their derivatives
    '''
    def sigmoid(self, n):
        return 1/(1 + np.power(e, -1 * n))

    def delta_sigmoid(self, n):
        return self.sigmoid(n) * self.sigmoid(1-n)
    '''
    The following above will give the activation functions with their derivatives
    '''

    '''
    Costs functions will be below
    '''

    def crossEntropy(self, H):
        return -1 * (np.multiply(self.Y, np.log(H)) + np.multiply(1-self.Y, np.log(1-H)))

X,Y = getData('X_train.mat', 'Y_train.mat')
Botanist = NeuralNetwork(X_train=X, Y_train=Y, hidden_size=3, reg=0)
Botanist.train(100)
Botanist.predict_for_testing(Botanist.weights, 'Test/X_test.mat', 'Test/Y_test.mat')

