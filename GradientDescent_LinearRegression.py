""""
Machine Learning - Linear Regression with Gradient Descent Algorithm
version: 0.1
author: Ade Kurniawan
description: this is my personal project to try to build Gradient Descent algorithm from the ground up. If you learn machine learning and data science or even you are a data scientist yourself, you would have no difficulties in understanding this code. The symbols and terms used here are influenced by the ones used in the Andrew Ng's Machine Learning course from Coursera. The structure of the program is inspired from the structure of Scikit-learn
"""

import numpy as np

class LinearRegression:
    def __init__(self, init_weights = 'default', alpha = 0.01, iterations = 1500, C = 0):
        self.w = init_weights # weights
        self.alpha = alpha # learning rate
        self.n_iter = iterations # number of iterations which will be performed
        self.C = C # regularization term
    def fit(self, X, y):
        y = y.reshape(-1,1)
        m, n = X.shape # number of datapoints, number of features/variables
        if self.w == 'default':
            #if default,initial weights are set to zero
            self.w = np.zeros((n+1, 1)) # +1 to include the intercept part of the model
        elif self.w =='random':
            #if random, the initial weights are set using random number generator, near zero value to avoid divergence
            self.w = np.random.normal(0,0.5, (n+1, 1))
        # add one column containing value 1s to  X to accomodate intercept, 
        ones = np.ones((m,1))
        X = np.concatenate((ones,X), axis = 1)
        #everthing is ready, commencing the iterations
        delta = np.concatenate((np.zeros((1,1)), np.ones((n,1))), axis = 0) #simply to exclude w[0] from being regularized (it's a matter of convention as far as I can remember)
        for k in range(self.n_iter):
            h = np.dot(X, self.w)
            grad = np.dot(X.T, h-y).sum(axis=1).reshape(-1, 1)
            self.w = self.w - self.alpha*grad/m - self.C*self.w*delta/m
    def predict(self, X):
        X = X.reshape(-1,1)
        prediction = w[0] + np.sum(w[1:], X)
        return prediction 
    def get_params(self):
        return self.w
# testing the program
# Example 1: y = 2x - 5
# In this example, I want to simulate predicting a model for a given dataset (which I create  using numpy random number generator). The dataset I use here will mimic this linear equation y = 2*x - 5, with adding noise to the target variable to make this simulation more useful
#predictor variable, 20 equally spaced datapoints ranging from 1 to 10
#it is highly advised that the shape of X is adjusted before it is fitted into the model, as the program is still in its infancy (there is no mechanism to automate the reshaping yet)
print("Modeling y = -5 + 2x")
X = np.linspace(1,10,20).reshape(-1,1)
# response variable (aka outcome or target)
y = 2*X - 5 + np.random.normal(0, 1, len(X)).reshape(-1,1)
lr_model = LinearRegression()
lr_model.fit(X,y)
w = lr_model.get_params()
# if the algorithm works properly, we will obtain w[0] around -5 and w[1] around 2
print("The model's weights are: w[0] = {0} and w[1] = {1}\n".format(w[0], w[1]))

#Example 2: y = 10*x1 - 15*x2 + 6
# In this second example, I want to stretch further: I'll try to simulate predictive task in which the dataset contains two predictor variables, x1 and x2. The linear equation in this example is y = 10*x1 - 15*x2 + 6, it means that we will need to train a model to obtain 3 weights corresponding to each predictor variable: w[0] for the intercept part, w[1] for x1 and w[2] for x2.
# predictor variables, It contains 100 datapoints with 2 columns, one for each variable
print("Modeling y = 6 +10x1 -15x2")
X = np.random.normal(0, 3, (100,2))
y = 10*X[:,0] - 15*X[:,1] + 6 + np.random.randint(-3,3,len(X))
lr_model2 = LinearRegression(alpha = 0.05)
lr_model2.fit(X,y)
w2 = lr_model2.get_params()
#if the algorithm works properly, we will obtain w[0] around 6, w[1] around 10, and w[2] around -15
print("The model's weights are: w[0] = {0}, w[1] = {1}, w[2] = {2}".format(w2[0], w2[1], w2[2]))