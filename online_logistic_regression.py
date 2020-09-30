from scipy.optimize import minimize
import numpy as np


# Disclaimer: this class is taken from:
# https://gdmarmerola.github.io/ts-for-contextual-bandits/


# Defining a class for Online Bayesian Logistic Regression
class OnlineLogisticRegression:

    # Initializing
    def __init__(self, lambda_, alpha, n_dim, bias, maxiter = 15):

        # Hyperparameter: deviation on the prior (L2 regularizer)
        self.lambda_ = lambda_; self.alpha = alpha; self.maxiter = maxiter

        # Initializing parameters of the model
        self.n_dim = n_dim
        # m: mean of the Bi, q inverse variance of the distribution
        self.m = np.zeros(self.n_dim)
        self.m[-1] = bias
        self.q = np.ones(self.n_dim) * self.lambda_

        # Initializing weights
        self.w = np.random.normal(self.m, self.alpha * (self.q)**(-1.0), size = self.n_dim)
        
    # Loss function
    def loss(self, w, *args):
        X, y = args
        # Note: the bias is removed from the "regularization term" of the loss
        return 0.5 * (self.q[:-1] * (w[:-1] - self.m[:-1])).dot(w[:-1] - self.m[:-1]) + np.sum([np.log(1 + np.exp(-y[j] * w.dot(X[j]))) for j in range(y.shape[0])])

    # Gradient
    def grad(self, w, *args):
        X, y = args
        return np.concatenate((self.q[:-1] * (w[:-1] - self.m[:-1]),0.0),axis = None) + (-1) * np.array([y[j] *  X[j] / (1. + np.exp(y[j] * w.dot(X[j]))) for j in range(y.shape[0])]).sum(axis = 0)

    # Fitting method
    def fit(self, X, y):
                
        # Step 1, find w
        self.w = minimize(self.loss, self.w, args = (X, y), jac = self.grad, method = "L-BFGS-B", options = {'maxiter': self.maxiter}).x
        self.m = self.w
        
        # Step 2, update q
        P = (1 + np.exp(1 - X.dot(self.m))) ** (-1)
        self.q = self.q + (P*(1-P)).dot(X ** 2)