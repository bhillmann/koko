"""
    This is an implementation of the mini-batch gradient descent algorithm for a linear perceptron model

    The class model is based off of sklearn:
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
"""
import numpy as np

from koko.optimization import minibatch_gradient_descent
from koko.utils import add_intercept


class LinearPerceptron:
    def __init__(self, epochs=5, eta=.01, optimization=None):
        """
        A Linear Perceptron implementation using Numpy vector optimizations
        :param epochs: the number of passes through the data
        :param eta: the learning rate
        :param optimization: the optimization method given the data matrix X, and the target vector y
        """
        self.n_classes = None
        self.weights = None
        self.epochs = epochs
        self.eta = eta
        if not optimization:
            self.optimization = lambda X, y: minibatch_gradient_descent(X, y,  self.eta, self._compute_gradient,
                                                                        self.epochs)

    def fit(self, X, y):
        """
        Fit the weights of the perceptron
        :param X: the data to fit
        :param y: the target vector
        """
        X = add_intercept(X)

        self.n_classes = np.unique(y).shape[0]

        if self.n_classes != 2:
            Exception()

        self.classes = dict(zip((0., 1.), np.unique(y)))

        target = np.array([1 if self.classes[1] == _ else 0. for _ in y])
        self.weights = self.optimization(X, target)

    def _compute_gradient(self, X, y, weights):
        """
        Computes the gradient of the perceptron model
        :param X: data matrix
        :param y: target vector
        :param weights: current weights
        :return: gradient vector
        """
        y = np.atleast_1d(y)
        X = np.atleast_2d(X)
        errors = y-self._activation_function(np.inner(weights, X))
        return -np.inner(errors, X.T)

    def predict(self, X):
        """
        Use the fitted model to predict new data
        :param X: data to predict
        :return: predicted calls vector
        """
        X = add_intercept(X)
        temp = self._activation_function(np.inner(self.weights, X))
        return np.array([self.classes[_] for _ in temp.astype(int)])

    def _activation_function(self, x):
        """
        The activation function for a perceptron
        :param x: prediction vector
        :return: binary activation vector
        """
        return np.array(x >= 0., dtype=np.int)
