"""
    This is an implementation of the mini-batch gradient descent as described in:
        Pegasos: Primal Estimated sub-GrAdient SOlver for SVM

    That can be found online at:
        http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf
"""

import numpy as np
from sklearn import datasets
from sklearn.linear_model import Perceptron as sk_Perceptron

from koko.optimization import minibatch_gradient_descent
from koko.utils import normalize, add_intercept, cross_val_score


class LinearPerceptron:
    def __init__(self, epochs=5, eta=.01, optimization=None):
        self.n_classes = None
        self.weights = None
        self.epochs = epochs
        self.eta = eta
        if not optimization:
            self.optimization = lambda X, y: minibatch_gradient_descent(X, y,  self.eta, self._compute_gradient,
                                                                        self.epochs)

    def fit(self, X, y):
        X = add_intercept(X)

        self.n_classes = np.unique(y).shape[0]

        if self.n_classes != 2:
            Exception()

        self.classes = dict(zip((0., 1.), np.unique(y)))

        target = np.array([1 if self.classes[1] == _ else 0. for _ in y])
        self.weights = self.optimization(X, target)

    def _compute_gradient(self, X, y, weights):
        y = np.atleast_1d(y)
        X = np.atleast_2d(X)
        errors = y-self._activation_function(np.inner(weights, X))
        return -np.inner(errors, X.T)

    def predict(self, X):
        X = add_intercept(X)
        temp = self._activation_function(np.inner(self.weights, X))
        return np.array([self.classes[_] for _ in temp.astype(int)])

    def _activation_function(self, x):
        return np.array(x >= 0., dtype=np.int)

if __name__ == '__main__':
    data = datasets.load_boston()
    X = data.data
    y = data.target

    X = normalize(X)

    y = y > np.percentile(y, 50)
    y = y.astype(np.int)

    y = y == 1
    y = y.astype(int)

    clf = LinearPerceptron(epochs=10, eta=.001)
    accuracy = np.array(cross_val_score(clf, X, y))
    print(np.mean(accuracy, axis=0))
    print(np.std(accuracy, axis=0))

    sk_clf = sk_Perceptron()
    sk_accuracy = np.array(cross_val_score(sk_clf, X, y))
    print(np.mean(sk_accuracy, axis=0))
    print(np.std(sk_accuracy, axis=0))

