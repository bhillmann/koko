"""
    This is an implementation of the mini-batch gradient descent as described in:
        Pegasos: Primal Estimated sub-GrAdient SOlver for SVM

    That can be found online at:
        http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf
"""

import numpy as np
from sklearn import datasets

from ml.optimization import stochastic_variance_reduced_gradient_descent_loss, stochastic_variance_reduced_gradient_descent
from ml.utils import normalize, cross_val_score, add_intercept, euclidean


class LinearPerceptron:
    def __init__(self, margin=2., m=30, epochs=1000, eta=.001, optimization=None, loss=False, kernel=np.inner):
        self.margin = margin
        self.n_classes = None
        self.weights = None
        self.epochs = epochs
        self.eta = eta
        self.m = m
        self.y = None
        self.kernel = kernel
        if not optimization:
            if loss:
                self.optimization = lambda X, y: stochastic_variance_reduced_gradient_descent_loss(X, y, self._compute_gradient, self.eta, self.epochs, self.m, self._loss_function)
            else:
                self.optimization = lambda X, y: stochastic_variance_reduced_gradient_descent(X, y,
                                                                                                   self._compute_gradient,
                                                                                                   self.eta,
                                                                                                   self.epochs, self.m)

    def fit(self, X, y):
        X = add_intercept(X)
        n_samples, n_features = X.shape

        self.n_classes = np.unique(y).shape[0]

        if self.n_classes != 2:
            Exception()

        self.classes = dict(zip((0., 1.), np.unique(y)))

        self.y = np.array([1 if self.classes[1] == _ else 0. for _ in y])
        self.weights, loss = self.optimization(X, self.y)

        return loss

    def _compute_gradient(self, X, y, weights):
        y = np.atleast_1d(y)
        X = np.atleast_2d(X)
        result = self.kernel(weights, X)
        return y - self._activation_function(result)

    def predict(self, X):
        X = add_intercept(X)
        self.y = np.atleast_1d(self.y)
        X = np.atleast_2d(X)
        temp = self.kernel(self.weights, X) > 0
        return np.array([self.classes[_] for _ in temp.astype(int)])

    def predict_proba(self, X):
        pass

    def _activation_function(self, x):
        return np.array(x < 0., dtype=np.int)

    def _loss_function(self, X, y, weights):
        n_samples, n_features = X.shape

        first_half = (self.margin/2.)*(weights.dot(weights))
        temp = 1 - y * np.inner(weights, X)
        temp[temp < 0] = 0
        temp = np.sum(temp)
        temp = (1./n_samples) * temp
        return first_half + temp

if __name__ == '__main__':
    data = datasets.load_digits()
    X = data.data
    y = data.target

    X = normalize(X)

    # y = y > np.percentile(y, 75)
    # y = y.astype(np.int)

    y = y == 3
    y = y.astype(int)

    clf = LinearPerceptron(loss=False, epochs=100, kernel=euclidean)
    # clf.fit(X, y)
    accuracy = np.array(cross_val_score(clf, X, y))
    print(np.mean(accuracy, axis=0))
    print(np.std(accuracy, axis=0))

    # sk_clf = sk_LogisticRegression()
    # accuracy = np.array(cross_val_score(sk_clf, X, y))
    # print(np.mean(accuracy, axis=0))
    # print(np.std(accuracy, axis=0))

