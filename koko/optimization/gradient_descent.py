from random import sample
import numpy as np


def minibatch_gradient_descent(X, y, eta, compute_gradient, epochs, batch_size=32):
    """
    Run minibatch gradient descent and return the weights
    :param X: the data matrix
    :param y: the target vector
    :param eta: the learning rate
    :param compute_gradient: a function that given X, y returns the gradient
    :param epochs: the number of passes through the data
    :param batch_size: the size of the minibatch (1 = stochastic, < 0 = normal)
    :return: The weights after minibatch gradient descent
    """
    n_samples, n_features = X.shape

    weights = np.zeros(n_features)

    if batch_size > n_samples or batch_size < 0:
        batch_size = n_samples

    t = int((n_samples/batch_size)*epochs)

    for i in range(t):
        samples_idx = sample(list(range(n_samples)), batch_size)
        phi = np.array(compute_gradient(X[samples_idx], y[samples_idx], weights))
        weights -= eta * phi
    return weights
