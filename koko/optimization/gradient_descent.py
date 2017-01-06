from random import sample
import numpy as np


def minibatch_gradient_descent(X, y, eta, compute_gradient, epochs, batch_size=32):
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
