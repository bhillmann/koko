import numpy as np
from random import shuffle


def normalize(X):
    """
    Row-wise subtract the mean divide by the standard deviation
    :param X: the matrix to be normalized
    :return: the normalized matrix
    """
    X = np.atleast_2d(X)
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    return drop_nans(X)


def drop_nans(X):
    """
    Drop all rows with NAs
    :param X: the matrix to drop rows with NAs
    :return: the matrix with all NA rows dropped
    """
    mask = np.all(np.isnan(X.T) | np.equal(X.T, 0), axis=1)
    X = X.T[~mask].T
    return X


def cross_val_score(clf, X: np.ndarray, y: np.ndarray, cv=10):
    """
    Do k-fold crossfold and return the true positive rate
    :param clf: a classifier
    :param X: the data matrix
    :param y: the target class vector
    :param cv: number of folds
    :return: the true positive rate per fold
    """
    accuracy = np.zeros(cv)
    splits = cross_validation_splits(y.shape[0], cv)
    for i, (training, validation) in enumerate(splits):
        clf.fit(X[training], y[training])
        prediction = clf.predict(X[validation])
        accuracy[i] = get_accuracy(prediction, y[validation])
    return accuracy


def get_accuracy(u, v):
    """
    Return the true positive rate
    :param u: predicted target vector
    :param v: target vector
    :return: the true positive rate
    """
    return np.sum(u == v)/v.shape[0]


def cross_validation_splits(num_samples, cv):
    """
    Given the number of samples, return the splits for cross-validation
    :param num_samples: the number of samples
    :param cv: the number of cross validation folds
    :return: generates a (training, validation) tuple containing the index of training and validation splits
    """
    nums = list(range(num_samples))
    shuffle(nums)
    for fold in range(cv):
        training = [x for i, x in enumerate(nums) if i % cv != fold]
        validation = [x for i, x in enumerate(nums) if i % cv == fold]
        yield training, validation


def add_intercept(X):
    """
    Add an intercept to data matrix X
    :param X: a numpy 2d data matrix
    :return: the data matrix with an intercept
    """
    X_without = np.zeros((X.shape[0], X.shape[1] + 1))
    X_without[:, :-1] = X
    X_without[:, -1].fill(1.)
    return X_without
