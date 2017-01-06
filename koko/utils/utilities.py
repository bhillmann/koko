import numpy as np
from random import shuffle


def normalize(X):
    """
    Row wise subtract the mean divide by the standard deviation
    :param X: the matrix to be normalized
    :return: the normalized matrix
    """
    X = np.atleast_2d(X)
    # Add a small amount of noise
    # X += np.random.normal(.00001, .00002, X.shape[1])
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    return drop_nans(X)


def drop_nans(X):
    mask = np.all(np.isnan(X.T) | np.equal(X.T, 0), axis=1)
    X = X.T[~mask].T
    return X


def cross_val_score(clf, X, y, cv=10):
    accuracy = np.zeros(cv)
    splits = cross_validation_splits(y.shape[0], cv)
    for i, (training, validation) in enumerate(splits):
        clf.fit(X[training], y[training])
        prediction = clf.predict(X[validation])
        accuracy[i] = get_accuracy(prediction, y[validation])
    return accuracy


def get_accuracy(u, v):
    return np.sum(u == v)/v.shape[0]


def cross_validation_splits(num_samples, cv):
    nums = list(range(num_samples))
    shuffle(nums)
    for fold in range(cv):
        training = [x for i, x in enumerate(nums) if i % cv != fold]
        validation = [x for i, x in enumerate(nums) if i % cv == fold]
        yield training, validation


def learning_curve(clf, X, y, hold_out_percent=20, percents=(10, 25, 50, 75, 100), k=10):
    accuracies = np.zeros((k, len(percents)))
    for split in range(k):
        num_samples = y.shape[0]
        num_hold_out = int((hold_out_percent/100.) * num_samples)
        nums = list(range(num_samples))
        shuffle(nums)
        hold_out = nums[:num_hold_out]
        train = nums[num_hold_out:]
        for i, percent in enumerate(percents):
            percent_train = int((percent/100.) * num_samples)
            clf.fit(X[train][:percent_train], y[train][:percent_train])
            prediction = clf.predict(X[hold_out])
            accuracies[split, i] = get_accuracy(prediction, y[hold_out])
    return accuracies


def add_intercept(X):
    X_without = np.zeros((X.shape[0], X.shape[1] + 1))
    X_without[:, :-1] = X
    X_without[:, -1].fill(1.)
    return X_without
