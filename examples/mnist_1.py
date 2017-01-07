from sklearn import datasets
from sklearn.linear_model import Perceptron as sk_Perceptron

from koko.classifiers import LinearPerceptron
from koko.utils import normalize, cross_val_score

import numpy as np

# Load the data
data = datasets.load_digits()
X = data.data
y = data.target

# Normalize the data
# Subtract the mean divide by the standard deviation
# Done row-wise
X = normalize(X)

# Change the multiclass prediction to 1 vs all
y = np.array(y == 1, dtype=np.int)

# Create a perceptron classifier
clf = LinearPerceptron(epochs=10, eta=.001)

# Run 10-fold cross-validation
accuracy = np.array(cross_val_score(clf, X, y))
print(np.mean(accuracy, axis=0))
print(np.std(accuracy, axis=0))

# Run 10-fold cross-validation of the sklearn perceptron
sk_clf = sk_Perceptron()
sk_accuracy = np.array(cross_val_score(sk_clf, X, y))
print(np.mean(sk_accuracy, axis=0))
print(np.std(sk_accuracy, axis=0))
