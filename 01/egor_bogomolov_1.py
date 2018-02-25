import numpy as np


def train_test_split(X, y, ratio):
    if len(X) != len(y):
        raise ValueError("X and y should have the same length")
    size = len(X)
    train_size = size * ratio
    test_size = size - train_size
    indexes = np.arange(size)
    np.random.shuffle(indexes)
    X_train = y_train = X_test = y_test = []
    for i in range(size):
        if i < train_size:
            
    return X_train, y_train, X_test, y_test