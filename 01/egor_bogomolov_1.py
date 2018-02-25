import numpy as np
import pandas as pd
from collections import Counter

file = 'wine.csv'
target_label = 'Class 1/2/3'
distance_label = '__distance__'


def train_test_split(X, y, ratio, random_seed=None):
    if len(X) != len(y):
        raise ValueError("X and y should have the same length")
    size = len(X)
    train_size = int(size * ratio)

    if random_seed is not None:
        np.random.seed(random_seed)
    indexes = np.arange(size)
    np.random.shuffle(indexes)

    X_train = X.iloc[indexes[:train_size]]
    y_train = y.iloc[indexes[:train_size]]
    X_test = X.iloc[indexes[train_size:size]]
    y_test = y.iloc[indexes[train_size:size]]

    return X_train, y_train, X_test, y_test


def knn(X_train, y_train, X_test, k=1, dist=np.linalg.norm):
    y = []
    X_joined = X_train.join(y_train)
    for index, row in X_test.iterrows():
        print(index)
        X_joined[distance_label] = [dist(r - row) for i, r in X_train.iterrows()]
        X_sorted = X_joined.sort_values(by=distance_label)
        counts = Counter()
        for i in range(k):
            counts[X_sorted.iloc[i][target_label]] += 1
        target, occurrences = counts.most_common(1)[0]
        y.append(int(target))
    return y


def print_precision_recall(y_pred, y_test):
    y_test = y_test.tolist()
    print(y_pred)
    print(y_test)
    n_classes = len(np.unique(y_test))
    fp = Counter()  # false positive
    fn = Counter()  # false negative
    tp = Counter()  # true positive

    for i in range(len(y_test)):
        if y_pred[i] == y_test[i]:
            tp[int(y_test[i])] += 1
        else:
            fn[int(y_test[i])] += 1
            fp[int(y_pred[i])] += 1

    for c in range(1, n_classes + 1):
        precision = recall = 0
        if tp[c] + fp[c]:
            precision = tp[c] / (tp[c] + fp[c])
        if tp[c] + fn[c] != 0:
            recall = tp[c] / (tp[c] + fn[c])
        print(c, precision, recall)


def loocv(X_train, y_train, dist=np.linalg.norm):
    missed = Counter()
    print(X_train)
    print(y_train)
    for k in range(1, len(X_train) + 1):
        print(k)
        missed[k] = 0
        for index, row in X_train.iterrows():
            result = knn(
                X_train.drop([index], axis=0),
                y_train.drop([index], axis=0),
                X_train.ix[index].to_frame(),
                k=k,
                dist=dist)
            if result != y_train.iloc(index):
                missed[k] += 1

    opt_k, misses = missed.most_common(1)[-1]
    print(opt_k, misses)
    print(missed.most_common(1)[0])
    return opt_k


def main():
    data = pd.read_csv(file)
    X = data.drop(target_label, 1)
    y = data[target_label]
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.5, random_seed=239)
    y_pred = knn(X_train, y_train, X_test, k=1)
    print_precision_recall(y_pred, y_test)
    print(loocv(X_train, y_train))


if __name__ == '__main__':
    main()
