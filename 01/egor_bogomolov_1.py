import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial import distance
from sklearn import preprocessing
from datetime import datetime

filename = 'wine.csv'
target_label = 'Class 1/2/3'
distance_label = '__distance__'


def read_data(fname):
    print("Reading data from {0}".format(fname))
    return pd.read_csv(fname)


def split_data(data):
    print("Splitting data into X and y, target label is {0}".format(target_label))
    X = data.drop(target_label, 1)
    y = data[target_label]
    return X, y


def normalize_data(data):
    print("Normalizing data with MinMaxScaler")
    x = data.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return pd.DataFrame(x_scaled)


def train_test_split(X, y, ratio):
    print("Splitting data, train ratio is {0}".format(ratio))
    if len(X) != len(y):
        raise ValueError("X and y should have the same length")
    size = len(X)
    train_size = int(size * ratio)

    indexes = np.arange(size)
    np.random.shuffle(indexes)

    X_train = X.iloc[indexes[:train_size]]
    y_train = y.iloc[indexes[:train_size]]
    X_test = X.iloc[indexes[train_size:size]]
    y_test = y.iloc[indexes[train_size:size]]

    return X_train, y_train, X_test, y_test


def knn(X_train, y_train, X_test, k=1, dist=distance.euclidean):
    y_test = []
    X_joined = X_train.join(y_train)
    for index, row in X_test.iterrows():
        X_joined[distance_label] = [dist(r, row) for i, r in X_train.iterrows()]
        X_sorted = X_joined.sort_values(by=distance_label)
        counts = Counter()
        for i in range(k):
            counts[X_sorted.iloc[i][target_label]] += 1
        target, occurrences = counts.most_common(1)[0]
        y_test.append(int(target))
    return y_test


def print_precision_recall(y_pred, y_test):
    y_test = y_test.tolist()
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
        precision = recall = 0.
        if tp[c] + fp[c] != 0:
            precision = tp[c] / float(tp[c] + fp[c])
        if tp[c] + fn[c] != 0:
            recall = tp[c] / float(tp[c] + fn[c])
        print("For class {0}:\nprecision = {1}, recall = {2}".format(c, precision, recall))


def loocv(X_train, y_train, dist=distance.euclidean, print_rate=10):
    size = len(X_train)
    min_k = 1
    max_k = min(30, len(X_train))
    print("LOOCV, choosing among k from {0} to {1}".format(min_k, max_k))
    start = datetime.now()
    missed = Counter()
    for k in range(min_k, max_k + 1):
        for index, row in X_train.iterrows():
            result = knn(
                X_train.drop([index], axis=0),
                y_train.drop([index], axis=0),
                row.to_frame().transpose(),
                k=k,
                dist=dist)
            if result != y_train[index]:
                missed[k] += 1
        opt_k, misses = missed.most_common()[-1]
        if k % print_rate == 0:
            print("Processed k up to {0}\nBest k = {1}, precision = {2}"
                  .format(k, opt_k, 1. - float(misses) / (size - 1)))

    opt_k, misses = missed.most_common()[-1]
    finish = datetime.now()
    print("LOOCV finished in {0} sec\nBest k = {1}, precision = {2}"
          .format((finish - start).total_seconds(), opt_k, misses))
    plt.figure(figsize=(9, 9))
    plt.plot(np.arange(min_k, max_k + 1), missed.values())
    plt.ylabel('LOO')
    plt.xlabel('Number of K')
    plt.show()
    return opt_k


def run_loocv_with_results(X_train, y_train, X_test, y_test, dist=distance.euclidean):
    k = loocv(X_train, y_train, dist)
    y_pred = knn(X_train, y_train, X_test, k, dist)
    print_precision_recall(y_pred, y_test)


def main():
    data = read_data(filename)
    X, y = split_data(data)
    X = normalize_data(X)
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.7)
    print("\nRunning LOOCV for Euclidean distance:")
    run_loocv_with_results(X_train, y_train, X_test, y_test, distance.euclidean)
    print("\nRunning LOOCV for Cityblock distance:")
    run_loocv_with_results(X_train, y_train, X_test, y_test, distance.cityblock)


if __name__ == '__main__':
    main()
