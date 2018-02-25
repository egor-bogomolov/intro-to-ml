import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial import distance
from sklearn import preprocessing

filename = 'wine.csv'
target_label = 'Class 1/2/3'
distance_label = '__distance__'


def normalize_data(data):
    x = data.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return pd.DataFrame(x_scaled)


def train_test_split(X, y, ratio):
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
        X_joined.sort_values(by=distance_label, inplace=True)
        counts = Counter()
        for i in range(k):
            counts[X_joined.iloc[i][target_label]] += 1
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
        print(c, precision, recall)


def loocv(X_train, y_train, dist=distance.euclidean):
    missed = Counter()
    for k in range(1, min(len(X_train) + 1, 11)):
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
    plt.figure(figsize=(9, 9))
    plt.plot(np.arange(1, 11), missed.values())
    plt.ylabel('LOO')
    plt.xlabel('Number of K')
    plt.show()
    return opt_k


def main():
    data = pd.read_csv(filename)
    X = data.drop(target_label, 1)
    X = normalize_data(X)
    y = data[target_label]
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.7)
    k = loocv(X_train, y_train)
    y_pred = knn(X_train, y_train, X_test, k)
    print_precision_recall(y_pred, y_test)


if __name__ == '__main__':
    main()
