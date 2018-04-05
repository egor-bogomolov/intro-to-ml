from collections import Counter
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

input_file = 'pima-indians-diabetes.csv'


def read_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    n = data.shape[0]
    n_features = data.shape[1]
    X, y = np.empty([n, n_features]), np.empty(n)
    for i in range(n):
        if data[i][-1] == 0:
            y[i] = -1
        else:
            y[i] = 1
        X[i] = np.append([-1], data[i][:-1])
    return X, y


def normalize(X):
    return StandardScaler().fit_transform(X)


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

    for c in set(y_test):
        precision = recall = 0.
        if tp[c] + fp[c] != 0:
            precision = tp[c] / float(tp[c] + fp[c])
        if tp[c] + fn[c] != 0:
            recall = tp[c] / float(tp[c] + fn[c])
        print("For class {0}:\nprecision = {1}, recall = {2}".format(c, precision, recall))


def log_loss(M):
    return np.log2(1. + np.exp(-M)), -1. / (np.log(2.) * (np.exp(M) + 1.))


def sigmoid_loss(M):
    return 2. / (1. + np.exp(M)), -2. * np.exp(M) / (1 + np.exp(M)) ** 2


class GradientDescent:

    def __init__(self, *, alpha, threshold=1e-3, loss=sigmoid_loss):
        if alpha <= 0:
            raise ValueError("alpha should be positive")
        if threshold <= 0:
            raise ValueError("threshold should be positive")
        self.alpha = alpha
        self.threshold = threshold
        self.loss = loss
        self.weights = None

    def raw_predict(self, X):
        return np.dot(X, self.weights)

    def fit(self, X, y, max_steps=1e10):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        errors = []
        self.weights = np.random.random(n_features) - 0.5
        w = self.weights + self.threshold * 2
        step = 1
        while np.linalg.norm(w - self.weights) >= self.threshold and step <= max_steps:
            vec_loss, vec_der = self.loss(self.raw_predict(X) * y)
            grad = np.zeros(n_features)
            for i in range(n_samples):
                grad += X[i] * y[i] * vec_der[i]
            self.weights, w = self.weights - self.alpha * grad / n_samples, self.weights
            error = sum(vec_loss) / n_samples
            errors.append(error)
            step += 1
        return errors

    def predict(self, X):
        return np.sign(self.raw_predict(X))


class SGD:
    def __init__(self, alpha, loss=log_loss, k=1, n_iter=1000):
        if alpha <= 0:
            raise ValueError("alpha should be positive")
        if k <= 0 or not isinstance(k, int):
            raise ValueError("k should be a positive integer")
        if n_iter <= 0 or not isinstance(n_iter, int):
            raise ValueError("n_iter should be a positive integer")
        self.k = k
        self.n_iter = n_iter
        self.alpha = alpha
        self.loss = loss
        self.weights = None

    def raw_predict(self, X):
        return np.dot(X, self.weights)

    def fit(self, X, y):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        errors = []
        self.weights = np.random.random(n_features) - 0.5
        eta = float(self.k) / n_samples
        vec_loss, vec_der = self.loss(self.raw_predict(X) * y)
        cur_loss = sum(vec_loss) / n_samples
        errors.append(cur_loss)
        for i in range(self.n_iter):
            indices = np.random.choice(n_samples, self.k)
            batchX, batchY = X[indices], y[indices]
            vec_loss, vec_der = self.loss(self.raw_predict(batchX) * batchY)
            error = sum(vec_loss) / self.k
            grad = np.zeros(n_features)
            for j in range(self.k):
                grad += batchX[j] * batchY[j] * vec_der[j]
            self.weights -= self.alpha * grad / self.k
            cur_loss = (1 - eta) * cur_loss + eta * error
            errors.append(cur_loss)
        return errors

    def predict(self, X):
        return np.sign(self.raw_predict(X))


def prepare_data():
    X, y = read_data(input_file)
    X = normalize(X)
    for row in X:
        row[0] = -1.
    return X, y


def plot_grad_descent(X, y, alphas, losses):
    print("Plotting gradient descent with different parameters")
    for i, loss in enumerate(losses):
        for alpha in alphas:
            errors = GradientDescent(alpha=alpha, loss=loss, threshold=1e-12).fit(X, y, max_steps=2000)
            plt.figure(i)
            plt.plot(np.arange(1, len(errors) + 1), errors, label=alpha)
        plt.legend(title=loss.__name__)
        plt.savefig('gradient_descent_' + loss.__name__ + '.png')
    plt.close()


def plot_sgd(X, y, alphas, losses, ks):
    print("Plotting sgd with different parameters")
    for i, loss in enumerate(losses):
        for j, k in enumerate(ks):
            for alpha in alphas:
                errors = SGD(alpha=alpha, loss=loss, n_iter=3000).fit(X, y)
                plt.figure(i)
                plt.subplot(1, len(ks), j + 1)
                plt.plot(np.arange(1, len(errors) + 1), errors, label=alpha)
        plt.legend(title=loss.__name__)
        plt.savefig('sgd_' + loss.__name__ + '.png')
    plt.close()


def gradient_descent(X_train, X_test, y_train, y_test):
    print("Testing gradient descent")
    clf = GradientDescent(alpha=0.01, loss=sigmoid_loss)
    clf.fit(X_train, y_train)
    print_precision_recall(clf.predict(X_test), y_test)


def stochastic_gradient_descent(X_train, X_test, y_train, y_test):
    print("Testing SGD")
    clf = SGD(alpha=.01, loss=sigmoid_loss)
    clf.fit(X_train, y_train)
    print_precision_recall(clf.predict(X_test), y_test)


def main():
    np.random.seed(15)
    X, y = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    gradient_descent(X_train, X_test, y_train, y_test)
    stochastic_gradient_descent(X_train, X_test, y_train, y_test)
    plot_grad_descent(X, y, alphas=[1e-6, 1e-4, 1e-2, 1.], losses=[log_loss, sigmoid_loss])
    plot_sgd(X, y, alphas=[1e-6, 1e-4, 1e-2, 1.], losses=[log_loss, sigmoid_loss], ks=[1, 10, 500])


if __name__ == '__main__':
    main()
