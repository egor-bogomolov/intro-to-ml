import codecs
import numpy as np
from sklearn.model_selection import train_test_split

input_file = 'spam'


def read_file(filename):
    labels = []
    messages = []
    for line in codecs.open(filename, "r", "utf_8_sig"):
        words = line.split()
        labels.append(words[0])
        words = words[1:]
        for i in range(len(words)):
            while len(words[i]) > 0 and not (words[i][0].isdigit() or words[i][0].isalpha()):
                words[i] = words[i][1:]
            while len(words[i]) > 0 and not (words[i][-1].isdigit() or words[i][-1].isalpha()):
                words[i] = words[i][:-1]
        messages.append(words)
    return labels, messages


def vectorize(messages):
    n = len(messages)
    words = np.unique(np.concatenate(messages).ravel())
    m = len(words)
    ind = {}
    for i in range(m):
        ind[words[i]] = i
    vectorized_array = np.zeros([n, m])
    for i in range(n):
        line = messages[i]
        for word in line:
            vectorized_array[i][ind[word]] += 1
    return vectorized_array, words


class NaiveBayes:

    alpha = 1
    prob_class = []
    count_word_class = []
    count_class = []
    prob_word_class = []
    words = []
    words_mapping = {}
    labels = []
    labels_mapping = {}

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):
        print("Fitting Naive Bayes classifier with {} items".format(len(X)))
        print("Vectorizing input...")
        X, self.words = vectorize(X)
        self.labels, counts = np.unique(y, return_counts=True)
        n = len(X)
        m = len(self.words)
        num_classes = len(self.labels)
        print("Number of labels: {0}, number of words: {1}".format(num_classes, m))
        print("Computing probabilities...")
        for i in range(num_classes):
            self.labels_mapping[self.labels[i]] = i
        for i in range(m):
            self.words_mapping[self.words[i]] = i
        for i in range(n):
            for j in range(num_classes):
                if y[i] == self.labels[j]:
                    y[i] = j
                    break

        self.prob_class = np.zeros(num_classes)
        for i in range(num_classes):
            self.prob_class[i] = float(counts[i]) / n

        self.prob_word_class = np.zeros([num_classes, m])
        self.count_word_class = np.zeros([num_classes, m])
        self.count_class = np.zeros(num_classes)
        for x, cls in zip(X, y):
            self.count_word_class[cls] += x
            self.count_class[cls] += sum(x)
        for i in range(num_classes):
            for j in range(m):
                self.prob_word_class[i][j] = \
                    (self.alpha + self.count_word_class[i][j]) / (self.alpha * m + self.count_class[i])
        print("Classifier was trained, alpha = {}".format(self.alpha))

    def vectorize_input(self, x):
        vector = np.zeros(len(self.words))
        for word in x:
            if word in self.words_mapping:
                vector[self.words_mapping[word]] += 1
        return vector

    def predict(self, X):
        X = self.vectorize_input(X)
        label = None
        best = 0
        for i in range(len(self.labels)):
            value = np.log(self.prob_class[i]) + sum(X * np.log(self.prob_word_class[i]))
            if label is None or best < value:
                best = value
                label = self.labels[i]
        return label

    def score(self, X, ys):
        print("Computing score on {} items".format(len(X)))
        correct = 0
        for x, y in zip(X, ys):
            if self.predict(x) == y:
                correct += 1
        return float(correct) / len(X)


def main():
    labels, messages = read_file(input_file)
    X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.2)
    clf = NaiveBayes(1.)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))


if __name__ == '__main__':
    main()
