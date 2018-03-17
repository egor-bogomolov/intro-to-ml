import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw

filename = 'halloween.csv'
target_label = 'type'
random_state = 239
K = 10


def entropy(items):
    total = len(items)
    labels, counts = np.unique(items, return_counts=True)
    resulting_entropy = 0
    for count in counts:
        prob = float(count) / total
        resulting_entropy -= prob * np.log2(prob)
    return resulting_entropy


def is_column_numeric(column):
    return np.issubdtype(column.dtype, np.number)


def is_item_numeric(item):
    return np.issubdtype(type(item), np.number)


class DecisionTree:
    node = None
    number_of_elements = 0

    def get_width(self):
        width = 1
        if self.node.false_branch is not None:
            width += self.node.false_branch.get_width()
        if self.node.true_branch is not None:
            width += self.node.true_branch.get_width()
        return width

    def get_depth(self):
        depth = 1
        if self.node.false_branch is not None:
            depth = max(depth, self.node.false_branch.get_depth() + 1)
        if self.node.true_branch is not None:
            depth = max(depth, self.node.true_branch.get_depth() + 1)
        return depth

    def build(self, X, y, score=entropy):
        # рекурсивный алгоритм построения дерева
        labels = np.unique(y)
        self.number_of_elements = len(X)
        if len(labels) == 1:
            print("All elements are of type {}".format(labels[0]))
            self.node = Node(predicate=(None, labels[0]))
        else:
            current_entropy = score(y)
            total = len(X)
            max_info_gain = 0
            predicate = (None, None)
            best_true_X, best_true_y, best_false_X, best_false_y = None, None, None, None
            for feature in X:
                numeric = is_column_numeric(X[feature])
                values = np.unique(X[feature])
                for value in values:
                    if numeric:
                        true_X = X.loc[X[feature] >= value]
                        false_X = X.loc[X[feature] < value]
                    else:
                        true_X = X.loc[X[feature] == value]
                        false_X = X.loc[X[feature] != value]
                    true_y = y[true_X.index]
                    false_y = y[false_X.index]
                    info_gain = current_entropy - (
                                entropy(true_y) * len(true_y) + entropy(false_y) * len(false_y)) / total
                    if info_gain >= max_info_gain:
                        max_info_gain = info_gain
                        predicate = (feature, value)
                        best_true_X, best_true_y, best_false_X, best_false_y = true_X, true_y, false_X, false_y

            print("Splitting by {0}, value = {1}".format(predicate[0], predicate[1]))
            self.node = Node(predicate,
                             build_decision_tree(best_true_X, best_true_y),
                             build_decision_tree(best_false_X, best_false_y))
        return self

    def predict(self, x):
        feature, value = self.node.predicate
        if feature is None:
            return value
        elif is_item_numeric(x[feature]):
            if x[feature] >= value:
                return self.node.true_branch.predict(x)
            else:
                return self.node.false_branch.predict(x)
        else:
            if x[feature] == value:
                return self.node.true_branch.predict(x)
            else:
                return self.node.false_branch.predict(x)


class Node:
    false_branch = None
    true_branch = None
    predicate = None

    def __init__(self, predicate, true_branch=None, false_branch=None):
        self.predicate = predicate
        self.true_branch = true_branch
        self.false_branch = false_branch


def read_data(fname):
    print("Reading data from {0}".format(fname))
    return pd.read_csv(fname)


def split_data(data):
    print("Splitting data into X and y, target label is {0}".format(target_label))
    X = data.drop(target_label, 1)
    y = data[target_label]
    return X, y


def build_decision_tree(X, y):
    print("Building decision tree from {} elements".format(len(X)))
    return DecisionTree().build(X, y)


def get_accuracy(tree, X, y):
    print("Computing accuracy for {} samples".format(len(X)))
    total = len(X)
    correct = 0
    for (index, row), ans in zip(X.iterrows(), y):
        if tree.predict(row) == ans:
            correct += 1
    accuracy = float(correct) / total
    print("accuracy = {0:.3f}".format(accuracy))
    return accuracy


def drawtree(tree, path='tree.jpg'):
    w = tree.get_width() * 100
    h = tree.get_depth() * 100
    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    drawnode(draw, tree, w / 2, 20)
    img.save(path, 'JPEG')


def drawnode(draw, tree, x, y):
    node = tree.node
    shift = 100
    width1, width2 = 0, 0
    if node.false_branch is not None:
        width1 = node.false_branch.get_width() * shift
    if node.true_branch is not None:
        width2 = node.true_branch.get_width() * shift
    left = x - (width1 + width2) / 2
    right = x + (width1 + width2) / 2
    # получите текстовое представление предиката для текущего узла
    if node.predicate[0] is not None:
        predicate = "Split {0} elements by {1}\nvalue = {2}".format(tree.number_of_elements, node.predicate[0], node.predicate[1])
    else:
        predicate = "{0} elements\n{1}".format(tree.number_of_elements, node.predicate[1])
    draw.text((x - 20, y - 10), predicate, (0, 0, 0))
    if node.false_branch is not None:
        draw.line((x, y, left + width1 / 2, y + shift), fill=(255, 0, 0))
        drawnode(draw, node.false_branch, left + width1 / 2, y + shift)
    if node.true_branch is not None:
        draw.line((x, y, right - width2 / 2, y + shift), fill=(255, 0, 0))
        drawnode(draw, node.true_branch, right - width2 / 2, y + shift)


def average_accuracy(X, y):
    total_acc = 0
    for k in range(K):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        decision_tree = build_decision_tree(X_train, y_train)
        total_acc += get_accuracy(decision_tree, X_test, y_test)
    print("Average accuracy = {0:.3f}".format(total_acc / K))


def build_and_draw(X, y):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0)
    decision_tree = build_decision_tree(X, y)
    # get_accuracy(decision_tree, X_test, y_test)
    drawtree(decision_tree)
    print("Width = {0}, depth = {1}".format(decision_tree.get_width(), decision_tree.get_depth()))


def main():
    data = read_data(filename)
    X, y = split_data(data)
    # average_accuracy(X, y)
    build_and_draw(X, y)


if __name__ == '__main__':
    main()
