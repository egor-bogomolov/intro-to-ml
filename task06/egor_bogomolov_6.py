import numpy as np
import scipy.io
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from PIL import Image

input_file = 'mnist-original.mat'


def load_data(filename):
    print("Loading data...")
    dataset = scipy.io.loadmat(filename)
    print("Data loaded")
    return dataset


def scale_to_unit_interval(ndar, eps=1e-8):
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                 in zip(img_shape, tile_shape, tile_spacing)]
    H, W = img_shape
    Hs, Ws = tile_spacing

    dt = X.dtype
    if output_pixel_vals:
        dt = 'uint8'
    out_array = np.zeros(out_shape, dtype=dt)

    for tile_row in range(tile_shape[0]):
        for tile_col in range(tile_shape[1]):
            if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                this_x = X[tile_row * tile_shape[1] + tile_col]
                if scale_rows_to_unit_interval:
                    this_img = scale_to_unit_interval(
                        this_x.reshape(img_shape))
                else:
                    this_img = this_x.reshape(img_shape)
                c = 1
                if output_pixel_vals:
                    c = 255
                out_array[
                tile_row * (H + Hs): tile_row * (H + Hs) + H,
                tile_col * (W + Ws): tile_col * (W + Ws) + W
                ] = this_img * c
    return out_array


def visualize_mnist(train_X):
    images = train_X[0:2500, :]
    image_data = tile_raster_images(images,
                                    img_shape=[28, 28],
                                    tile_shape=[50, 50],
                                    tile_spacing=(2, 2))
    im_new = Image.fromarray(np.uint8(image_data))
    im_new.save('mnist.png')


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1. - sigmoid(x))


def mse(X, y):
    return (X - y) ** 2 / 2.


def mse_derivative(X, y):
    return X - y


class NeuralNetwork:

    def __init__(self, layers):
        self.num_layers = len(layers)
        self.weights = []
        self.layers = layers
        self.outputs = []

    def train(self, X, y, max_iter=10000, learning_rate=1.):
        print("Training network on {} samples. {} iterations".format(len(X), max_iter))
        self.weights = [np.random.randn(x + 1, y) for x, y in zip(self.layers[:-1], self.layers[1:])]
        n_samples = len(X)
        eta = 1. / n_samples
        answers = np.zeros([len(y), self.layers[-1]])
        for i, ans in enumerate(y):
            answers[i][ans] = 1

        cur_loss = sum(mse(self.forward(X), answers)) / n_samples
        for i in range(max_iter):
            if (i + 1) % 10000 == 0:
                print("Finished {} iterations".format(i + 1))
            i = np.random.randint(0, n_samples)
            self.forward(X[i])
            cur_loss = (1 - eta) * cur_loss + eta * self.backward(learning_rate, answers[i])

    def forward(self, X):
        self.outputs = [X]
        for w in self.weights:
            X = sigmoid(np.dot(X, w[:-1]) - w[-1])
            self.outputs.append(X)
        return X

    def backward(self, step, y):
        eps = self.outputs[-1] - y
        loss = mse(self.outputs[-1], y)
        for k in reversed(range(self.num_layers - 1)):
            der = sigmoid_derivative(self.outputs[k].dot(self.weights[k][:-1]) - self.weights[k][-1])
            self.weights[k] -= step * np.outer(np.append(self.outputs[k], -1), eps * der)
            eps = self.weights[k].dot(eps * der)[:-1]
        return loss

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)


def main():
    np.random.seed(239)
    data = load_data(input_file)
    trainX, testX, trainY, testY = train_test_split(data['data'].T / 255.0, data['label'].squeeze().astype("int0"),
                                                    test_size=0.3)
    nn = NeuralNetwork([trainX.shape[1], 30, 10, 10])
    nn.train(trainX, trainY, 60000, learning_rate=.1)
    predictions = nn.predict(testX)
    print(accuracy_score(testY, predictions))


if __name__ == '__main__':
    main()
