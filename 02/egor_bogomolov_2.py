from scipy.spatial import distance
from random import randint, seed

from datetime import datetime
import numpy as np
import cv2

filename = 'apple.png'
filename_out = 'apple-16.png'
clusters = 16


def read_image(path):
    print("Reading image from {0}".format(path))
    image = cv2.imread(path)
    return image


def convert_image_to_array(image):
    width, height = image.shape[:2]
    return np.array([np.array(image[i][j]).astype("int32") for i in range(width) for j in range(height)])


def write_image(path, image):
    print("Writing image to {0}".format(path))
    cv2.imwrite(path, image)


def compress_objects(X):
    print("Compressing equal objects")
    values, counts = np.unique(X, return_counts=True, axis=0)
    return values, counts


def find_closest_in_set(obj, set_of_objects, distance_metric):
    distances_to_objects = map(lambda x: distance_metric(x, obj), set_of_objects)
    ind = np.argmin(distances_to_objects)
    return distances_to_objects[ind], ind


def initial_centroids(X, values, n_clusters, n_samples, distance_metric):
    print("Picked random centroid #1")
    centroids = [X[randint(0, n_samples)]]
    for i in range(n_clusters - 1):
        print("Looking for centroid #{0}".format(i + 2))
        modified_values = map(lambda obj: find_closest_in_set(obj, centroids, distance_metric)[0], values)
        ind = np.argmax(modified_values)
        centroids.append(values[ind])
    print("Finished picking centroids")
    return centroids


def k_means(X, n_clusters, distance_metric=distance.euclidean):
    print("Starting k-means algorithm...")
    start = datetime.now()
    n_samples = X.shape[0]
    n_features = X.shape[1]
    values, counts = compress_objects(X)
    centroids = initial_centroids(X, values, n_clusters, n_samples, distance_metric)
    iteration = 1
    while True:
        print("Iteration #{0}".format(iteration))
        new_centroid_sum = np.zeros([n_clusters, n_features])
        new_centroid_cnt = np.zeros(n_clusters)
        for value, count in zip(values, counts):
            dist, closest_id = find_closest_in_set(value, centroids, distance_metric)
            new_centroid_sum[closest_id] += count * value
            new_centroid_cnt[closest_id] += count
        max_diff = 0
        print("Diff in centroids:")
        for i in range(n_clusters):
            old_centroid = centroids[i]
            new_centroid = new_centroid_sum[i] / new_centroid_cnt[i]
            difference = old_centroid - new_centroid
            max_diff = max(max_diff, max(np.absolute(difference)))
            print(difference)
            centroids[i] = new_centroid
        if max_diff == 0:
            break
        print(centroids)
        iteration += 1
    labels = []
    print("Computing labels for all objects")
    for x in X:
        dist, closest_id = find_closest_in_set(x, centroids, distance_metric)
        labels.append(closest_id)
    finish = datetime.now()
    print("K-means finished in {0} sec".format((finish - start).total_seconds()))
    return labels, centroids


def create_histogram(labels):
    n_classes = max(labels) + 1
    total = len(labels)
    print("Creating histogram, {0} classes, {1} items".format(n_classes, total))
    hist = [0 for _ in range(n_classes)]
    for label in labels:
        hist[label] += 1
    for i in range(n_classes):
        hist[i] /= float(total)
    print("Finished histogram")
    return hist


def plot_colors(hist, centroids):
    print("Creating plot")
    w = 500
    h = 500
    start_x = 0
    bar = np.zeros((h, w, 3), np.uint8)
    for percent, color in zip(hist, centroids):
        end_x = start_x + percent * w
        cv2.rectangle(bar, (int(start_x), 0), (int(end_x), h), color.astype("uint8").tolist(),  -1)
        start_x = end_x
    print("Finished plot")
    return bar


def display_image(img, caption):
    cv2.namedWindow(caption, cv2.WINDOW_NORMAL)
    cv2.imshow(caption, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def recolor_image(image, labels, centroids):
    width, height = image.shape[:2]
    indexes = [(i, j) for i in range(width) for j in range(height)]
    for (i, j), label in zip(indexes, labels):
        image[i][j] = centroids[label].astype("uint8").tolist()


def recolor(image, n_colors):
    print("Recoloring image")
    image_array = convert_image_to_array(image)
    labels, centroids = k_means(image_array, n_colors)
    hist = create_histogram(labels)
    bar = plot_colors(hist, centroids)
    display_image(bar, 'colors')
    recolor_image(image, labels, centroids)
    return image


def main():
    seed(13)
    image = read_image(filename)
    image = recolor(image, clusters)
    write_image(filename_out, image)


if __name__ == '__main__':
    main()
