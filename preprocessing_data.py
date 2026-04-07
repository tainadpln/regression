import pandas as pd
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

train_images = idx2numpy.convert_from_file('C:\\Users\\ataii\\regression\\data\\train-images.idx3-ubyte')
train_labels = idx2numpy.convert_from_file('C:\\Users\\ataii\\regression\\data\\train-labels.idx1-ubyte')
train_data = (train_images, train_labels)

test_images = idx2numpy.convert_from_file('C:\\Users\\ataii\\regression\\data\\t10k-images.idx3-ubyte')
test_labels = idx2numpy.convert_from_file('C:\\Users\\ataii\\regression\\data\\t10k-labels.idx1-ubyte')
test_data = (test_images, test_labels)

def split_data(data, labels):
    data_images = data[0][data[1] == labels]
    data_labels = data[1][data[1] == labels]
    return data_images, data_labels

def prepare_data(train_data, test_data, digits=range(10)):
    train_results = [split_data(train_data, labels) for labels in digits]
    test_results = [split_data(test_data, labels) for labels in digits]

    train_images = np.concatenate([result[0] for result in train_results])
    train_labels = np.concatenate([result[1] for result in train_results])

    test_images = np.concatenate([result[0] for result in test_results])
    test_labels = np.concatenate([result[1] for result in test_results])

    train_images = train_images.reshape(train_images.shape[0], -1).astype(np.float64)
    train_images /= 255.0
    test_images = test_images.reshape(test_images.shape[0], -1).astype(np.float64)
    test_images /= 255.0

    return train_images, train_labels, test_images, test_labels
