import pandas as pd
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

train_images = idx2numpy.convert_from_file('C:\\Users\\ataii\\OneDrive\\Desktop\\regression\\data\\train-images.idx3-ubyte')
train_labels = idx2numpy.convert_from_file('C:\\Users\\ataii\\OneDrive\\Desktop\\regression\\data\\train-labels.idx1-ubyte')
train_data = (train_images, train_labels)
print(train_images.shape)

test_images = idx2numpy.convert_from_file('C:\\Users\\ataii\\OneDrive\\Desktop\\regression\\data\\t10k-images.idx3-ubyte')
test_labels = idx2numpy.convert_from_file('C:\\Users\\ataii\\OneDrive\\Desktop\\regression\\data\\t10k-labels.idx1-ubyte')
test_data = (test_images, test_labels)

def split_data(data, labels):
    data_images = data[0][data[1] == labels]
    data_labels = data[1][data[1] == labels]
    return data_images, data_labels

train_images_0, train_labels_0 = split_data(train_data, 0)
train_images_1, train_labels_1 = split_data(train_data, 1)
train_images_2, train_labels_2 = split_data(train_data, 2)
train_images_3, train_labels_3 = split_data(train_data, 3)
train_images_4, train_labels_4 = split_data(train_data, 4)
train_images_5, train_labels_5 = split_data(train_data, 5)
train_images_6, train_labels_6 = split_data(train_data, 6)
train_images_7, train_labels_7 = split_data(train_data, 7)
train_images_8, train_labels_8 = split_data(train_data, 8)
train_images_9, train_labels_9 = split_data(train_data, 9)

test_images_0, test_labels_0 = split_data(test_data, 0)
test_images_1, test_labels_1 = split_data(test_data, 1)
test_images_2, test_labels_2 = split_data(test_data, 2)
test_images_3, test_labels_3 = split_data(test_data, 3)
test_images_4, test_labels_4 = split_data(test_data, 4)
test_images_5, test_labels_5 = split_data(test_data, 5)
test_images_6, test_labels_6 = split_data(test_data, 6)
test_images_7, test_labels_7 = split_data(test_data, 7)
test_images_8, test_labels_8 = split_data(test_data, 8)
test_images_9, test_labels_9 = split_data(test_data, 9)

train_images = np.concatenate((train_images_0, train_images_1, train_images_2, train_images_3, train_images_4, train_images_5, train_images_6, train_images_7, train_images_8, train_images_9))
train_labels = np.concatenate((train_labels_0, train_labels_1, train_labels_2, train_labels_3, train_labels_4, train_labels_5, train_labels_6, train_labels_7, train_labels_8, train_labels_9))

test_images = np.concatenate((test_images_0, test_images_1, test_images_2, test_images_3, test_images_4, test_images_5, test_images_6, test_images_7, test_images_8, test_images_9))
test_labels = np.concatenate((test_labels_0, test_labels_1, test_labels_2, test_labels_3, test_labels_4, test_labels_5, test_labels_6, test_labels_7, test_labels_8, test_labels_9))

train_images = train_images.reshape(train_images.shape[0], -1).astype(np.float64)
train_images /= 255.0
test_images = test_images.reshape(test_images.shape[0], -1).astype(np.float64)
test_images /= 255.0
