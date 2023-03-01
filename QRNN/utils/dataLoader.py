"""
Script used for data preparation. It consists of three parts:
1) general functions, used to split, shuffle and manipulate arrays for
    further use
2) generation of arrays of mathematical functions (e.g. sine, bessel
    and more)
3) preparing real world data (sunspots, hurricanes)

Additionally, can load MNIST
"""

import collections
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import signal
from scipy.special import jv, eval_legendre


# ----------------------------------------------------------------------
# general functions
# ----------------------------------------------------------------------

def shuffle_data(x, y):
    """
    Shuffle the data with its labels (do not mix data with labels)
    E.g. (x_i,y_i) -> (x_j, y_j)

    :param x: np.array, input to shuffle
    :param y: np.array, labels to shuffle
    :return: (np.array, np.array), tuple of shuffled (x,y)
    """
    shuffler = np.random.permutation(len(x))
    x = np.array(x)[shuffler.astype(int)]
    y = np.array(y)[shuffler.astype(int)]

    return x, y


def make_batches(data, batch_size):
    """
    Take data and output batched data

    :param data: np.array, pre-batched data
    :param batch_size: int, batch size
    :return: np.array, batched data
    """

    if batch_size is not None and batch_size != 1:
        data = np.reshape(data[:data.shape[0] // batch_size * batch_size],
                          (data.shape[0] // batch_size, -1, data.shape[1]))
        data = np.transpose(data, axes=(0, 2, 1))

    return data


def split_data(data, seq_in_len, seq_out_len, batch_size=None, shuffle=True):
    """
    Preparation of train and validation data set for given list of
    functions value

    :param data: np.array, data set to split
    :param seq_in_len: int, length of the input sequence
    :param seq_out_len: int, length of the output sequence
    :param batch_size: int, batch size for splitting the data into
        batches (works only for quantum circuit, classical do it on
        its own)
    :param shuffle: bool, shuffle the data points

    :return: tuple (np.array, np.array),(np.array, np.array), np.array,
        values for train (1) and validation (2) and data itself (3)
    """

    x_train, y_train = [], []

    test_frac = 5 / 10  # fraction of the train examples

    test_len = int((len(data) - seq_in_len) * test_frac)

    # 1) One way to produce train data for data series it to have
    # x.size = seq_in_len and y.size = seq_out_len;
    # then we calculate the loss only at the end of the processing sequence x
    # 2) The other way is to have
    # x.size = seq_in_len and y.size >= x.size
    # where we still calculate the loss at the end of processing sequence x,
    # but this approach is more general;
    # it can be used to easily calculate the loss of copying task

    for i in range(test_len):
        x_train.append(data[i:i + seq_in_len])
        y_train.append(data[i + 1: i + seq_in_len + seq_out_len])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # shuffle the data points
    if shuffle:
        x_train, y_train = shuffle_data(x_train, y_train)

    # produce batches -- magic, but it works, but only for quantum
    # circuit, for classical it splits automatically so no need
    x_train = make_batches(x_train, batch_size)
    y_train = make_batches(y_train, batch_size)

    # the same as above but for validation set
    x_val, y_val = [], []

    for i in range(test_len, len(data) - seq_in_len - seq_out_len):
        x_val.append(data[i:i + seq_in_len])
        y_val.append(data[i + 1: i + seq_in_len + seq_out_len])

    x_val, y_val = np.array(x_val), np.array(y_val)

    return (x_train, y_train), (x_val, y_val), data


def generate_function(function, length, resolution, noise):
    """
    Generate datapoints for timeseries prediction of specific length and
    density

    :param function: FunctionType, the function to produce points
    :param length: int, the max value for the data points
    :param resolution: int, number of points of f() for x in [0, length]
    :param noise: float, strength of the noise added to the data

    :return: np.array(), {f(x_1), ..., f(x_N)}, N = resolution,
        x_N = length
    """
    data = np.array(
        [function(x) + (random.random() - 1 / 2) * noise
         for x in np.linspace(0, length, resolution)])

    data -= np.min(data)
    data /= np.max(data)

    # rescale the date to range [a, b] for better performance with mean
    # photons it is better to keep it between [0,1]
    a = 0
    b = 1
    data += a / (b - a)
    data /= 1 / (b - a)

    return data


# ----------------------------------------------------------------------
# mathematical functions
# ----------------------------------------------------------------------

def triangle_wave(length, resolution, seq_in_len, seq_out_len,
                  noise=0.02, batch_size=1):
    def triangle_fn(x): return signal.sawtooth(x, 0.5)

    data = generate_function(triangle_fn, length, resolution, noise)

    return split_data(data, seq_in_len, seq_out_len, batch_size)


def sine_wave(length, resolution, seq_in_len, seq_out_len,
              noise=0.02, batch_size=1):
    data = generate_function(math.sin, length, resolution, noise)

    return split_data(data, seq_in_len, seq_out_len, batch_size)


def cos_wave_damped(length, resolution, seq_in_len, seq_out_len,
                    noise=0.02, batch_size=1):
    dumping = 0.1

    def cos_dump(x): return math.exp(- dumping * x) * math.cos(x)

    data = generate_function(cos_dump, length, resolution, noise)

    return split_data(data, seq_in_len, seq_out_len, batch_size)


def sine_wave_2(length, resolution, seq_in_len, seq_out_len,
                noise=0.02, batch_size=1):
    freq_1, freq_2 = 1, 2

    def sin2(x): return 0.5 * math.sin(freq_1 * x) + 0.5 * math.sin(freq_2 * x)

    data = generate_function(sin2, length, resolution, noise)

    return split_data(data, seq_in_len, seq_out_len, batch_size)


def bessel(length, resolution, seq_in_len, seq_out_len,
           noise=0.02, batch_size=1):
    degree = 0

    def bessel_fn(x): return jv(degree, x)

    data = generate_function(bessel_fn, length, resolution, noise)

    return split_data(data, seq_in_len, seq_out_len, batch_size)


def legandre(length, resolution, seq_in_len, seq_out_len,
             noise=0.02, batch_size=1):
    # legandre polynomials are orthogonal polynomials on the period [-1,1]
    def legandre_fn(x): return eval_legendre(3, 2 * x / length - 1)

    data = generate_function(legandre_fn, length, resolution, noise)

    return split_data(data, seq_in_len, seq_out_len, batch_size)


def legandre_4(length, resolution, seq_in_len, seq_out_len,
               noise=0.02, batch_size=1):
    # legandre polynomials are orthogonal polynomials on the period [-1,1]
    def legandre_4_fn(x): return eval_legendre(4, 2 * x / length - 1)

    data = generate_function(legandre_4_fn, length, resolution, noise)

    return split_data(data, seq_in_len, seq_out_len, batch_size)


def line(length, resolution, seq_in_len, seq_out_len,
         noise=0.2, batch_size=1):
    def line_fn(x): return 0.5 * x

    data = generate_function(line_fn, length, resolution, noise)

    return split_data(data, seq_in_len, seq_out_len, batch_size)


# ----------------------------------------------------------------------
# Script  for preparing MNIST dataset for use. Including down-sampling,
# selecting only 2 digits, splitting into training and validation and
# more.
#
# Mostly taken from https://www.tensorflow.org/quantum/tutorials/mnist
# ----------------------------------------------------------------------


def mnist_data(resolution, size, batch_size):
    # load the training data from the provided files
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[
        ..., np.newaxis] / 255.0

    def filter_num(x, y, num1=3, num2=6):
        keep = (y == num1) | (y == num2)
        x, y = x[keep], y[keep]
        y = y == num1
        return x, y

    x_train, y_train = filter_num(x_train, y_train)
    x_test, y_test = filter_num(x_test, y_test)

    x_train_small = tf.image.resize(x_train, (resolution, resolution)).numpy()
    x_test_small = tf.image.resize(x_test, (resolution, resolution)).numpy()

    def remove_contradicting(xs, ys):
        mapping = collections.defaultdict(set)
        orig_x = {}
        # Determine the set of labels for each unique image:
        for x, y in zip(xs, ys):
            orig_x[tuple(x.flatten())] = x
            mapping[tuple(x.flatten())].add(y)

        new_x = []
        new_y = []
        for flatten_x in mapping:
            x = orig_x[flatten_x]
            labels = mapping[flatten_x]
            if len(labels) == 1:
                new_x.append(x)
                new_y.append(next(iter(labels)))
            else:
                # Throw out images that match more than one label.
                pass

        return np.array(new_x), np.array(new_y)

    x_train_nocon, y_train_nocon = remove_contradicting(x_train_small, y_train)

    train_size = size * 8 // 10
    val_size = size - train_size

    # make sequence out of them
    x_train_nocon = x_train_nocon.reshape(
        -1, resolution ** 2)[:train_size].astype(np.float32)
    y_train_nocon = y_train_nocon[:train_size].astype(np.float32)

    x_test_small = x_test_small.reshape(
        -1, resolution ** 2)[:val_size].astype(np.float32)
    y_test = y_test[:val_size].astype(np.float32)

    # produce batches -- magic, but it works, but only for quantum
    # circuit, for classical it splits automatically so no need
    if batch_size is not None and batch_size != 1:
        x_train_nocon = np.reshape(
            x_train_nocon[:x_train_nocon.shape[0] // batch_size * batch_size],
            (x_train_nocon.shape[0] // batch_size, -1, x_train_nocon.shape[1]))
        x_train_nocon = np.transpose(x_train_nocon, axes=(0, 2, 1))

        y_train_nocon = np.reshape(
            y_train_nocon[:y_train_nocon.shape[0] // batch_size * batch_size],
            (y_train_nocon.shape[0] // batch_size, -1, 1))
        y_train_nocon = np.transpose(y_train_nocon, axes=(0, 2, 1))

    return (x_train_nocon, y_train_nocon), (x_test_small, y_test)
