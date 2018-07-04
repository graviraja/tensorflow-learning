import tensorflow as tf
import pandas as pd
import numpy as np

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
LABEL_NAME = 'Species'


def read_from_file(filename):
    df = pd.read_csv(filename, names=CSV_COLUMN_NAMES, header=0)
    data, label = df, df.pop(LABEL_NAME)
    return data, label


def load_data():
    train_filename = 'data/iris_training.csv'
    test_filename = 'data/iris_test.csv'
    train_data, train_label = read_from_file(train_filename)
    test_data, test_label = read_from_file(test_filename)
    return train_data, train_label, test_data, test_label


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset


def eval_input_fn(features, labels, batch_size):
    features = dict(features)
    if labels is not None:
        inputs = (features, labels)
    else:
        inputs = features
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
    return dataset
