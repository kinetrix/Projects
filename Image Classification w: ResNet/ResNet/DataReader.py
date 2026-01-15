import os
import pickle
import numpy as np

""" This script implements the functions for reading data.
"""

def load_data(data_dir):
    """ Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches are stored.
    
    Returns:
        x_train: An numpy array of shape [50000, 3072]. 
        (dtype=np.float32)
        y_train: An numpy array of shape [50000,]. 
        (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072]. 
        (dtype=np.float32)
        y_test: An numpy array of shape [10000,]. 
        (dtype=np.int32)
    """
    ### YOUR CODE HERE
    def unpickle(file):
        with open(file, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
        return data_dict

    x_train_batches = []
    y_train_batches = []

    # Load all 5 training batches
    for i in range(1, 6):
        file_path = os.path.join(data_dir, f'data_batch_{i}')
        data_batch = unpickle(file_path)
        x_train_batches.append(data_batch[b'data'])
        y_train_batches.append(data_batch[b'labels'])

    # Concatenate all training batches into single numpy arrays
    x_train = np.concatenate(x_train_batches)
    y_train = np.concatenate(y_train_batches)

    # Load the test batch
    test_file_path = os.path.join(data_dir, 'test_batch')
    test_batch = unpickle(test_file_path)
    x_test = test_batch[b'data']
    y_test = test_batch[b'labels']

    # Convert data to the required types
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.int32)
    x_test = x_test.astype(np.float32)
    y_test = np.array(y_test).astype(np.int32)
    ### YOUR CODE HERE

    return x_train, y_train, x_test, y_test

def train_vaild_split(x_train, y_train, split_index=45000):
    """ Split the original training data into a new training dataset
        and a validation dataset.
    
    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        split_index: An integer.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid