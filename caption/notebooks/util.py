"""Utility module for the ImageCLEF concept detection notebooks."""

from math import ceil
import csv
import json
from os import listdir
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import tensorflow as tf
from tensorflow.contrib.estimator import add_metrics, linear_logit_fn_builder, multi_label_head, binary_classification_head
import h5py as h5
import sklearn
from sklearn.decomposition import PCA


def build_labels(labels_file: str, concept_map: dict) -> dict:
    """
    Return: dict <str, list<int>>
      maps an image ID to a list of concept uid indices (integers!) 
    """
    # read labels_file as csv
    # Format: «file-index»\t«concepts»
    #
    # concepts can be separated by either commas or semi-colons
    images = {}
    with open(labels_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            imageid = row[0]
            these_labels = filter(
                lambda x: x, map(
                    str.strip, row[1].replace(';', ',').split(',')))
            these_labels = filter(lambda x: x in concept_map, these_labels)
            label_indices = [concept_map[lbl] for lbl in these_labels]
            images[imageid] = label_indices
    return images


def build_features(features_file: str) -> np.ndarray:
    with h5.File(features_file, mode='r') as f:
        features = np.array(f['data'])
    return features


def build_features_with_ids(features_file: str) -> tuple:
    with h5.File(features_file, mode='r') as f:
        features = np.array(f['data']).astype(np.float32)
        ids = np.array(f['id'])
    return (ids, features)


def build_target_labels(nsamples, fids_list, label_voc, concepts, offset) -> lil_matrix:
    """Create a sparse matrix of all labels in a data set portion.
    Args:
      nsamples  : int the number of data points
      fids_list : str or list, path to the file containing the IDs which belong
                  in this data set portion, OR a list of strings containing the IDs
      label_voc : dict <str, list<int>> maps file IDs to their list of concept indices 
      concepts  : list <str> sequence of concepts to consider in classification
      offset  : int offset in number of concepts (as already assumed in `concepts`)
    Returns: sparse.lil_matrix
    """
    if isinstance(concepts, str):
        concepts = [concepts]
    if isinstance(fids_list, str):
        fids_list = [f.strip() for f in open(fids_list, encoding="utf-8")]
        
    y = lil_matrix((nsamples, len(concepts)), dtype=bool)
    for (i, l) in enumerate(filter(lambda x: x, fids_list)):
        fid = l.strip()
        if fid in label_voc:
            if fid in label_voc:
                for j in label_voc[fid]:
                    target_id = j - offset
                    if target_id >= 0 and target_id < len(concepts):
                        y[i, target_id] = True
    return y


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def print_predictions(test_predictions, results, filename=None, key=None):
    """Print test predictions to a submission file, and the results in a separate JSON file.
    Args:
     test_predictions: iterable of pairs (id, concept_list)
     results: arbitrary data to output as JSON
     filename: override main output file name
     key: a key to build the file name in the format "«key»-«timestamp».csv"
          if `filename` is `None`
    """
    if not filename:
        timestamp = time.strftime("%Y-%m-%d_%H%M", time.gmtime())
        if key:
            filename = "outputs/{}-{}.csv".format(key, timestamp)
        else:
            filename = "outputs/" + timestamp + ".csv"

    with open(filename, mode='w') as f:
        for (fid, concepts) in test_predictions:
            line = '{}\t{}\n'.format(fid, ';'.join(concepts))
            f.write(line)
    # also write a log of the outcomes associated to the file as json
    log_filename = filename[:-4] + '.json'
    with open(log_filename, mode='w') as f:
        f.write(json.dumps(results, cls=NumpyEncoder))
    print("Saved:", filename)


def filter_any_concepts(val_x, val_y) -> (np.ndarray, lil_matrix):
    """Return the data points with at least one positive label."""
    n_validation = val_x.shape[0]
    zero_entries = np.array([y.count_nonzero() == 0 for y in val_y])
    n_zero_entries = np.sum(zero_entries)
    print("{} items in validation set without concepts ({:.4}% of validation set)".format(
        n_zero_entries, n_zero_entries * 100.0 / n_validation))
    print("Continuing with {} validation points".format(n_validation - n_zero_entries))
    return val_x[~zero_entries], val_y[~zero_entries]


def f1(precision, recall):
    if precision + recall == 0:
        # clear NaNs
        return 0
    return 2 * precision * recall / (precision + recall)


def max_normalize(bocs: np.ndarray) -> np.ndarray:
    """Linearly normalize the bags so that the maximum of each bag is 1."""
    return bocs / np.max(bocs + 1e-10, axis=1, keepdims=True)


def tf_idf_normalize(bocs: np.ndarray) -> np.ndarray:
    """tf-idf normalization."""
    tf = bocs / np.sum(1e-10 + bocs, axis=1, keepdims=True)
    dcount = np.sum(bocs.astype(np.bool).astype(np.float), axis=0)
    idf = np.log(len(bocs) / dcount)
    return tf * idf


def power_normalize(bocs: np.ndarray) -> np.ndarray:
    """Power-law and L1 vector normalization."""
    # element-wise square root, then L1 normalization
    o = np.sqrt(bocs)
    o /= np.sum(o, axis=1, keepdims=True)
    return o


class Datasets:
    def __init__(self, train_ids, train_x, train_y, val_x, val_y, test_ids, test_x):
        self.train_ids = train_ids
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.test_ids = test_ids
        self.test_x = test_x
        
    @property
    def d(self):
        return self.train_x.shape[1]

    @staticmethod
    def from_h5_files(train_h5, val_h5, test_h5, labels_train, labels_val, concepts_to_train, offset=0, normalizer_fn=None):
        train_ids, train_x = build_features_with_ids(train_h5)
        train_y = build_target_labels(
            train_x.shape[0],
            train_ids,
            labels_train,
            concepts_to_train,
            offset
        )
        assert train_x.shape[0] == len(train_ids)
        assert train_x.shape[0] == train_y.shape[0]

        val_ids, val_x = build_features_with_ids(val_h5)
        val_y = build_target_labels(
            val_x.shape[0],
            val_ids,
            labels_val,
            concepts_to_train,
            offset
        )
        assert val_x.shape[0] == len(val_ids)
        assert val_x.shape[0] == val_y.shape[0]

        val_x, val_y = filter_any_concepts(val_x, val_y)

        train_y = train_y.toarray().astype(np.float32)
        val_y = val_y.toarray().astype(np.float32)

        test_ids, test_x = build_features_with_ids(test_h5)

        assert test_x.shape[0] == len(test_ids)
        
        assert train_x.shape[1] == val_x.shape[1]
        assert train_x.shape[1] == test_x.shape[1]

        if normalizer_fn is not None:
            assert callable(normalizer_fn)
            train_x = normalizer_fn(train_x)
            val_x = normalizer_fn(val_x)
            test_x = normalizer_fn(test_x)
        
        return Datasets(train_ids, train_x, train_y, val_x, val_y, test_ids, test_x)
    
    @staticmethod
    def from_h5_files_partition(train_h5, train_indices, test_h5, labels_all, concepts_to_train, offset=0, normalizer_fn=None):
        all_ids, all_x = build_features_with_ids(train_h5)
        
        all_y = build_target_labels(
            all_x.shape[0],
            all_ids,
            labels_all,
            concepts_to_train,
            offset
        )
        
        train_ids = all_ids[train_indices]
        train_x = all_x[train_indices]
        train_y = all_y[train_indices]
        assert train_x.shape[0] == len(train_ids)
        assert train_x.shape[0] == train_y.shape[0]

        val_x = all_x[~train_indices]
        val_y = all_y[~train_indices]
        assert val_x.shape[0] == val_y.shape[0]

        val_x, val_y = filter_any_concepts(val_x, val_y)

        train_y = train_y.toarray().astype(np.float32)
        val_y = val_y.toarray().astype(np.float32)

        test_ids, test_x = build_features_with_ids(test_h5)

        assert test_x.shape[0] == len(test_ids)
        assert train_x.shape[1] == val_x.shape[1]
        assert train_x.shape[1] == test_x.shape[1]

        if normalizer_fn is not None:
            assert callable(normalizer_fn)
            train_x = normalizer_fn(train_x)
            val_x = normalizer_fn(val_x)
            test_x = normalizer_fn(test_x)
        
        return Datasets(train_ids, train_x, train_y, val_x, val_y, test_ids, test_x)
    
    @staticmethod
    def from_pair_files_partition(train_h5, train_list_file, train_indices, test_h5, test_list_file,
                                  labels_all, concepts_to_train, offset=0, normalizer_fn=None):
        all_x = build_features(train_h5)
        all_ids = [x.strip() for x in open(train_list_file)]
        all_y = build_target_labels(
            all_x.shape[0],
            all_ids,
            labels_all,
            concepts_to_train,
            offset
        )

        train_x = all_x[train_indices]
        train_ids = all_ids[train_indices]
        train_y = all_y[train_indices]
        assert train_x.shape[0] == len(train_ids)
        assert train_x.shape[0] == train_y.shape[0]

        val_x = all_x[~train_indices]
        val_ids = all_ids[~train_indices]
        val_y = all_y[~train_indices]

        assert val_x.shape[0] == len(val_ids)
        assert val_x.shape[0] == val_y.shape[0]

        val_x, val_y = filter_any_concepts(val_x, val_y)

        train_y = train_y.toarray().astype(np.float32)
        val_y = val_y.toarray().astype(np.float32)

        test_x = build_features(test_h5)
        test_ids = [x.strip() for x in open(test_list_file)]
        assert test_x.shape[0] == len(test_ids)
        
        assert train_x.shape[1] == val_x.shape[1]
        assert train_x.shape[1] == test_x.shape[1]

        if normalizer_fn is not None:
            assert callable(normalizer_fn)
            train_x = normalizer_fn(train_x)
            val_x = normalizer_fn(val_x)
            test_x = normalizer_fn(test_x)

        return Datasets(train_ids, train_x, train_y, val_x, val_y, test_ids, test_x)

    
    @staticmethod
    def from_pair_files(train_h5, train_list_file, val_h5, val_list_file, test_h5, test_list_file,
                        labels_train, labels_val, concepts_to_train, offset=0, normalizer_fn=None):
        train_x = build_features(train_h5)
        train_ids = [x.strip() for x in open(train_list_file)]
        train_y = build_target_labels(
            train_x.shape[0],
            train_ids,
            labels_train,
            concepts_to_train,
            offset
        )
        assert train_x.shape[0] == len(train_ids)
        assert train_x.shape[0] == train_y.shape[0]

        val_x = build_features(val_h5)
        val_ids = [x.strip() for x in open(val_list_file)]
        val_y = build_target_labels(
            val_x.shape[0],
            val_ids,
            labels_val,
            concepts_to_train,
            offset
        )
        assert val_x.shape[0] == len(val_ids)
        assert val_x.shape[0] == val_y.shape[0]

        val_x, val_y = filter_any_concepts(val_x, val_y)

        train_y = train_y.toarray().astype(np.float32)
        val_y = val_y.toarray().astype(np.float32)

        test_x = build_features(test_h5)
        test_ids = [x.strip() for x in open(test_list_file)]
        assert test_x.shape[0] == len(test_ids)
        
        assert train_x.shape[1] == val_x.shape[1]
        assert train_x.shape[1] == test_x.shape[1]

        if normalizer_fn is not None:
            assert callable(normalizer_fn)
            train_x = normalizer_fn(train_x)
            val_x = normalizer_fn(val_x)
            test_x = normalizer_fn(test_x)

        return Datasets(train_ids, train_x, train_y, val_x, val_y, test_ids, test_x)
