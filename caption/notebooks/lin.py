"""Utility module containing functions for concept detection through logistic regression."""

import tensorflow as tf
from tensorflow.contrib.estimator import add_metrics, linear_logit_fn_builder, multi_label_head, binary_classification_head
from util import f1, Datasets
import numpy as np
import matplotlib.pyplot as plt

def create_input_fn(features, target, num_epochs=1, batch_size=32, shuffle=False):
    return tf.estimator.inputs.numpy_input_fn(
        x={'x': features},
        y=target,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle,
        queue_capacity=512,
        num_threads=1
    )


def build_model_fn(n_classes, x_shape, learning_rate=0.1, optimizer=None, thresholds=None):
    """Build TensorFlow model descriptor. This makes a logistic regression model with a
    multi-label head and multiple predefined operating point thresholds.
    """
    column_x = tf.feature_column.numeric_column(
        'x',
        shape=x_shape,
        dtype=tf.float32,
        normalizer_fn=None
    )
    optimizer = optimizer or tf.train.FtrlOptimizer(
        learning_rate=learning_rate,
        learning_rate_power=-0.5,
        l1_regularization_strength=0.005,
        l2_regularization_strength=0.0,
    )
    logit_fn = linear_logit_fn_builder(
        n_classes, feature_columns=[column_x])

    thresholds = thresholds or [0.5]
    if n_classes == 1:
        head = binary_classification_head(
            weight_column=None,
            thresholds=thresholds
        )
    else:
        head = multi_label_head(
            n_classes=n_classes,
            thresholds=thresholds
        )
    def _train_op_fn(loss):
        return optimizer.minimize(
            loss,
            global_step=tf.train.get_global_step())
    
    def _model_fn(features, labels, mode, config):
        logits = logit_fn(features=features)
        
        return head.create_estimator_spec(
            features=features,
            mode=mode,
            logits=logits,
            labels=labels,
            train_op_fn=_train_op_fn)
    
    return _model_fn


def predict(estimator, idlist, x, threshold, concepts_to_train, checkpoint_path=None):
    """Build a prediction in the form of a list of ids with concepts.
    Args:
      estimator: tf.Estimator
      idlist: list
      x: ndarray
      threshold: float
      concepts_to_train: list
      checkpoint_path: str
    Return: iterable
    """
    assert x.shape[0] == len(idlist)
    pred = estimator.predict(create_input_fn(x, None, num_epochs=1, batch_size=100),
        predict_keys=None,
        checkpoint_path=checkpoint_path
    )
    pred = [p['probabilities'] for p in pred]
    assert len(pred) == len(idlist)
    return (
        (
            fileid,
            [c for (c, v) in zip(concepts_to_train, multi) if v >= threshold]
        ) for (fileid, multi) in zip(idlist, pred))


class TrainBundle:
    def __init__(self):
        self.global_steps = []
        self.all_metrics = []
        self.f1_scores_y = []
        self.aucs = []


def build_train_and_eval_function(estimator: tf.estimator.Estimator, bundle: TrainBundle, dset: Datasets,
                                  thresholds: list, concepts_to_train: list, epochs_per_eval: int = 1):
    def train_and_eval(num_epochs=1):
        best_f1 = 0.0
        test_predictions = None
        for _i in range(0, num_epochs, epochs_per_eval):
            estimator.train(
                create_input_fn(dset.train_x, dset.train_y, batch_size=64, num_epochs=epochs_per_eval, shuffle=True),
                hooks=None,
                steps=None,
                max_steps=None,
                saving_listeners=None)
            e = estimator.evaluate(
                create_input_fn(dset.val_x, dset.val_y)
            )
            e = add_f1_score_metrics(e, thresholds)
            bundle.global_steps.append(e['global_step'])
            bundle.all_metrics.append(e)
            f1_scores = [e['f1/positive_threshold_{}'.format(t)] for t in thresholds]
            cbi = np.argmax(f1_scores)
            current_best_f1 = f1_scores[cbi]
            print('F1 scores: {} (best: {})'.format(f1_scores, current_best_f1))
            if current_best_f1 > best_f1:
                best_f1 = current_best_f1
                best_threshold = thresholds[cbi]
                test_predictions = predict(estimator, dset.test_ids, dset.test_x, best_threshold, concepts_to_train)
            bundle.f1_scores_y.append(f1_scores)
            if bundle.aucs is not None:
                bundle.aucs.append(e['auc'])
        return (best_f1, test_predictions)
    return train_and_eval


def get_config(name):
    # a run configuration that saves stuff properly
    return tf.estimator.RunConfig(
        log_step_count_steps=1000,
        save_summary_steps=None)


def add_f1_score_metrics(metrics, thresholds):
    print("Building some other metrics!")
    for threshold in thresholds:
        precision = metrics["precision/positive_threshold_{}".format(threshold)]
        recall = metrics["recall/positive_threshold_{}".format(threshold)]
        metrics["f1/positive_threshold_{}".format(threshold)] = f1(precision, recall)
    return metrics


def show_eval(bundle: TrainBundle, thresholds, name="model"):
    "Show the validation curves and save them to a file."
    x = np.broadcast_to(np.expand_dims(range(1, len(bundle.global_steps) + 1), axis=1), (len(bundle.global_steps), len(thresholds)))
    y = np.array(bundle.f1_scores_y)
    labels = ['f1 @ {}'.format(t) for t in thresholds]
    plt.plot(x, y)
    plt.legend(labels, loc='best')
    plt.savefig('outputs/linear-estimator-{}.svg'.format(name))
    plt.show()
