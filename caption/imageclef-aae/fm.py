"Feature matching module"
import tensorflow as tf
import numpy as np



# collection name for network activations subjected to feature matching
FEATURE_MATCH = "feature_match"

def feature_matching_loss(collection_name: str = FEATURE_MATCH, scope=None, weight=1., ord=2, name="feature_matching") -> tf.Tensor:
    """Calculate the feature matching loss between the tensors in the given
    feature matching collection.
    """
    features = tf.get_collection(collection_name, scope=scope)
    assert len(features) == 2, "expected 2 feature matching tensors, got {}".format(len(features))

    [f1, f2] = features
    f1 = tf.reduce_mean(f1, axis=0)
    f2 = tf.reduce_mean(f2, axis=0)
    if weight is not None and weight != 1.:
        return tf.multiply(tf.norm(f1 - f2, ord=ord), weight, name=name)
    return tf.norm(f1 - f2, ord=ord, name=name)


def f_encoder(incoming: tf.Tensor, channels_out: int = 128):
    """A simple encoder to be used in denoising feature matching."""
    shape = incoming.shape.as_list()
    if len(shape) > 2:
        incoming = tf.reshape(incoming, [-1, np.prod(shape[1:])])
    return tf.layers.dense(incoming, channels_out)

def f_decoder(incoming: tf.Tensor, dimensions_out):
    """A simple decoder to be used in denoising feature matching."""
    assert len(incoming.shape) == 2
    if isinstance(dimensions_out, int):
        channels_out = dimensions_out
        shape_out = [-1, dimensions_out]
    else:
        channels_out = np.prod(dimensions_out)
        shape_out = [-1, *dimensions_out]

    net = tf.layers.dense(incoming, channels_out)
    return tf.reshape(net, shape_out)


def corrupt_by_name(incoming: tf.Tensor, name: str, factor = 1.):
    """Corrupt the given activations.
    Args:
      incoming: the input tensor
      name: one of "normal", "dropout", or "dropout+normal"
    """
    return {
        "normal": lambda x: corrupt_normal(x, stddev=factor),
        "dropout": lambda x: tf.nn.dropout(x, keep_prob=(1 / (1 + factor * 0.5))),
        "dropout+normal": lambda x: tf.nn.dropout(corrupt_normal(x, stddev=factor), keep_prob=(1 / 1 + factor))
    }[name](incoming)


def corrupt_normal(incoming: tf.Tensor, stddev=1.):
    noise = tf.random_normal(tf.shape(incoming), mean=0.0, stddev=stddev)
    return incoming + noise


