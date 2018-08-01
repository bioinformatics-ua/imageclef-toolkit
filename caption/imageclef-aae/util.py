"""Additional utility functions."""

import tensorflow as tf
from tensorflow.contrib import gan as tfgan
from rel_loss import relativistic_average_discriminator_loss, relativistic_average_generator_loss


def gan_loss_by_name(gan_model: tfgan.GANModel, name: str, add_summaries=True):
    if name == "WASSERSTEIN":
        return tfgan.gan_loss(
            gan_model,
            generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
            discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
            gradient_penalty_weight=10.0,
            add_summaries=add_summaries
        )
    elif name == "RELATIVISTIC_AVG":
        return tfgan.gan_loss(
            gan_model,
            generator_loss_fn=relativistic_average_generator_loss,
            discriminator_loss_fn=relativistic_average_discriminator_loss,
            add_summaries=add_summaries
        )
    return tfgan.gan_loss(
        gan_model,
        generator_loss_fn=tfgan.losses.modified_generator_loss,
        discriminator_loss_fn=tfgan.losses.modified_discriminator_loss,
        add_summaries=add_summaries
    )


def assert_is_image(data):
    data.shape.assert_has_rank(4)
    data.shape[1:].assert_is_fully_defined()


def image_grid_summary(incoming, grid_size=4, name='data') -> tf.Tensor:
    """Creates and returns an image summary for a batch of images."""
    assert_is_image(incoming)
    num_images = grid_size ** 2
    image_shape = incoming.shape.as_list()[1:3]
    channels = incoming.shape.as_list()[3]

    return tf.summary.image(
        name,
        tfgan.eval.eval_utils.image_grid(
            incoming[:num_images],
            grid_shape=(grid_size, grid_size),
            image_shape=image_shape,
            num_channels=channels),
        max_outputs=1)


def image_summaries_2level_generated(model: tfgan.GANModel, grid_size=4):
    """Creates and returns image summaries for fake images only.

    Args:
        model: A GANModel tuple.
        grid_size: The size of an image grid.

    Raises:
        ValueError: If generated data aren't images.
    """
    generated_data = model.generated_data
    if isinstance(generated_data, tuple):
        generated_data = generated_data[0]
    if isinstance(generated_data, list):
        generated_data = generated_data[0]

    return image_grid_summary(generated_data, grid_size=grid_size, name='generated_data')


def image_summaries_generated(model: tfgan.GANModel, grid_size=4):
    """Creates and returns image summaries for fake images only.

    Args:
        model: A GANModel tuple.
        grid_size: The size of an image grid.

    Raises:
        ValueError: If generated data aren't images.
    """
    return image_summaries_2level_generated(model, grid_size=grid_size)


def image_summaries_2level(model: tfgan.GANModel, grid_size=4) -> (tf.Tensor, tf.Tensor):
    """Creates and returns image summaries for real and fake images.

    Args:
        model: A GANModel tuple.
        grid_size: The size of an image grid.

    Raises:
        ValueError: If real and generated data aren't images.
    """
    real_data = model.real_data
    if isinstance(real_data, tuple):
        real_data = real_data[0]
    generated_data = model.generated_data
    if isinstance(generated_data, tuple):
        generated_data = generated_data[0]
    if isinstance(generated_data, list):
        generated_data = generated_data[0]

    return (
        image_grid_summary(real_data, grid_size=grid_size, name='real_data'),
        image_grid_summary(
            generated_data, grid_size=grid_size, name='generated_data')
    )


def random_hypersphere(shape, name="random_hypersphere") -> tf.Tensor:
    "Generate a tensor of random noise in a hypersphere surface."
    assert len(shape) == 2
    with tf.name_scope(name):
        normal = tf.random_normal(shape)
        return tf.nn.l2_normalize(normal, axis=1)


def random_noise(shape, format, name="random_noise") -> tf.Tensor:
    assert len(shape) == 2
    if format == "SPHERE":
        return random_hypersphere(shape, name=name)
    elif format == "UNIFORM":
        return tf.random_uniform(shape, minval=-1.0, maxval=1.0, name=name)
    elif format == "NORMAL":
        return tf.random_normal(shape, mean=0.0, stddev=1.0, name=name)


def minibatch_stddev(incoming: tf.Tensor, name="minibatch_stddev") -> tf.Tensor:
    """Perform minibatch standard deviation normalization,
       as explained in http://arxiv.org/abs/1710.10196
    Args:
        incoming: Tensor of rank 4, channel-last (NHWC)
    Return:
        Tensor of rank 4 (w 1 extra element in the channels axis)
    """
    input_shape = incoming.shape.as_list()
    assert len(input_shape) == 4
    with tf.name_scope(name):
        _mean, variance = tf.nn.moments(incoming, axes=[0, 3])
        avg_stddev = tf.reduce_mean(variance)
        avg_stddev = tf.reshape(avg_stddev, [1, 1, 1, 1])
        avg_stddev = tf.tile(avg_stddev, [tf.shape(
            incoming)[0]] + input_shape[1:3] + [1])
        out = tf.concat([incoming, avg_stddev], axis=3)
        assert out.shape.as_list()[:3] == input_shape[:3]
        assert out.shape.as_list()[3] == input_shape[3] + 1
        return out

def unit_norm_loss(incoming: tf.Tensor, weight=1e-3, name="z_reg") -> tf.Tensor:
    "A loss that enforces the given data to have unit norm"
    assert len(incoming.shape.as_list()) == 2
    return tf.multiply(tf.abs(tf.nn.l2_loss(incoming) - 1.0), weight, name="z_reg")


def add_unit_norm_loss(incoming: tf.Tensor, weight=1e-3, add_summary=False, name="z_reg"):
    "A loss that enforces the given data to have unit norm"
    activation_reg = unit_norm_loss(incoming, weight, name=name)
    tf.add_to_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES, activation_reg)
    if add_summary:
        tf.summary.scalar('z_reg_loss', activation_reg)


def drift_loss(incoming, factor=1e-3, weights=1.0, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE, scope=None):
    with tf.name_scope(scope, "drift_loss") as sc:
        _, var_d = tf.nn.moments(incoming, axes=[0, 1])
        l = factor * var_d
        return tf.losses.compute_weighted_loss(
            l, weights=weights, scope=sc, loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES,
            reduction=reduction)


def add_drift_regularizer(discriminator_out, factor=1e-3, add_summary=False):
    drift = drift_loss(discriminator_out, factor=factor)
    if add_summary:
        tf.summary.scalar('discriminator_drift', drift)
    return drift


def upsample_2d(incoming, name="upsample"):
    input_shape = incoming.shape.as_list()
    nh = input_shape[1] * 2
    nw = input_shape[2] * 2
    return tf.image.resize_nearest_neighbor(incoming, (nh, nw), name=name)


def pixelwise_feature_vector_norm(incoming: tf.Tensor, epsilon=1e-11, name="lrn") -> tf.Tensor:
    """Perform a variant of local response normalization,
       called pixelwise feature vector normalization,
       as explained in http://arxiv.org/abs/1710.10196 (section 4.2)
    Args:
        incoming: Tensor of rank 4, channel-last (NHWC)
        epsilon: small constant for numerical stability
    Return:
        Tensor of same shape
    """
    with tf.name_scope(name):
        norms = tf.reduce_mean(tf.square(incoming), axis=3, keepdims=True)
        return incoming / (tf.sqrt(norms) + epsilon)


def spectral_norm(w, iteration=1):
    """Spectral normalization.
    Adapted from: https://github.com/taki0112/Spectral_Normalization-Tensorflow
    """
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable(
        "u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for _i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


@tf.contrib.framework.add_arg_scope
def conv2d_sn(inputs,
              num_outputs,
              kernel_size,
              stride=1,
              activation_fn=tf.nn.leaky_relu,
              normalizer_fn=None,
              normalizer_params=None,
              weights_initializer=tf.random_normal_initializer(stddev=0.02),
              weights_regularizer=None,
              biases_initializer=tf.zeros_initializer(),
              biases_regularizer=None,
              reuse=None,
              variables_collections=None,
              outputs_collections=None,
              trainable=True,
              scope=None):

    with tf.variable_scope(scope, default_name="conv", reuse=reuse):
        w = tf.get_variable(
            "weights",
            shape=[kernel_size, kernel_size, inputs.get_shape()[-1],
                   num_outputs],
            initializer=weights_initializer,
            regularizer=weights_regularizer,
            trainable=trainable)
        if variables_collections:
            tf.add_to_collections(variables_collections, w)
        net = tf.nn.conv2d(input=inputs, filter=spectral_norm(w),
                           strides=[1, stride, stride, 1], padding='SAME')

        if biases_initializer is not None:
            b = tf.get_variable(
                "biases", [num_outputs],
                initializer=biases_initializer,
                regularizer=biases_regularizer,
                trainable=trainable)
            if variables_collections:
                tf.add_to_collections(variables_collections, b)
            net = tf.nn.bias_add(net, b)

        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            net = normalizer_fn(net, **normalizer_params)

        if activation_fn:
            net = activation_fn(net)

        if outputs_collections:
            tf.add_to_collections(outputs_collections, net)

    return net


@tf.contrib.framework.add_arg_scope
def conv2d_t_sn(inputs,
              num_outputs,
              kernel_size,
              stride=1,
              activation_fn=tf.nn.leaky_relu,
              normalizer_fn=None,
              normalizer_params=None,
              weights_initializer=tf.random_normal_initializer(stddev=0.02),
              weights_regularizer=None,
              biases_initializer=tf.zeros_initializer(),
              biases_regularizer=None,
              reuse=None,
              variables_collections=None,
              outputs_collections=None,
              trainable=True,
              scope=None):

    with tf.variable_scope(scope, default_name="dconv", reuse=reuse):
        x_shape = inputs.get_shape().as_list()
        output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, num_outputs]

        w = tf.get_variable(
            "weights",
            shape=[kernel_size, kernel_size, inputs.get_shape()[-1],
                   num_outputs],
            initializer=weights_initializer,
            regularizer=weights_regularizer)
        if variables_collections:
            tf.add_to_collections(variables_collections, w)
        
        net = tf.nn.conv2d_transpose(
            inputs, filter=spectral_norm(w), output_shape=output_shape,
            strides=[1, stride, stride, 1], padding='SAME')

        if biases_initializer is not None:
            b = tf.get_variable(
                "biases", [num_outputs],
                initializer=biases_initializer, regularizer=biases_regularizer)
            if variables_collections:
                tf.add_to_collections(variables_collections, b)
            net = tf.nn.bias_add(net, b)

        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            net = normalizer_fn(net, **normalizer_params)

        if activation_fn:
            net = activation_fn(net)

        if outputs_collections:
            tf.add_to_collections(outputs_collections, net)

    return net