"""Module for functions specific to a basic generative adversarial network.
"""

import tensorflow as tf
from tensorflow.contrib import gan as tfgan
from util import add_drift_regularizer, image_summaries_generated, image_grid_summary, minibatch_stddev
from ae import conv_block, conv_t_block, build_dcgan_encoder, build_dcgan_generator, build_encoder, build_1lvl_generator, code_autoencoder_mse_cosine
from AMSGrad import AMSGrad

batch_norm = tf.contrib.layers.batch_norm
conv2d = tf.contrib.layers.conv2d
conv2d_t = tf.contrib.layers.conv2d_transpose
fully_connected = tf.contrib.layers.fully_connected
variance_scaling_initializer = tf.contrib.layers.variance_scaling_initializer
arg_scope = tf.contrib.framework.arg_scope


def build_dcgan_discriminator(
        x: tf.Tensor, z: tf.Tensor,
        add_drift_loss: bool = True, batch_norm: bool = True,
        input_noise_factor: float = 0., hidden_noise_factor: float = 0.25,
        add_summaries: bool = False, mode=None):
    "Discriminator based on DCGAN, with a few extra bells and whistles."

    if batch_norm:
        norm_fn = tf.contrib.layers.batch_norm
        norm_params = {
            'center': True,
            'scale': True,
            'fused': True
        }
    else:
        norm_fn = None
        norm_params = None

    # add dropout noise
    x_noisy = tf.layers.dropout(x, rate=input_noise_factor, training=True)

    # collect all activations and trainable variables
    with arg_scope([fully_connected, conv2d], outputs_collections=[tf.GraphKeys.ACTIVATIONS],
                   variables_collections=[tf.GraphKeys.TRAINABLE_VARIABLES],
                   weights_initializer=tf.random_normal_initializer(stddev=0.02)):
        net = x_noisy
        # use batch norm and leaky ReLU on all body layers
        with arg_scope([conv2d], normalizer_fn=norm_fn,
                       normalizer_params=norm_params,
                       activation_fn=tf.nn.leaky_relu):
            # uncomment the line below for 64x64 resolution
            net = conv2d(net, 128, 5, 2)  # Bx32x32x128
            net = conv2d(net, 256, 5, 2)  # Bx16x16x256
            net = conv2d(net, 512, 5, 2)  # Bx8x8x512
            net = tf.layers.dropout(net, rate=hidden_noise_factor)
            net = conv2d(net, 1024, 5, 2)  # Bx4x4x1024
            net = tf.layers.dropout(net, rate=hidden_noise_factor)
        net = minibatch_stddev(net)  # add stddev to minibatch
        assert net.shape.as_list()[1:4] == [4, 4, 1025]
        net = tf.reshape(net, [-1, 4 * 4 * 1025])
        net = fully_connected(net, 1, activation_fn=None)
        if add_drift_loss:
            add_drift_regularizer(net, add_summary=add_summaries)
    return net


def build_discriminator_1lvl(
        x: tf.Tensor, z: tf.Tensor,
        add_drift_loss: bool = True, batch_norm: bool = True,
        input_noise_factor: float = 0., hidden_noise_factor: float = 0.25,
        add_summaries: bool = False, mode=None):
    "1-level discriminator"

    if batch_norm:
        norm_fn = tf.contrib.layers.batch_norm
        norm_params = {
            'center': True,
            'scale': True,
            'fused': True
        }
    else:
        norm_fn = None
        norm_params = None

    # add dropout noise
    x_noisy = tf.layers.dropout(x, rate=input_noise_factor, training=True)

    # collect all activations and trainable variables
    with arg_scope([fully_connected, conv2d], outputs_collections=[tf.GraphKeys.ACTIVATIONS],
                   variables_collections=[tf.GraphKeys.TRAINABLE_VARIABLES],
                   weights_initializer=tf.random_normal_initializer(stddev=0.02)):
        net = x_noisy
        # use batch norm and leaky ReLU on all body layers
        with arg_scope([conv2d], normalizer_fn=norm_fn,
                       normalizer_params=norm_params,
                       activation_fn=tf.nn.leaky_relu):
            net = conv_block(net, [32, 64])  # Bx32x32x64
            net = conv_block(
                net, [64, 128])  # Bx16x16x128
            net = conv_block(
                net, [128, 256], dropout=hidden_noise_factor)  # Bx8x8x256
            net = conv_block(
                net, [256, 512], dropout=hidden_noise_factor)  # Bx4x4x512
        net = minibatch_stddev(net)  # add stddev to minibatch
        net = tf.reshape(net, [-1, 4 * 4 * 513])
        net = fully_connected(net, 1, activation_fn=None)
        if add_drift_loss:
            add_drift_regularizer(net, add_summary=add_summaries)
    return net


def build_discriminator_2lvl(x, z, add_drift_loss=True, noise_factor=0.25, mode=None):
    "2-level discriminator"
    if isinstance(x, list):
        [x1, x2] = x
    else:
        # real data, resize second level input
        x1 = x
        x2 = tf.layers.average_pooling2d(x1, 4, 4)  # Bx16x16x3

    # add noise to inputs
    # x1_noisy = x1 + \
    #    tf.random_normal(tf.shape(x1), mean=0.0, stddev=0.5 * noise_factor)
    x1_noisy = tf.layers.dropout(x1, rate=0.5, training=True)
    # x2_noisy = x2 + \
    #    tf.random_normal(tf.shape(x2), mean=0.0, stddev=0.5 * noise_factor)
    x2_noisy = tf.layers.dropout(x2, rate=0.5, training=True)

    # collect all activations and trainable variables
    with arg_scope([fully_connected, conv2d], outputs_collections=[tf.GraphKeys.ACTIVATIONS],
                   variables_collections=[tf.GraphKeys.TRAINABLE_VARIABLES],
                   weights_initializer=tf.random_normal_initializer(stddev=0.02)):
        # use batch norm and leaky ReLU on all body layers
        with arg_scope([conv2d], normalizer_fn=batch_norm,
                       normalizer_params={'center': True,
                                          'scale': True, 'fused': True},
                       activation_fn=tf.nn.leaky_relu):
            net = conv_block(x1_noisy, [32, 64],
                             dropout=noise_factor)  # Bx32x32x64
            net = conv_block(
                net, [64, 128], dropout=noise_factor)  # Bx16x16x64

            lvl2_net = conv2d(x2_noisy, 64, 1, 1, scope="convh")
            # obtain partial features from both and concatenate them at the same level.
            net = tf.concat([net, lvl2_net], axis=3)  # Bx16x16x192

            net = conv_block(
                net, [128, 256], dropout=noise_factor)  # Bx8x8x256
            net = conv_block(
                net, [256, 512], dropout=noise_factor)  # Bx4x4x512

        net = minibatch_stddev(net)  # Bx4x4x513
        net = tf.reshape(net, [-1, 4 * 4 * 513])
        net = fully_connected(net, 1, activation_fn=None)
        if add_drift_loss:
            add_drift_regularizer(net)
    return net


def build_gan_harness(image_input: tf.Tensor,
                      noise: tf.Tensor,
                      generator,
                      discriminator,
                      noise_format: str = 'SPHERE',
                      adversarial_training: str = 'WASSERSTEIN',
                      no_trainer: bool = False,
                      summarize_activations: bool = False) -> tuple:
    image_size = image_input.shape.as_list()[1]
    noise_dim = noise.shape.as_list()[1]
    nchannels = image_input.shape.as_list()[3]
    print("Plain Generative Adversarial Network: {}x{} images".format(
        image_size, image_size))

    def _generator_fn(z):
        return generator(
            z, nchannels=nchannels, add_summaries=True, mode='TRAIN')

    def _discriminator_fn(x, z):
        return discriminator(
            x, z, add_drift_loss=True, batch_norm=True, mode='TRAIN')

    gan_model = tfgan.gan_model(
        _generator_fn, _discriminator_fn, image_input, noise,
        generator_scope='Generator',
        discriminator_scope='Discriminator',
        check_shapes=True)  # set to False for 2-level architectures

    sampled_x = gan_model.generated_data
    image_grid_summary(sampled_x, grid_size=4, name='generated_data')
    if summarize_activations:
        tf.contrib.layers.summarize_activations()
    tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    if adversarial_training == "WASSERSTEIN":
        gan_loss = tfgan.gan_loss(
            gan_model,
            generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
            discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
            gradient_penalty_weight=10.0,
            add_summaries=True
        )
    else:
        gan_loss = tfgan.gan_loss(
            gan_model,
            generator_loss_fn=tfgan.losses.modified_generator_loss,
            discriminator_loss_fn=tfgan.losses.modified_discriminator_loss,
            add_summaries=True
        )

    if no_trainer:
        train_ops = None
    else:
        train_ops = tfgan.gan_train_ops(gan_model, gan_loss,
                                        generator_optimizer=AMSGrad(
                                            1e-5, beta1=0.5, beta2=0.99),
                                        discriminator_optimizer=AMSGrad(
                                            1e-4, beta1=0.5, beta2=0.99),
                                        summarize_gradients=True)
    return (gan_model, gan_loss, train_ops)
