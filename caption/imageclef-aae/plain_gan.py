"""Module for functions specific to a basic generative adversarial network.
"""

import tensorflow as tf
from tensorflow.contrib import gan as tfgan
from util import add_drift_regularizer, image_summaries_generated, image_grid_summary, minibatch_stddev, conv2d_sn, conv2d_t_sn
from ae import conv_block, conv_t_block, build_dcgan_encoder, build_dcgan_generator, build_encoder, build_1lvl_generator, code_autoencoder_mse_cosine
from AMSGrad import AMSGrad
from rel_loss import relativistic_average_discriminator_loss, relativistic_average_generator_loss
from fm import feature_matching_loss, FEATURE_MATCH

batch_norm = tf.contrib.layers.batch_norm
conv2d = tf.contrib.layers.conv2d
conv2d_t = tf.contrib.layers.conv2d_transpose
fully_connected = tf.contrib.layers.fully_connected
variance_scaling_initializer = tf.contrib.layers.variance_scaling_initializer
arg_scope = tf.contrib.framework.arg_scope


def build_dcgan_discriminator(
        x: tf.Tensor, z: tf.Tensor, nlevels: int = 4,
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
            nb = 1024 >> (nlevels - 1)
            for i in range(nlevels - 1):
                net = conv2d(net, nb, 5, 2)
                nb *= 2
                if i - 1 == nlevels // 2:
                    tf.add_to_collection(FEATURE_MATCH, net) # so it can be fetched for feature matching

            net = tf.layers.dropout(net, rate=hidden_noise_factor)
        net = minibatch_stddev(net)  # add stddev to minibatch
        assert net.shape.as_list()[1:4] == [4, 4, 1025]
        net = tf.reshape(net, [-1, 4 * 4 * 1025])
        net = fully_connected(net, 1, activation_fn=None)
        if add_drift_loss:
            add_drift_regularizer(net, add_summary=add_summaries)
    return net


def build_discriminator_1lvl(
        x: tf.Tensor, z: tf.Tensor, nlevels: int = 4,
        add_drift_loss: bool = True,
        batch_norm: bool = False,
        spectral_norm: bool = True,
        input_noise_factor: float = 0., hidden_noise_factor: float = 0.25,
        add_summaries: bool = False, mode=None):
    "1-level discriminator"
    assert (x.shape.as_list()[1] >> nlevels) == 4
    assert not (batch_norm is True and spectral_norm is True)

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
                   weights_initializer=tf.random_normal_initializer(
                       stddev=0.02),
                   biases_initializer=None):
        net = x_noisy
        # use batch norm and leaky ReLU on all body layers
        with arg_scope([conv2d], normalizer_fn=norm_fn,
                       normalizer_params=norm_params,
                       activation_fn=tf.nn.leaky_relu):
            nb = 512 >> nlevels
            for i in range(nlevels):
                nf = hidden_noise_factor if i == nlevels - 1 else None
                net = conv_block(net, [nb, nb * 2],
                                 spectral_norm=spectral_norm, dropout=nf)
                nb *= 2
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


def build_sndcgan_discriminator(x: tf.Tensor, z: tf.Tensor, nlevels: int = 4, conv_activation_fn=tf.nn.leaky_relu,
                                spectral_norm: bool = True, add_summaries=False, mode=None):
    """Build a discriminator based on SNDCGAN, presented in:
        Miyato et al. Spectral Normalization for generative adversarial networks (2018)

    Args:
      x : the real data [B, H, W, C]
      z : the prior code [B, N]
      nlevels : number of upsampling network levels (4 for 64x64, 5 for 128x128, etc.)
      conv_activation_fn : a function for the non-linearity operation after each hidden convolution layer
      spectral_norm : whether to use spectral normalization
      add_summaries : whether to produce summary operations
      mode : set to 'TRAIN' during the training process
    """
    if spectral_norm:
        conv2d = conv2d_sn
    assert len(x.shape.as_list()) == 4

    net = x
    with arg_scope([fully_connected, conv2d, conv2d_t], outputs_collections=[tf.GraphKeys.ACTIVATIONS],
                   variables_collections=[tf.GraphKeys.TRAINABLE_VARIABLES],
                   weights_initializer=tf.random_normal_initializer(stddev=0.02)):
        # use leaky ReLU and pixelwise norm on all conv layers (except the head)
        with arg_scope([conv2d, conv2d_t], activation_fn=conv_activation_fn):
            nb = 64
            for i in range(nlevels):
                net = conv2d(net, nb, 3, 1)
                net = conv2d(net, nb, 4, 2)
                nb = min(nb * 2, 512)
                if i == nlevels - 1:
                    tf.add_to_collection(FEATURE_MATCH, net) # so it can be fetched for feature matching

            net = conv2d(net, nb, 3, 1)
            net = tf.reshape(net, [-1, 4 * 4 * nb])
        net = fully_connected(net, 1, activation_fn=None, scope="y")
    return net


def gan_loss(
        gan_model: tfgan.GANModel,
        generator_loss_fn=tfgan.losses.modified_generator_loss,
        discriminator_loss_fn=tfgan.losses.modified_discriminator_loss,
        gradient_penalty_weight=None,
        gradient_penalty_epsilon=1e-10,
        gradient_penalty_target=1.0,
        feature_matching=False,
        add_summaries=False):
    """ Create A GAN loss set, with support for feature matching.
    Args:
        bigan_model: the model
        feature_matching: Whether to add a feature matching loss to the encoder
      and generator.
    """
    gan_loss = tfgan.gan_loss(
        gan_model,
        generator_loss_fn=generator_loss_fn,
        discriminator_loss_fn=discriminator_loss_fn,
        gradient_penalty_weight=gradient_penalty_weight,
        gradient_penalty_target=1.0,
        add_summaries=add_summaries)

    if feature_matching:
        fm_loss = feature_matching_loss(scope=gan_model.discriminator_scope.name)
        if add_summaries:
            tf.summary.scalar("feature_matching_loss", fm_loss)
        # or combine the original adversarial loss with FM
        gen_loss = gan_loss.generator_loss + fm_loss
        disc_loss = gan_loss.discriminator_loss
        gan_loss = tfgan.GANLoss(gen_loss, disc_loss)

    return gan_loss


def gan_loss_by_name(gan_model: tfgan.GANModel, name: str, feature_matching=False, add_summaries=True, wasserstein_penalty_weight=10.0):
    if name == "WASSERSTEIN":
        return gan_loss(
            gan_model,
            generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
            discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
            gradient_penalty_weight=wasserstein_penalty_weight,
            feature_matching=feature_matching,
            add_summaries=add_summaries
        )
    elif name == "RELATIVISTIC_AVG":
        return gan_loss(
            gan_model,
            generator_loss_fn=relativistic_average_generator_loss,
            discriminator_loss_fn=relativistic_average_discriminator_loss,
            feature_matching=feature_matching,
            add_summaries=add_summaries
        )
    return gan_loss(
        gan_model,
        generator_loss_fn=tfgan.losses.modified_generator_loss,
        discriminator_loss_fn=tfgan.losses.modified_discriminator_loss,
        feature_matching=feature_matching,
        add_summaries=add_summaries
    )


def basic_accuracy(labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    preds = tf.clip_by_value(tf.math.sign(logits), 0, 1)
    trues = tf.math.count_nonzero(tf.math.equal(labels, preds), dtype=tf.float32)
    return trues / tf.cast(tf.shape(labels)[0], tf.float32)


def build_gan_harness(image_input: tf.Tensor,
                      noise: tf.Tensor,
                      generator: tf.keras.Model,
                      discriminator: tf.keras.Model,
                      generator_learning_rate=0.01,
                      discriminator_learning_rate=0.01,
                      noise_format: str = 'SPHERE',
                      adversarial_training: str = 'WASSERSTEIN',
                      feature_matching: bool = False,
                      no_trainer: bool = False,
                      summarize_activations: bool = False) -> tuple:
    image_size = image_input.shape.as_list()[1]
    nchannels = image_input.shape.as_list()[3]
    print("Plain Generative Adversarial Network: {}x{}x{} images".format(
        image_size, image_size, nchannels))
    def _generator_fn(z):
        return generator([z], training=True)

    def _discriminator_fn(x, z):
        return discriminator([x, z], training=True)

    gan_model = tfgan.gan_model(
        _generator_fn, _discriminator_fn, image_input, noise,
        generator_scope='Generator',
        discriminator_scope='Discriminator',
        check_shapes=True)  # set to False for 2-level architectures

    sampled_x = gan_model.generated_data
    image_grid_summary(sampled_x, grid_size=3, name='generated_data')
    if summarize_activations:
        tf.contrib.layers.summarize_activations()
    tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    loss = gan_loss_by_name(
        gan_model, adversarial_training, feature_matching=feature_matching, add_summaries=True)

    if adversarial_training != 'WASSERSTEIN' and adversarial_training != 'RELATIVISTIC_AVG':
        disc_accuracy_gen = basic_accuracy(tf.zeros_like(gan_model.discriminator_gen_outputs), gan_model.discriminator_gen_outputs)
        disc_accuracy_real = basic_accuracy(tf.ones_like(gan_model.discriminator_real_outputs), gan_model.discriminator_real_outputs)
        disc_accuracy = (disc_accuracy_gen + disc_accuracy_real) * 0.5
        with tf.name_scope('Discriminator'):
            tf.summary.scalar('accuracy', disc_accuracy)

    if no_trainer:
        train_ops = None
    else:
        train_ops = tfgan.gan_train_ops(gan_model, loss,
                                        generator_optimizer=tf.train.AdamOptimizer(
                                            generator_learning_rate, beta1=0., beta2=0.99),
                                        discriminator_optimizer=tf.train.AdamOptimizer(
                                            discriminator_learning_rate, beta1=0., beta2=0.99),
                                        summarize_gradients=True)
    return (gan_model, loss, train_ops)
