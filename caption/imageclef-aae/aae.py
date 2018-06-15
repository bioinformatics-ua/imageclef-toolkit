"""Adversarial auto-encoder components."""

import tensorflow as tf
from tensorflow.contrib import gan as tfgan
from util import add_drift_regularizer, image_grid_summary
from ae import conv_block, conv_t_block, build_encoder, build_1lvl_generator, code_autoencoder_mse
from aae_train import aegan_model, aegan_train_ops, AEGANModel, AEGANTrainOps
from AMSGrad import AMSGrad

layer_norm = tf.contrib.layers.layer_norm

conv2d = tf.contrib.layers.conv2d
conv2d_t = tf.contrib.layers.conv2d_transpose
fully_connected = tf.contrib.layers.fully_connected
arg_scope = tf.contrib.framework.arg_scope


def build_code_discriminator(z: tf.Tensor, add_drift_loss=True, input_noise_factor=0.125, hidden_noise_factor=0.25,
                             add_summaries=True, mode=None) -> tf.Tensor:
    "1-level latent code discriminator (only looks at z)"

    assert len(z.shape.as_list()) == 2

    # add Gaussian noise
    z_noisy = z
    if input_noise_factor:
        z_noisy += tf.random_normal(tf.shape(z),
                                    mean=0.0, stddev=0.1 * input_noise_factor)

    # collect all activations and trainable variables
    with arg_scope([fully_connected], outputs_collections=[tf.GraphKeys.ACTIVATIONS],
                   variables_collections=[tf.GraphKeys.TRAINABLE_VARIABLES],
                   weights_initializer=tf.random_normal_initializer(mean=0, stddev=0.02)):
        # use batch norm and leaky ReLU on all body layers
        with arg_scope([fully_connected], normalizer_fn=layer_norm,
                       normalizer_params={'center': True, 'scale': True},
                       activation_fn=tf.nn.leaky_relu):
            net = fully_connected(z_noisy, 1024)  # Bx1024
            net = tf.layers.dropout(
                net, rate=hidden_noise_factor, training=True)
            net = fully_connected(net, 1024)  # Bx1024
            net = tf.layers.dropout(
                net, rate=hidden_noise_factor, training=True)
            net = fully_connected(net, 1024)  # Bx1024
            net = tf.layers.dropout(
                net, rate=hidden_noise_factor, training=True)
        # net = minibatch_stddev(net)  # add stddev to minibatch
        net = fully_connected(net, 1, activation_fn=None)
        if add_drift_loss:
            add_drift_regularizer(net, add_summary=add_summaries)
    return net


def build_aae_harness(image_input: tf.Tensor,
                      noise: tf.Tensor,
                      decoder_fn,
                      discriminator_fn,
                      encoder_fn,
                      noise_format: str = 'SPHERE',
                      adversarial_training: str = 'WASSERSTEIN',
                      no_trainer: bool = False,
                      summarize_activations: bool = False) -> (AEGANModel, tf.Tensor, tf.Tensor, AEGANTrainOps):
    """Build an adversarial auto-encoder's training harness."""

    image_size = image_input.shape.as_list()[1]
    noise_dim = noise.shape.as_list()[1]
    print("Adversarial Auto-Encoder: {}x{} images".format(image_size, image_size))

    # AAE's are inverted from the perspective of a GAN: generated samples are latent codes.
    # So the encoder is the generator's AAE, and the generator is the encoder's AAE.
    def _generator_fn(x):
        return encoder_fn(x, noise_dim, sphere_regularize=noise_format == 'SPHERE',
                          add_summary=False, mode='TRAIN')

    def _encoder_fn(z):
        return decoder_fn(z, mode='TRAIN')

    def _discriminator_fn(z, x):
        return discriminator_fn(z, add_drift_loss=True, mode='TRAIN')

    gan_model = aegan_model(
        _generator_fn, _discriminator_fn, _encoder_fn, noise, image_input,
        generator_scope='Encoder',  # switching scope names here
        discriminator_scope='Discriminator',
        encoder_scope='Generator',
        check_shapes=True)  # must be false when passing tuples/lists of tensors

    image_grid_summary(gan_model.encoder_gen_outputs,
                       grid_size=4, name='generated_data')
    if summarize_activations:
        tf.contrib.layers.summarize_activations()
    tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # summarize encoded Z
    with tf.variable_scope(gan_model.generator_scope):
        tf.summary.histogram('encoded_z', gan_model.generated_data)

    assert gan_model.generated_data is not None
    assert gan_model.encoder_gen_outputs is not None
    assert gan_model.discriminator_gen_outputs is not None
    assert gan_model.discriminator_real_outputs is not None

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

    # add auto-encoder reconstruction loss
    z = image_input
    encoded_z = gan_model.encoder_gen_outputs
    rec_loss = code_autoencoder_mse(z, encoded_z)

    if no_trainer:
        train_ops = None
    else:
        train_ops = aegan_train_ops(gan_model, gan_loss, rec_loss,
                                    generator_optimizer=AMSGrad(
                                        1e-4, beta1=0.0, beta2=0.9),
                                    discriminator_optimizer=AMSGrad(
                                        1e-4, beta1=0.0, beta2=0.9),
                                    reconstruction_optimizer=AMSGrad(
                                        1e-4, beta1=0.0, beta2=0.9),
                                    summarize_gradients=True
                                    )

    return (gan_model, gan_loss, rec_loss, train_ops)
