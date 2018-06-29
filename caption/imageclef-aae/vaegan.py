"""Variational Auto-encoder GAN components."""

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


def vae_reg(mu, stddev):
    encoder_dist = tf.distributions.Normal(loc=mu, scale=stddev)
    unit_normal_dist = tf.distributions.Normal(
        loc=tf.zeros_like(mu), scale=tf.ones_like(stddev))
    kl_divergence = tf.distributions.kl_divergence(encoder_dist, unit_normal_dist)
    kl_divergence = tf.reduce_mean(kl_divergence)
    return kl_divergence

def build_vaegan_harness(image_input: tf.Tensor,
                      noise_dim: int,
                      decoder_fn,
                      discriminator_fn,
                      encoder_fn,
                      adversarial_training: str = 'WASSERSTEIN',
                      no_trainer: bool = False,
                      summarize_activations: bool = False) -> (AEGANModel, tf.Tensor, tf.Tensor, AEGANTrainOps):
    """Build an adversarial auto-encoder's training harness."""

    image_size = image_input.shape.as_list()[1]
    nchannels = image_input.shape.as_list()[3]
    print("Adversarial Auto-Encoder: {}x{} images".format(image_size, image_size))

    # The VAEGAN's generator receives the encoder's output as prior code.
    def _encoder_fn(x):
        # TODO split
        out = encoder_fn(x, noise_dim * 2, add_summary=False, mode='TRAIN')
        mu = out[:, :noise_dim]
        stddev = out[:, noise_dim:]
        rnd_sample = tf.random_normal(tf.shape(mu))
        return (mu, stddev, mu + stddev * rnd_sample)

    def _generator_fn(x):
        _mu, _stddev, noise = _encoder_fn(x)
        return decoder_fn(noise, nchannels=nchannels, mode='TRAIN')

    def _discriminator_fn(x, z):
        return discriminator_fn(x, add_drift_loss=True, mode='TRAIN')

    gan_model = aegan_model(
        _generator_fn, _discriminator_fn, _encoder_fn, None, image_input,
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
    z_mu, z_stddev, encoded_z = gan_model.encoder_gen_outputs
    rec_loss = code_autoencoder_mse(z, encoded_z)
    vreg_loss = vae_reg(z_mu, z_stddev)

    if no_trainer:
        train_ops = None
    else:
        train_ops = aegan_train_ops(gan_model, gan_loss, rec_loss + vreg_loss,
                                    generator_optimizer=AMSGrad(
                                        1e-4, beta1=0.5, beta2=0.99),
                                    discriminator_optimizer=AMSGrad(
                                        1e-4, beta1=0.5, beta2=0.99),
                                    reconstruction_optimizer=AMSGrad(
                                        1e-4, beta1=0.5, beta2=0.99),
                                    summarize_gradients=True
                                    )

    return (gan_model, gan_loss, rec_loss, train_ops)
