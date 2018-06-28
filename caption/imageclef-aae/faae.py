"""Module for functions specific to the Flipped Adversarial Autoencoder.
"""

import tensorflow as tf
from tensorflow.contrib import gan as tfgan
from util import add_drift_regularizer, image_summaries_generated, image_grid_summary, minibatch_stddev
from ae import conv_block, conv_t_block, build_dcgan_encoder, build_dcgan_generator, build_encoder, build_1lvl_generator, code_autoencoder_mse_cosine
from aae_train import aegan_model, aegan_train_ops
from AMSGrad import AMSGrad
from plain_gan import build_dcgan_discriminator, build_discriminator_1lvl, build_discriminator_2lvl


def build_faae_harness(image_input: tf.Tensor,
                       noise: tf.Tensor,
                       generator,
                       discriminator,
                       encoder,
                       noise_format: str = 'SPHERE',
                       adversarial_training: str = 'WASSERSTEIN',
                       no_trainer: bool = False,
                       summarize_activations: bool = False):
    image_size = image_input.shape.as_list()[1]
    noise_dim = noise.shape.as_list()[1]
    nchannels = image_input.shape.as_list()[3]
    print("Flipped Adversarial Auto-Encoder: {}x{} images".format(image_size, image_size))

    def _generator_fn(z):
        return generator(
            z, nchannels=nchannels, add_summaries=True, mode='TRAIN')

    def _encoder_fn(x):
        return encoder(
            x, noise_dim, batch_norm=False, add_summary=False,
            sphere_regularize=(noise_format == 'SPHERE'),
            mode='TRAIN')

    def _discriminator_fn(x, z):
        return discriminator(
            x, z, add_drift_loss=True, batch_norm=False, mode='TRAIN')

    gan_model = aegan_model(
        _generator_fn, _discriminator_fn, _encoder_fn, image_input, noise,
        generator_scope='Generator',
        discriminator_scope='Discriminator',
        encoder_scope='Encoder',
        check_shapes=True)  # set to False for 2-level architectures

    sampled_x = gan_model.generated_data
    image_grid_summary(sampled_x, grid_size=4, name='generated_data')
    if summarize_activations:
        tf.contrib.layers.summarize_activations()
    tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # summarize encoded Z
    with tf.variable_scope(gan_model.encoder_scope):
        tf.summary.histogram('encoded_z', gan_model.encoder_gen_outputs)

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
    rec_loss = code_autoencoder_mse_cosine(
        gan_model.generator_inputs, gan_model.encoder_gen_outputs, 1e-3, add_summary=True)

    if no_trainer:
        train_ops = None
    else:
        train_ops = aegan_train_ops(gan_model, gan_loss, rec_loss,
                                    generator_optimizer=AMSGrad(
                                        1e-4, beta1=0.5, beta2=0.99),
                                    discriminator_optimizer=AMSGrad(
                                        1e-4, beta1=0.5, beta2=0.99),
                                    reconstruction_optimizer=AMSGrad(
                                        1e-4, beta1=0.5, beta2=0.99),
                                    summarize_gradients=True)

    return (gan_model, gan_loss, rec_loss, train_ops)
