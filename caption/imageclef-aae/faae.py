"""Module for functions specific to the Flipped Adversarial Autoencoder.
"""

import tensorflow as tf
from tensorflow.contrib import gan as tfgan
from util import add_drift_regularizer, image_summaries_generated, image_grid_summary, minibatch_stddev
from plain_gan import gan_loss_by_name
from ae import conv_block, conv_t_block, build_dcgan_encoder, build_dcgan_generator, build_encoder, build_1lvl_generator, code_autoencoder_mse_cosine
from aae_train import aegan_model, aegan_train_ops
from AMSGrad import AMSGrad
from plain_gan import build_dcgan_discriminator, build_discriminator_1lvl, build_discriminator_2lvl


def build_faae_harness(image_input: tf.Tensor,
                       noise: tf.Tensor,
                       generator: tf.keras.Model,
                       discriminator: tf.keras.Model,
                       encoder: tf.keras.Model,
                       generator_learning_rate = 1e-4,
                       discriminator_learning_rate = 2e-4,
                       reconstruction_learning_rate = 5e-5,
                       noise_format: str = 'SPHERE',
                       adversarial_training: str = 'WASSERSTEIN',
                       no_trainer: bool = False,
                       summarize_activations: bool = False):
    image_size = image_input.shape.as_list()[1]
    print("Flipped Adversarial Auto-Encoder: {}x{} images".format(image_size, image_size))

    def _generator_fn(z):
        return generator([z], training=True)

    def _encoder_fn(x):
        return encoder([x], training=True)

    def _discriminator_fn(x, z):
        return discriminator([x, z], training=True)

    gan_model = aegan_model(
        _generator_fn, _discriminator_fn, _encoder_fn, image_input, noise,
        generator_scope='Generator',
        discriminator_scope='Discriminator',
        encoder_scope='Encoder',
        check_shapes=True)  # set to False for 2-level architectures

    sampled_x = gan_model.generated_data
    image_grid_summary(sampled_x, grid_size=2, name='generated_data')
    if summarize_activations:
        tf.contrib.layers.summarize_activations()
    tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # summarize encoded Z
    with tf.variable_scope(gan_model.encoder_scope):
        tf.summary.histogram('encoded_z', gan_model.encoder_gen_outputs)

    gan_loss = gan_loss_by_name(gan_model, adversarial_training, add_summaries=True)

    # add auto-encoder reconstruction loss
    rec_loss = code_autoencoder_mse_cosine(
        gan_model.generator_inputs, gan_model.encoder_gen_outputs, 1e-3, add_summary=True)

    if no_trainer:
        train_ops = None
    else:
        train_ops = aegan_train_ops(gan_model, gan_loss, rec_loss,
                                    generator_optimizer=AMSGrad(
                                        generator_learning_rate, beta1=0.5, beta2=0.999),
                                    discriminator_optimizer=AMSGrad(
                                        discriminator_learning_rate, beta1=0.5, beta2=0.999),
                                    reconstruction_optimizer=AMSGrad(
                                        reconstruction_learning_rate, beta1=0.5, beta2=0.999),
                                    summarize_gradients=True)

    return (gan_model, gan_loss, rec_loss, train_ops)
