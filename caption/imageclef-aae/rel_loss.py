"""Module for relativistic GAN loss functions.

Please see `The relativistic discriminator: a key element missing from standard GAN`
    (https://arxiv.org/abs/1807.00734) for more details.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.contrib import gan as tfgan


def relativistic_discriminator_loss_impl(
        discriminator_real_outputs: tf.Tensor,
        discriminator_gen_outputs: tf.Tensor,
        label_smoothing=0.0,
        real_weights=1.0,
        generated_weights=1.0,
        scope=None,
        loss_collection=tf.GraphKeys.LOSSES,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
        add_summaries=False):
    """Relativistic discriminator loss for GANs (RSGAN), with label smoothing.

    L = - real_weights * log(sigmoid(D(x_r) - D(G(z))))
        - generated_weights * log(sigmoid(D(G(z)) - D(x_r)))

    See `The relativistic discriminator: a key element missing from standard GAN`
    (https://arxiv.org/abs/1807.00734) for more details.

    Args:
      discriminator_real_outputs: Discriminator output on real data.
      discriminator_gen_outputs: Discriminator output on generated data. Expected
        to be in the range of (-inf, inf).
      label_smoothing: The amount of smoothing for positive labels. This technique
      is taken from `Improved Techniques for Training GANs` (https://arxiv.org/abs/1606.03498). `0.0` means no smoothing.
      Label smoothing is not hereby proven to improve or deteriorate performance in RaSGANs.
      real_weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `real_data`, and must be broadcastable to `real_data` (i.e., all
        dimensions must be either `1`, or the same as the corresponding
        dimension).
      generated_weights: Same as `real_weights`, but for `generated_data`.
      scope: The scope for the operations performed in computing the loss.
      loss_collection: collection to which this loss will be added.
      reduction: A `tf.losses.Reduction` to apply to loss.
      add_summaries: Whether or not to add summaries for the loss.

    Returns:
      A loss Tensor. The shape depends on `reduction`.
    """
    with tf.name_scope(scope, 'discriminator_relativistic_avg_loss', (
            discriminator_real_outputs, discriminator_gen_outputs, real_weights,
            generated_weights, label_smoothing)) as scope:

        loss_on_real = tf.losses.sigmoid_cross_entropy(
            tf.ones_like(discriminator_real_outputs),
            discriminator_real_outputs - discriminator_gen_outputs,
            real_weights, label_smoothing, scope,
            loss_collection=None, reduction=reduction)
        loss_on_generated = tf.losses.sigmoid_cross_entropy(
            tf.zeros_like(discriminator_gen_outputs),
            discriminator_gen_outputs - discriminator_real_outputs,
            generated_weights, scope=scope,
            loss_collection=None, reduction=reduction)

        loss = (loss_on_real + loss_on_generated) / 2.

        if loss_collection:
            tf.add_to_collection(loss_collection, loss)

        if add_summaries:
            tf.summary.scalar(
                'discriminator_gen_loss', loss_on_generated)
            tf.summary.scalar('discriminator_real_loss', loss_on_real)
            tf.summary.scalar('discriminator_relativistic_avg_loss', loss)

    return loss


def relativistic_average_discriminator_loss_impl(
        discriminator_real_outputs: tf.Tensor,
        discriminator_gen_outputs: tf.Tensor,
        label_smoothing=0.0,
        real_weights=1.0,
        generated_weights=1.0,
        scope=None,
        loss_collection=tf.GraphKeys.LOSSES,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
        add_summaries=False):
    """Relativistic average discriminator loss for GANs (RaSGAN), with label smoothing.

    L = - real_weights * log(sigmoid(D(x_r) - mean(D(G(z)))))
        - generated_weights * log(sigmoid(D(G(z)) - mean(D(x_r))))

    See `The relativistic discriminator: a key element missing from standard GAN`
    (https://arxiv.org/abs/1807.00734) for more details.

    Args:
      discriminator_real_outputs: Discriminator output on real data.
      discriminator_gen_outputs: Discriminator output on generated data. Expected
        to be in the range of (-inf, inf).
      label_smoothing: The amount of smoothing for positive labels. This technique
      is taken from `Improved Techniques for Training GANs` (https://arxiv.org/abs/1606.03498). `0.0` means no smoothing.
      Label smoothing is not hereby proven to improve or deteriorate performance in RaSGANs.
      real_weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `real_data`, and must be broadcastable to `real_data` (i.e., all
        dimensions must be either `1`, or the same as the corresponding
        dimension).
      generated_weights: Same as `real_weights`, but for `generated_data`.
      scope: The scope for the operations performed in computing the loss.
      loss_collection: collection to which this loss will be added.
      reduction: A `tf.losses.Reduction` to apply to loss.
      add_summaries: Whether or not to add summaries for the loss.

    Returns:
      A loss Tensor. The shape depends on `reduction`.
    """
    with tf.name_scope(scope, 'discriminator_relativistic_avg_loss', (
            discriminator_real_outputs, discriminator_gen_outputs, real_weights,
            generated_weights, label_smoothing)) as scope:

        d_real, d_gen = _relativistic_avg_discriminator_outputs(
            discriminator_real_outputs, discriminator_gen_outputs,
            add_summaries=add_summaries)

        loss_on_real = tf.losses.sigmoid_cross_entropy(
            tf.ones_like(discriminator_real_outputs), d_real,
            real_weights, label_smoothing, scope,
            loss_collection=None, reduction=reduction)
        loss_on_generated = tf.losses.sigmoid_cross_entropy(
            tf.zeros_like(discriminator_gen_outputs), d_gen,
            generated_weights, scope=scope,
            loss_collection=None, reduction=reduction)

        loss = (loss_on_real + loss_on_generated) / 2.

        if loss_collection:
            tf.add_to_collection(loss_collection, loss)

        if add_summaries:
            tf.summary.scalar(
                'discriminator_gen_loss', loss_on_generated)
            tf.summary.scalar('discriminator_real_loss', loss_on_real)
            tf.summary.scalar('discriminator_relativistic_avg_loss', loss)

    return loss


def relativistic_average_discriminator_loss(
        gan_model: tfgan.GANModel,
        label_smoothing=0.0,
        real_weights=1.0,
        generated_weights=1.0,
        scope=None,
        loss_collection=tf.GraphKeys.LOSSES,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
        add_summaries=False):

    return relativistic_average_discriminator_loss_impl(
        gan_model.discriminator_real_outputs,
        gan_model.discriminator_gen_outputs,
        label_smoothing=label_smoothing,
        real_weights=real_weights,
        generated_weights=generated_weights,
        scope=scope,
        loss_collection=loss_collection,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
        add_summaries=add_summaries)


def relativistic_average_generator_loss_impl(
        discriminator_real_outputs: tf.Tensor,
        discriminator_gen_outputs: tf.Tensor,
        label_smoothing=0.0,
        real_weights=1.0,
        generated_weights=1.0,
        scope=None,
        loss_collection=tf.GraphKeys.LOSSES,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
        add_summaries=False):
    """Generator loss for Relativistic average GANs (RaSGANs).

    See `The relativistic discriminator: a key element missing from standard GAN`
    (https://arxiv.org/abs/1807.00734) for more details.

    Args:
      discriminator_real_outputs: Discriminator output on real data.
      discriminator_gen_outputs: Discriminator output on generated data. Expected
        to be in the range of (-inf, inf).
      label_smoothing: The amount of smoothing for positive labels. This technique
        is taken from `Improved Techniques for Training GANs`
        (https://arxiv.org/abs/1606.03498). `0.0` means no smoothing.
        Label smoothing is not hereby proven to improve or deteriorate performance in RaSGANs.
      weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `discriminator_gen_outputs`, and must be broadcastable to `labels` (i.e.,
        all dimensions must be either `1`, or the same as the corresponding
        dimension).
      scope: The scope for the operations performed in computing the loss.
      loss_collection: collection to which this loss will be added.
      reduction: A `tf.losses.Reduction` to apply to loss.
      add_summaries: Whether or not to add summaries for the loss.

    Returns:
      A loss Tensor. The shape depends on `reduction`.
    """
    with tf.name_scope(scope, 'generator_relativistic_avg_loss',
                       [discriminator_gen_outputs]) as scope:

        (d_real, d_gen) = _relativistic_avg_discriminator_outputs(
            discriminator_real_outputs, discriminator_gen_outputs,
            add_summaries=False) # only discriminator needs to know this

        loss_on_real = tf.losses.sigmoid_cross_entropy(
            tf.zeros_like(discriminator_real_outputs), d_real,
            real_weights, label_smoothing, scope,
            loss_collection=None, reduction=reduction)
        loss_on_generated = tf.losses.sigmoid_cross_entropy(
            tf.ones_like(discriminator_gen_outputs), d_gen,
            generated_weights, scope=scope,
            loss_collection=None, reduction=reduction)

        loss = (loss_on_real + loss_on_generated) / 2.

        if loss_collection:
            tf.add_to_collection(loss_collection, loss)

        if add_summaries:
            tf.summary.scalar(
                'generator_gen_loss', loss_on_generated)
            tf.summary.scalar('generator_real_loss', loss_on_real)
            tf.summary.scalar('generator_relativistic_avg_loss', loss)

    return loss


def relativistic_average_generator_loss(
        gan_model: tfgan.GANModel,
        label_smoothing=0.0,
        real_weights=1.0,
        generated_weights=1.0,
        scope=None,
        loss_collection=tf.GraphKeys.LOSSES,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
        add_summaries=False):

    return relativistic_average_generator_loss_impl(
        gan_model.discriminator_real_outputs,
        gan_model.discriminator_gen_outputs,
        label_smoothing=label_smoothing,
        real_weights=real_weights,
        generated_weights=generated_weights,
        scope=scope,
        loss_collection=loss_collection,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
        add_summaries=add_summaries)


def _relativistic_avg_discriminator_outputs(
        discriminator_real_outputs: tf.Tensor,
        discriminator_gen_outputs: tf.Tensor,
        add_summaries: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
    """Calculate the relativistic average discriminator outputs for both real and generated data:

        d_real = D(x_r) - mean(D(x_g))
        d_gen = D(x_g) - mean(D(x_r))

    Returns: (d_real, d_gen)
    """
    avg_disc_on_generated = tf.reduce_mean(discriminator_gen_outputs)
    avg_disc_on_real = tf.reduce_mean(discriminator_real_outputs)

    if add_summaries:
      tf.summary.scalar("avg_disc_on_generated", avg_disc_on_generated)
      tf.summary.scalar("avg_disc_on_real", avg_disc_on_real)

    d_real = discriminator_real_outputs - avg_disc_on_generated
    d_gen = discriminator_gen_outputs - avg_disc_on_real

    return (d_real, d_gen)
