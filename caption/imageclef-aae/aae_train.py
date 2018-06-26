"""Components for the training process of an auto-encoding GAN."""

import tensorflow as tf
from tensorflow.contrib import gan as tfgan


class AEGANModel(tfgan.GANModel):
    def __new__(cls,
                generator_inputs,
                generated_data,
                generator_variables,
                generator_scope,
                generator_fn,
                real_data,
                discriminator_real_outputs,
                discriminator_gen_outputs,
                discriminator_variables,
                discriminator_scope,
                discriminator_fn,
                encoder_real_outputs,
                encoder_gen_outputs,
                encoder_variables,
                encoder_scope,
                encoder_fn):
        self = super(AEGANModel, cls).__new__(
            cls,
            generator_inputs,
            generated_data,
            generator_variables,
            generator_scope,
            generator_fn,
            real_data,
            discriminator_real_outputs,
            discriminator_gen_outputs,
            discriminator_variables,
            discriminator_scope,
            discriminator_fn)
        self.encoder_real_outputs = encoder_real_outputs
        self.encoder_gen_outputs = encoder_gen_outputs
        self.encoder_variables = encoder_variables
        self.encoder_scope = encoder_scope
        self.encoder_fn = encoder_fn
        return self


def aegan_model(
    # Lambdas defining models.
    generator_fn,
    discriminator_fn,
    encoder_fn,
    # Real data and conditioning.
    real_data,
    generator_inputs,
    # Optional scopes.
    generator_scope='Generator',
    discriminator_scope='Discriminator',
    encoder_scope='Encoder',
    # Options.
        check_shapes=True):

    gan_model = tfgan.gan_model(
        generator_fn,
        discriminator_fn,
        real_data,
        generator_inputs,
        generator_scope=generator_scope,
        discriminator_scope=discriminator_scope,
        check_shapes=check_shapes)

    with tf.variable_scope(encoder_scope) as enc_scope:
        encoder_gen_outputs = encoder_fn(gan_model.generated_data)
    with tf.variable_scope(enc_scope, reuse=True):
        real_data = tf.convert_to_tensor(real_data)
        encoder_real_outputs = encoder_fn(real_data)

    encoder_variables = tf.trainable_variables(scope=encoder_scope)

    return AEGANModel(
        generator_inputs,
        gan_model.generated_data,
        gan_model.generator_variables,
        gan_model.generator_scope,
        generator_fn,
        real_data,
        gan_model.discriminator_real_outputs,
        gan_model.discriminator_gen_outputs,
        gan_model.discriminator_variables,
        gan_model.discriminator_scope,
        discriminator_fn,
        encoder_real_outputs,
        encoder_gen_outputs,
        encoder_variables,
        enc_scope,
        encoder_fn)


class AEGANTrainOps(tfgan.GANTrainOps):
    def __new__(cls, gen_train_op, disc_train_op, rec_train_op, global_step_inc):
        self = super(AEGANTrainOps, cls).__new__(
            cls, gen_train_op, disc_train_op, global_step_inc)
        self.rec_train_op = rec_train_op
        return self


def aegan_train_ops(
    model,
    loss,
    rec_loss,
    generator_optimizer,
    discriminator_optimizer,
    reconstruction_optimizer,
    check_for_unused_update_ops=True,
    # Optional args to pass directly to the `create_train_op`.
        **kwargs):
    """Returns AE GAN train ops. Implementation derived from `tfgan.gan_train_opts`.

    The highest-level call in TFGAN. It is composed of functions that can also
    be called, should a user require more control over some part of the GAN
    training process.

    Args:
      model: An AEGANModel.
      loss: A GANLoss.
      rec_loss: A Tensor with the reconstruction loss
      generator_optimizer: The optimizer for generator updates.
      discriminator_optimizer: The optimizer for the discriminator updates.
      reconstruction_optimizer: The optimizer for the reconstruction updates.
      check_for_unused_update_ops: If `True`, throws an exception if there are
        update ops outside of the generator or discriminator scopes.
      **kwargs: Keyword args to pass directly to
        `training.create_train_op` for both the generator and
        discriminator train op.

    Returns:
      A GANTrainOps tuple of (generator_train_op, discriminator_train_op) that can
      be used to train a generator/discriminator pair.
    """
    # Create global step increment op.
    global_step = tf.train.get_or_create_global_step()
    global_step_inc = global_step.assign_add(1)

    # Get generator, encoder, and discriminator update ops. We split them so that update
    # ops aren't accidentally run multiple times. For now, throw an error if
    # there are update ops that aren't associated with either the generator or
    # the discriminator. Might modify the `kwargs` dictionary.
    (gen_update_ops, dis_update_ops, enc_update_ops) = _get_update_ops(
        kwargs, model.generator_scope.name, model.discriminator_scope.name, model.encoder_scope.name,
        check_for_unused_update_ops)

    rec_update_ops = list(enc_update_ops)

    generator_global_step = None
    if isinstance(generator_optimizer,
                  tf.train.SyncReplicasOptimizer):
        # TODO(joelshor): Figure out a way to get this work without including the
        # dummy global step in the checkpoint.
        # WARNING: Making this variable a local variable causes sync replicas to
        # hang forever.
        generator_global_step = tf.get_variable(
            'dummy_global_step_generator',
            shape=[],
            dtype=global_step.dtype.base_dtype,
            initializer=tf.zeros_initializer(),
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES])
        gen_update_ops += [generator_global_step.assign(global_step)]
    with tf.name_scope('generator_train'):
        gen_train_op = tf.contrib.training.create_train_op(
            total_loss=loss.generator_loss,
            optimizer=generator_optimizer,
            variables_to_train=model.generator_variables,
            global_step=generator_global_step,
            update_ops=gen_update_ops,
            **kwargs)

    discriminator_global_step = None
    if isinstance(discriminator_optimizer,
                  tf.train.SyncReplicasOptimizer):
        # See comment above `generator_global_step`.
        discriminator_global_step = tf.get_variable(
            'dummy_global_step_discriminator',
            shape=[],
            dtype=global_step.dtype.base_dtype,
            initializer=tf.zeros_initializer(),
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES])
        dis_update_ops += [discriminator_global_step.assign(global_step)]
    with tf.name_scope('discriminator_train'):
        disc_train_op = tf.contrib.training.create_train_op(
            total_loss=loss.discriminator_loss,
            optimizer=discriminator_optimizer,
            variables_to_train=model.discriminator_variables,
            global_step=discriminator_global_step,
            update_ops=dis_update_ops,
            **kwargs)

    reconstruction_global_step = None
    if isinstance(reconstruction_optimizer,
                  tf.train.SyncReplicasOptimizer):
        # See comment above `generator_global_step`.
        reconstruction_global_step = tf.get_variable(
            'dummy_global_step_reconstruction',
            shape=[],
            dtype=global_step.dtype.base_dtype,
            initializer=tf.zeros_initializer(),
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES])
        rec_update_ops += [reconstruction_global_step.assign(global_step)]
    with tf.name_scope('reconstruction_train'):
        rec_train_op = tf.contrib.training.create_train_op(
            total_loss=rec_loss,
            optimizer=reconstruction_optimizer,
            variables_to_train=model.generator_variables + model.encoder_variables,
            global_step=reconstruction_global_step,
            update_ops=rec_update_ops,
            **kwargs)

    return AEGANTrainOps(gen_train_op, disc_train_op, rec_train_op, global_step_inc)


def _get_update_ops(kwargs, gen_scope, dis_scope, enc_scope, check_for_unused_ops=True):
    """Gets generator, discriminator, and encoder update ops.

    Args:
      kwargs: A dictionary of kwargs to be passed to `create_train_op`.
        `update_ops` is removed, if present.
      gen_scope: A scope for the generator.
      dis_scope: A scope for the discriminator.
      enc_scope: A scope for the encoder.
      check_for_unused_ops: A Python bool. If `True`, throw Exception if there are
        unused update ops.

    Returns:
      A 2-tuple of (generator update ops, discriminator train ops).

    Raises:
      ValueError: If there are update ops outside of the generator or
        discriminator scopes.
    """
    if 'update_ops' in kwargs:
        update_ops = set(kwargs['update_ops'])
        del kwargs['update_ops']
    else:
        update_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    all_gen_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS, gen_scope))
    all_dis_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS, dis_scope))
    all_enc_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS, enc_scope))

    if check_for_unused_ops:
        unused_ops = update_ops - all_gen_ops - all_dis_ops - all_enc_ops
        if unused_ops:
            raise ValueError('There are unused update ops: %s' % unused_ops)

    gen_update_ops = list(all_gen_ops & update_ops)
    dis_update_ops = list(all_dis_ops & update_ops)
    enc_update_ops = list(all_enc_ops & update_ops)

    return (gen_update_ops, dis_update_ops, enc_update_ops)

