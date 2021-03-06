"""Adversarial auto-encoder components."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.contrib import gan as tfgan
from util import add_drift_regularizer, image_grid_summary
from plain_gan import gan_loss_by_name
from ae import conv_block, conv_t_block, build_encoder, build_1lvl_generator, autoencoder_mse, autoencoder_bce
from aae_train import aegan_model, aegan_train_ops, AEGANModel, AEGANTrainOps
from AMSGrad import AMSGrad

layer_norm = tf.contrib.layers.layer_norm

fully_connected = tf.contrib.layers.fully_connected
arg_scope = tf.contrib.framework.arg_scope

def build_code_discriminator(z: tf.Tensor, add_drift_loss=False, input_noise_factor=0., hidden_noise_factor=0.25,
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
            net = fully_connected(net, 1024)  # Bx1024
            net = fully_connected(net, 1024)  # Bx1024
            net = keras.layers.Dropout(rate=hidden_noise_factor)(net, training=True)
        net = fully_connected(net, 1, activation_fn=None)
        if add_drift_loss:
            add_drift_regularizer(net, add_summary=add_summaries)
    return net


class CodeDiscriminator(tf.keras.Model):

    def __init__(self, nb: int = 512, add_drift_loss: bool = False, input_noise_factor=0., hidden_noise_factor=0.25):
        super(CodeDiscriminator, self).__init__()
        self.nb = nb
        self.add_drift_loss = add_drift_loss
        self.input_noise_factor = input_noise_factor
        self.hidden_noise_factor = hidden_noise_factor

    def call(self, inputs, training):
        if isinstance(inputs, list):
            inputs = inputs[0]
        return build_code_discriminator(
            inputs, add_drift_loss=self.add_drift_loss,
            input_noise_factor=self.input_noise_factor,
            hidden_noise_factor=self.hidden_noise_factor,
            mode='TRAIN' if training else 'PREDICT')


def build_aae_harness(image_input: tf.Tensor,
                      noise: tf.Tensor,
                      decoder_fn,
                      discriminator_fn,
                      encoder_fn,
                      noise_format: str = 'SPHERE',
                      adversarial_training: str = 'WASSERSTEIN',
                      sample_noise: tf.Tensor = None,
                      discriminator_learning_rate=1e-5,
                      generator_learning_rate=1e-5,
                      reconstruction_learning_rate=1e-5,
                      no_trainer: bool = False,
                      summarize_activations: bool = False) -> (AEGANModel, tf.Tensor, tf.Tensor, AEGANTrainOps):
    """Build an adversarial auto-encoder's training harness."""

    image_size = image_input.shape.as_list()[1]
    noise_dim = noise.shape.as_list()[1]
    nchannels = image_input.shape.as_list()[3]
    print("Adversarial Auto-Encoder: {}x{}x{} images → {}D code".format(image_size, image_size, nchannels, noise_dim))

    # AAE's are inverted from the perspective of a GAN: generated samples are latent codes.
    # So the encoder is the generator's AAE, and the generator is the encoder's AAE.
    def _generator_fn(x):
        return encoder_fn([x], training=True)

    def _encoder_fn(z):
        return decoder_fn([z], training=True)

    def _discriminator_fn(z, x):
        return discriminator_fn([z], training=True)

    if sample_noise is not None:
        image_input_noisy = image_input + sample_noise
    else:
        image_input_noisy = image_input

    gan_model = aegan_model(
        _generator_fn, _discriminator_fn, _encoder_fn, noise, image_input_noisy,
        generator_scope='Encoder',  # switching scope names here
        discriminator_scope='Discriminator',
        encoder_scope='Generator',
        check_shapes=True)  # must be false when passing tuples/lists of tensors

    image_grid_summary(gan_model.encoder_gen_outputs,
                       grid_size=3, name='generated_data')
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

    gan_loss = gan_loss_by_name(gan_model, adversarial_training, add_summaries=True)

    # add auto-encoder reconstruction loss
    generated_x = gan_model.encoder_gen_outputs
    rec_loss = autoencoder_mse(image_input, generated_x)
    tf.summary.scalar('reconstruction_loss', rec_loss)

    if no_trainer:
        train_ops = None
    else:
        train_ops = aegan_train_ops(gan_model, gan_loss, rec_loss,
                                    generator_optimizer=tf.train.AdamOptimizer(
                                        generator_learning_rate, beta1=0.5, beta2=0.999),
                                    discriminator_optimizer=tf.train.AdamOptimizer(
                                        discriminator_learning_rate, beta1=0.5, beta2=0.999),
                                    reconstruction_optimizer=tf.train.AdamOptimizer(
                                        reconstruction_learning_rate, beta1=0.5, beta2=0.999),
                                    summarize_gradients=True
                                    )

    return (gan_model, gan_loss, rec_loss, train_ops)
