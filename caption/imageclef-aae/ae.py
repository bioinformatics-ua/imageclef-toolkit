import tensorflow as tf
from tensorflow.contrib import gan as tfgan
from util import image_summaries_2level, random_hypersphere, upsample_2d, drift_loss, minibatch_stddev, pixelwise_feature_vector_norm, conv2d_sn, conv2d_t_sn, add_unit_norm_loss

batch_norm = tf.contrib.layers.batch_norm
conv2d = tf.contrib.layers.conv2d
conv2d_t = tf.contrib.layers.conv2d_transpose
fully_connected = tf.contrib.layers.fully_connected
variance_scaling_initializer = tf.contrib.layers.variance_scaling_initializer
arg_scope = tf.contrib.framework.arg_scope

N_CHANNELS = 3


def conv_t_block(incoming: tf.Tensor, nb: int, ks: int = 3, spectral_norm: bool = False):
    if spectral_norm:
        conv2d = conv2d_sn
        conv2d_t = conv2d_t_sn

    if isinstance(nb, int):
        nb = (nb, nb)
    if isinstance(ks, int):
        ks = (ks, ks)
    net = conv2d_t(incoming, nb[0], ks[0], stride=2)
    net = conv2d(net, nb[1], ks[1], stride=1)
    return net


def conv_block(incoming: tf.Tensor, nb: int, ks: int = 3, spectral_norm: bool = False, dropout: float = 0, dropout_istraining: bool = True):
    if spectral_norm:
        conv2d = conv2d_sn
    else:
        conv2d = tf.contrib.layers.conv2d

    if isinstance(nb, int):
        nb = [nb, nb]
    net = conv2d(incoming, nb[0], ks, 2)
    net = conv2d(net, nb[1], ks, 1)
    if dropout:
        net = tf.layers.dropout(net, rate=dropout, training=dropout_istraining)
    return net


def build_dcgan_generator(z: tf.Tensor, nchannels: int = N_CHANNELS, nlevels: int = 4, conv_activation_fn=tf.nn.leaky_relu,
                          batch_norm: bool = True, add_summaries=False, mode=None):
    """Build a generator based on DCGAN. Difference: this uses leaky ReLU by default.
    Args:
      z : the prior code Tensor [B, N]
      nchannels : number of channels of the output (3 for RGB)
      nlevels : number of upsampling network levels (4 for 64x64, 5 for 128x128, etc.)
      conv_activation_fn : a function for the non-linearity operation after each hidden convolution layer
      batch_norm : whether to perform batch normalization after each layer
      add_summaries : whether to produce summary operations
      mode : set to 'TRAIN' during the training process
    """
    is_training = mode == 'TRAIN'
    if batch_norm:
        normalizer_fn = tf.contrib.layers.batch_norm
        normalizer_params = {
            'fused': True,
            'center': True,
            'is_training': is_training
        }
    else:
        normalizer_fn = None
        normalizer_params = None

    net = z
    with arg_scope([fully_connected, conv2d, conv2d_t], outputs_collections=[tf.GraphKeys.ACTIVATIONS],
                   variables_collections=[tf.GraphKeys.TRAINABLE_VARIABLES],
                   weights_initializer=tf.random_normal_initializer(stddev=0.02)):
        # use leaky ReLU and pixelwise norm on all conv layers (except the head)
        with arg_scope([conv2d, conv2d_t], activation_fn=conv_activation_fn,
                       normalizer_fn=normalizer_fn, normalizer_params=normalizer_params):
            net = fully_connected(
                net, 1024 * 4 * 4, activation_fn=tf.nn.leaky_relu)
            net = tf.reshape(net, [-1, 4, 4, 1024])  # Bx4x4x1024
            nb = 512
            for _ in range(nlevels - 1):
                net = conv2d_t(net, nb, 5, 2)
                nb = nb // 2
        net = conv2d_t(net, nchannels, 5, 2,
                       activation_fn=tf.nn.tanh, scope="y")
    return net


def build_sndcgan_generator(z: tf.Tensor, nchannels: int = N_CHANNELS, nlevels: int = 4,
                            bottom_res: int = 4, bottom_nb: int = 512,
                            conv_activation_fn=tf.nn.relu,
                            batch_norm: bool = True, add_summaries=False, mode=None):
    """Build a generator based on SNDCGAN, presented in:
        Miyato et al. Spectral Normalization for generative adversarial networks (2018)

    Args:
      z : the prior code Tensor [B, N]
      nchannels : number of channels of the output (3 for RGB)
      nlevels : number of upsampling network levels (4 for 64x64, 5 for 128x128, etc.)
      conv_activation_fn : a function for the non-linearity operation after each hidden convolution layer
      batch_norm : whether to perform batch normalization after each layer
      add_summaries : whether to produce summary operations
      mode : set to 'TRAIN' during the training process
    """
    assert len(z.shape.as_list()) == 2

    is_training = mode == 'TRAIN'

    if batch_norm:
        normalizer_fn = tf.contrib.layers.batch_norm
        normalizer_params = {
            'fused': True,
            'center': True,
            'is_training': is_training
        }
    else:
        normalizer_fn = None
        normalizer_params = None

    net = z
    with arg_scope([fully_connected, conv2d, conv2d_t], outputs_collections=[tf.GraphKeys.ACTIVATIONS],
                   variables_collections=[tf.GraphKeys.TRAINABLE_VARIABLES],
                   weights_initializer=tf.random_normal_initializer(stddev=0.02)):
        with arg_scope([conv2d, conv2d_t], activation_fn=conv_activation_fn,
                       normalizer_fn=normalizer_fn, normalizer_params=normalizer_params):
            net = fully_connected(
                net, bottom_nb * bottom_res * bottom_res, activation_fn=None)
            net = tf.reshape(net, [-1, bottom_res, bottom_res, bottom_nb])  # Bx4x4x512
            nb = bottom_nb
            for _ in range(nlevels):
                net = conv2d_t(net, nb, 4, 2)
                nb = nb // 2
        net = conv2d(net, nchannels, 3, 1, activation_fn=tf.nn.tanh, scope="y")
    assert len(net.shape.as_list()) == 4
    return net


class SNDCGANGenerator(tf.keras.Model):

    def __init__(self, nchannels: int = N_CHANNELS, nlevels: int = 4,
                 bottom_res: int = 4, bottom_nb: int = 512,
                 conv_activation=tf.nn.relu, batch_norm: bool = True):
        tf.keras.Model.__init__(self)
        self.nchannels = nchannels
        self.nlevels = nlevels
        self.conv_activation = conv_activation
        self.batch_norm = batch_norm
        self.bottom_res = bottom_res
        self.bottom_nb = bottom_nb
    
    def call(self, inputs, training):
        if isinstance(inputs, list):
            inputs = inputs[0]
        return build_sndcgan_generator(
            inputs, self.nchannels, self.nlevels, bottom_res=self.bottom_res,
            bottom_nb=self.bottom_nb, conv_activation_fn=self.conv_activation,
            batch_norm=self.batch_norm, mode='TRAIN' if training else 'PREDICT')


def build_1lvl_generator(z: tf.Tensor, nchannels: int = N_CHANNELS, nlevels: int = 4, mode=None) -> tf.Tensor:
    """"1-level generator
    Args:
      z : the prior code Tensor [B, N]
      mode : set to 'TRAIN' during the training process
    """
    net = z
    is_training = mode == 'TRAIN'
    if batch_norm:
        norm_fn = tf.contrib.layers.batch_norm
        norm_params = {
            'fused': True,
            'center': True,
            'is_training': is_training
        }
    else:
        norm_fn = pixelwise_feature_vector_norm
        norm_params = None

    with arg_scope([fully_connected, conv2d, conv2d_t], outputs_collections=[tf.GraphKeys.ACTIVATIONS],
                   variables_collections=[tf.GraphKeys.TRAINABLE_VARIABLES],
                   weights_initializer=tf.random_normal_initializer(stddev=0.02)):
        # use leaky ReLU and pixelwise norm on all conv layers (except the head)
        with arg_scope([conv2d, conv2d_t], activation_fn=tf.nn.leaky_relu, normalizer_fn=norm_fn, normalizer_params=norm_params):
            net = fully_connected(
                net, 512 * 4 * 4, activation_fn=tf.nn.leaky_relu, scope="fc0")
            net = tf.layers.dropout(net, rate=0.5, training=is_training)
            net = tf.reshape(net, [-1, 4, 4, 512])  # Bx4x4x512
            net = conv2d(net, 3, 1, 512)  # Bx4x4x512
            nb = 512
            for _ in range(nlevels):
                net = conv_t_block(net, [nb, nb // 2])
                nb = nb // 2

        net = conv2d(net, nchannels, 1, 1,
                     activation_fn=tf.nn.tanh, scope="y")
    return net


def build_generator_2lvl(z: tf.Tensor, nchannels=N_CHANNELS, mode=None) -> [tf.Tensor, tf.Tensor]:
    "2-level generator"
    net = z
    is_training = mode == 'TRAIN'
    with arg_scope([fully_connected, conv2d, conv2d_t], outputs_collections=[tf.GraphKeys.ACTIVATIONS],
                   variables_collections=[tf.GraphKeys.TRAINABLE_VARIABLES],
                   weights_initializer=tf.random_normal_initializer(stddev=0.02)):
        # use leaky ReLU and pixelwise norm on all conv layers (except the 2 heads)
        with arg_scope([conv2d, conv2d_t], activation_fn=tf.nn.leaky_relu, normalizer_fn=pixelwise_feature_vector_norm) as conv_scope:
            net = fully_connected(
                net, 512 * 4 * 4, activation_fn=tf.nn.leaky_relu, scope="fc0")
            net = tf.layers.dropout(net, rate=0.5, training=is_training)
            net = tf.reshape(net, [-1, 4, 4, 512])  # Bx4x4x512
            # net = conv2d(net, 3, 1, 512)  # Bx4x4x512
            net = conv_t_block(net, [512, 256])  # Bx8x8x256
            net = conv_t_block(net, [256, 128])  # Bx16x16x128
        net1 = conv2d(net, nchannels, 1, 1,
                      activation_fn=tf.nn.tanh, scope="y_small")  # Bx16x16x3

        with arg_scope(conv_scope):
            net = conv_t_block(net, [128, 64])  # Bx32x32x64
            net = conv_t_block(net, [64, 32])  # Bx64x64x32
        net2 = conv2d(net, nchannels, 1, 1,
                      activation_fn=tf.nn.tanh, scope="y")  # Bx64x64x3
    return [net2, net1]


def build_encoder(x: tf.Tensor, noise_dims: int, nlevels: int = 4, batch_norm=False, add_summary=True, sphere_regularize=False, mode=None) -> tf.Tensor:
    "build an encoder network which maps a sample to the latent space"

    # make it work for 2-level generators: if a list is found, take the full size sample
    if isinstance(x, list):
        x = x[0]

    is_training = mode == 'TRAIN'
    if batch_norm:
        normalizer_fn = tf.contrib.layers.batch_norm
        normalizer_params = {
            'fused': True,
            'center': True,
            'is_training': is_training
        }
    else:
        normalizer_fn = None
        normalizer_params = None

    net = x
    with arg_scope([conv2d, fully_connected], outputs_collections=[tf.GraphKeys.ACTIVATIONS],
                   variables_collections=[tf.GraphKeys.TRAINABLE_VARIABLES],
                   weights_initializer=tf.random_normal_initializer(stddev=0.02)):
        # use leaky ReLU on all body layers
        with arg_scope([conv2d], activation_fn=tf.nn.leaky_relu,
                       normalizer_fn=normalizer_fn,
                       normalizer_params=normalizer_params):
            nb = 64
            for _ in range(nlevels):
                if nb == 512:
                    net = conv_block(net, nb)
                else:
                    net = conv_block(net, [nb, nb * 2])
                    nb <<= 1

        assert net.shape.as_list()[1:4] == [4, 4, 512]
        net = tf.reshape(net, [-1, 4 * 4 * 512])
        net = fully_connected(net, noise_dims, activation_fn=None,
                              biases_initializer=None)
    if sphere_regularize:
        # add activation regularizer (to approach unit magnitude)
        add_unit_norm_loss(net, weight=1e-3, add_summary=add_summary and mode == 'TRAIN')

    assert(len(net.shape.as_list()) == 2)
    if add_summary:
        tf.summary.histogram("encoded_z", net)
    return net


def build_dcgan_encoder(x: tf.Tensor, noise_dims: int, nlevels: int = 4, add_summary=True, batch_norm=True, sphere_regularize=False, mode=None) -> tf.Tensor:
    "build an encoder network which maps a sample to the latent space, based on DCGAN"

    # make it work for 2-level generators
    if isinstance(x, list):
        x = x[0]  # use full size sample

    if batch_norm:
        norm_fn = tf.contrib.layers.batch_norm
        norm_params = {
            'center': True,
            'scale': True,
            'fused': True,
            'is_training': mode == 'TRAIN'
        }
    else:
        norm_fn = None
        norm_params = None

    net = x
    with arg_scope([conv2d, fully_connected], outputs_collections=[tf.GraphKeys.ACTIVATIONS],
                   variables_collections=[tf.GraphKeys.TRAINABLE_VARIABLES],
                   weights_initializer=tf.random_normal_initializer(stddev=0.02)):
        # use leaky ReLU on all body layers
        with arg_scope([conv2d], normalizer_fn=norm_fn,
                       normalizer_params=norm_params,
                       activation_fn=tf.nn.leaky_relu):

            nb = 1024 >> (nlevels - 1)
            for _ in range(nlevels):
                net = conv2d(net, nb, 5, 2)
                nb <<= 1
        assert net.shape.as_list()[1:4] == [4, 4, 1024]
        net = tf.reshape(net, [-1, 4 * 4 * 1024])
        net = fully_connected(net, noise_dims, activation_fn=None,
                              biases_initializer=None)
    if sphere_regularize:
        # add activation regularizer (to approach unit magnitude)
        add_unit_norm_loss(net, weight=1e-3, add_summary=add_summary and mode == 'TRAIN')

    assert(len(net.shape.as_list()) == 2)
    if add_summary:
        tf.summary.histogram("encoded_z", net)
    return net


def build_sndcgan_encoder(x: tf.Tensor, noise_dims: int, nlevels: int = 4, conv_activation_fn=tf.nn.leaky_relu,
                          batch_norm: bool = True, sphere_regularize: bool = False, add_summaries=False, mode=None):
    """Build an encoder based on SNDCGAN discriminator, presented in:
        Miyato et al. Spectral Normalization for generative adversarial networks (2018)

    No spectral normalization is employed, as that is meant to be used for the discriminator.
    
    The number of kernels, on the other hand, is defined according to:
        Kurach et al. The GAN Landscape: Losses, Architectures, Regularization, and Normalization (2018)

    Args:
      x : the real data [B, H, W, C]
      noise_dims : number of output channels
      nlevels : number of upsampling network levels (4 for 64x64, 5 for 128x128, etc.)
      conv_activation_fn : a function for the non-linearity operation after each hidden convolution layer
      batch_norm : whether to use batch normalization
      add_summaries : whether to produce summary operations
      mode : set to 'TRAIN' during the training process
    """
    if batch_norm:
        normalizer_fn = tf.contrib.layers.batch_norm
        normalizer_params = {'fused': True, 'scale': True, 'center': True}
    else:
        normalizer_fn = None
        normalizer_params = None

    fs = 4 << (nlevels - 3)

    net = x
    with arg_scope([fully_connected, conv2d, conv2d_t], outputs_collections=[tf.GraphKeys.ACTIVATIONS],
                   variables_collections=[tf.GraphKeys.TRAINABLE_VARIABLES],
                   weights_initializer=tf.random_normal_initializer(stddev=0.02)):
        # use leaky ReLU and pixelwise norm on all conv layers (except the head)
        with arg_scope([conv2d, conv2d_t], activation_fn=conv_activation_fn,
                       normalizer_fn=normalizer_fn, normalizer_params=normalizer_params):
            nb = 64
            for _ in range(3):
                net = conv2d(net, nb, 3, 1)
                nb = nb * 2
                net = conv2d(net, nb, 4, 2)

            net = conv2d(net, nb, 3, 1)
            assert net.shape.as_list()[1:] == [fs, fs, nb]
            net = tf.reshape(net, [-1, fs * fs * nb])
        net = fully_connected(net, noise_dims, activation_fn=None)

        if sphere_regularize:
            # add activation regularizer (to approach unit magnitude)
            add_unit_norm_loss(net, weight=2e-3, add_summary=add_summaries and mode == 'TRAIN')

    return net


class SNDCGANEncoder(tf.keras.Model):

    def __init__(self, channels_out: int = 512, nlevels: int = 4,
                 conv_activation=tf.nn.relu, batch_norm: bool = True,
                 sphere_regularize: bool = False):
        super(SNDCGANEncoder, self).__init__()
        self.channels_out = channels_out
        self.nlevels = nlevels
        self.conv_activation = conv_activation
        self.batch_norm = batch_norm
        self.sphere_regularize = sphere_regularize
    
    def call(self, inputs, training):
        if isinstance(inputs, list):
            inputs = inputs[0]
        return build_sndcgan_encoder(
            inputs, self.channels_out, self.nlevels, conv_activation_fn=self.conv_activation,
            batch_norm=self.batch_norm, sphere_regularize=self.sphere_regularize, add_summaries=training,
            mode='TRAIN' if training else 'PREDICT')


def autoencoder_mse(x, x_decoded, add_summary=False):
    out = tf.losses.mean_squared_error(x, x_decoded)
    if add_summary:
        tf.summary.scalar('autoencoder_mse', out)
    return out


def autoencoder_bce(x, x_decoded_logits, add_summary=False):
    out = tf.losses.sigmoid_cross_entropy(x, x_decoded_logits)
    if add_summary:
        tf.summary.scalar('autoencoder_bce', out)
    return out


def code_autoencoder_cosine(z, encoded_z, add_summary=False):
    # unit-normalize encoded_z (`z` is already unit-normal)
    encoded_z = tf.nn.l2_normalize(encoded_z, axis=1)
    out = tf.losses.cosine_distance(z, encoded_z, axis=1)
    if add_summary:
        tf.summary.scalar('autoencoder_cosine', out)
    return out


def code_autoencoder_mse_cosine(z, encoded_z, weight_factor=1.0, add_summary=False):
    return (
        autoencoder_mse(z, encoded_z, add_summary=add_summary) +
        code_autoencoder_cosine(
            z, encoded_z, add_summary=add_summary) * weight_factor
    )
