import tensorflow as tf
from tensorflow.contrib import gan as tfgan
from util import image_summaries_2level, random_hypersphere, upsample_2d, drift_loss, minibatch_stddev, pixelwise_feature_vector_norm

batch_norm = tf.contrib.layers.batch_norm
conv2d = tf.contrib.layers.conv2d
conv2d_t = tf.contrib.layers.conv2d_transpose
fully_connected = tf.contrib.layers.fully_connected
variance_scaling_initializer = tf.contrib.layers.variance_scaling_initializer
arg_scope = tf.contrib.framework.arg_scope

N_CHANNELS = 3

def conv_t_block(incoming: tf.Tensor, nb: int, ks=3):
    if isinstance(nb, int):
        nb = (nb, nb)
    if isinstance(ks, int):
        ks = (ks, ks)
    net = conv2d_t(incoming, nb[0], ks[0], stride=2)
    net = conv2d(net, nb[1], ks[1], stride=1)
    return net

def conv_block(incoming: tf.Tensor, nb: int, ks: int = 3, dropout: float = 0, dropout_istraining: bool = True):
    if isinstance(nb, int):
        nb = [nb, nb]
    net = conv2d(incoming, nb[0], ks, 2)
    net = conv2d(net, nb[1], ks, 1)
    if dropout:
        net = tf.layers.dropout(net, rate=dropout, training=dropout_istraining)
    return net

def build_dcgan_generator(z: tf.Tensor, conv_activation_fn=tf.nn.leaky_relu, batch_norm: bool = True, add_summaries=False, mode=None):
    """Build a generator based on DCGAN. Difference: this uses leaky ReLU by default.
    Args:
      z : the prior code Tensor [B, N]
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
            net = conv2d_t(net, 512, 5, 2) # Bx8x8x512
            net = conv2d_t(net, 256, 5, 2) # Bx16x16x256
            # comment the line below for 32x32 resolution
            net = conv2d_t(net, 128, 5, 2) # Bx32x32x128
        net = conv2d_t(net, N_CHANNELS, 5, 2, activation_fn=tf.nn.tanh, scope="y")
    return net

def build_1lvl_generator(z: tf.Tensor, mode=None) -> tf.Tensor:
    """"1-level generator
    Args:
      z : the prior code Tensor [B, N]
      mode : set to 'TRAIN' during the training process
    """
    net = z
    is_training = mode == 'TRAIN'
    with arg_scope([fully_connected, conv2d, conv2d_t], outputs_collections=[tf.GraphKeys.ACTIVATIONS],
                   variables_collections=[tf.GraphKeys.TRAINABLE_VARIABLES],
                   weights_initializer=tf.random_normal_initializer(stddev=0.02)):
        # use leaky ReLU and pixelwise norm on all conv layers (except the head)
        with arg_scope([conv2d, conv2d_t], activation_fn=tf.nn.leaky_relu, normalizer_fn=pixelwise_feature_vector_norm):
            net = fully_connected(
                net, 512 * 4 * 4, activation_fn=tf.nn.leaky_relu, scope="fc0")
            net = tf.layers.dropout(net, rate=0.5, training=is_training)
            net = tf.reshape(net, [-1, 4, 4, 512])  # Bx4x4x512
            #net = conv2d(net, 3, 1, 512)  # Bx4x4x512
            net = conv_t_block(net, [512, 256])  # Bx8x8x256
            net = conv_t_block(net, [256, 128])  # Bx16x16x128
            net = conv_t_block(net, [128, 64])  # Bx32x32x64
            net = conv_t_block(net, [64, 32])  # Bx64x64x32
        net = conv2d(net, N_CHANNELS, 1, 1, activation_fn=tf.nn.tanh, scope="y")
    return net

def build_generator_2lvl(z: tf.Tensor, mode=None) -> [tf.Tensor, tf.Tensor]:
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
            #net = conv2d(net, 3, 1, 512)  # Bx4x4x512
            net = conv_t_block(net, [512, 256])  # Bx8x8x256
            net = conv_t_block(net, [256, 128])  # Bx16x16x128
        net1 = conv2d(net, N_CHANNELS, 1, 1, activation_fn=tf.nn.tanh, scope="y_small") # Bx16x16x3

        with arg_scope(conv_scope):
            net = conv_t_block(net, [128, 64])  # Bx32x32x64
            net = conv_t_block(net, [64, 32])  # Bx64x64x32
        net2 = conv2d(net, N_CHANNELS, 1, 1, activation_fn=tf.nn.tanh, scope="y") # Bx64x64x3
    return [net2, net1]


def build_encoder(x: tf.Tensor, noise_dims: int, batch_norm=True, add_summary=True, sphere_regularize=False, mode=None) -> tf.Tensor:
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
            net = conv_block(net, [64, 128])  # Bx32x32x128
            net = conv_block(net, [128, 256])  # Bx16x16x256
            net = conv_block(net, [256, 512])  # Bx8x8x512
            net = conv_block(net, 512)  # Bx4x4x512
        assert  net.shape.as_list()[1:4] == [4, 4, 512]
        net = tf.reshape(net, [-1, 4 * 4 * 512])
        net = fully_connected(net, noise_dims, activation_fn=None,
                            biases_initializer=None)
    if sphere_regularize:
        # add activation regularizer (to approach unit magnitude)
        activation_reg = tf.multiply(tf.abs(tf.nn.l2_loss(net) - 1.0), 1e-3, name="z_reg")
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, activation_reg)
        if add_summary and mode == 'TRAIN':
            tf.summary.scalar('z_reg_loss', activation_reg)

    assert(len(net.shape.as_list()) == 2)
    if add_summary:
        tf.summary.histogram("encoded_z", net)
    return net

def build_dcgan_encoder(x: tf.Tensor, noise_dims: int, add_summary=True, batch_norm=True, sphere_regularize=False, mode=None) -> tf.Tensor:
    "build an encoder network which maps a sample to the latent space, based on DCGAN"

    # make it work for 2-level generators
    if isinstance(x, list):
        x = x[0] # use full size sample
    
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
            # comment the line below for 32x32 input resolution
            net = conv2d(net, 128, 5, 2)  # Bx32x32x128
            net = conv2d(net, 256, 5, 2) # Bx16x16x256
            net = conv2d(net, 512, 5, 2)  # Bx8x8x512
            net = conv2d(net, 1024, 5, 2)  # Bx4x4x1024
        net = tf.reshape(net, [-1, 4 * 4 * 1024])
        net = fully_connected(net, noise_dims, activation_fn=None,
                              biases_initializer=None)
    if sphere_regularize:
        # add activation regularizer (to approach unit magnitude)
        activation_reg = tf.multiply(tf.abs(tf.nn.l2_loss(net) - 1.0), 1e-3, name="z_reg")
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, activation_reg)
        if add_summary or mode == 'TRAIN':
            tf.summary.scalar('z_reg_loss', activation_reg)

    assert(len(net.shape.as_list()) == 2)
    if add_summary:
        tf.summary.histogram("encoded_z", net)
    return net

def code_autoencoder_mse(z, encoded_z, add_summary=True):
    out = tf.losses.mean_squared_error(z, encoded_z)
    if add_summary:
        tf.summary.scalar('autoencoder_mse', out)
    return out

def code_autoencoder_cosine(z, encoded_z, add_summary=True):
    # unit-normalize encoded_z (`z` is already unit-normal)
    encoded_z = tf.nn.l2_normalize(encoded_z, axis=1)
    out = tf.losses.cosine_distance(z, encoded_z, axis=1)
    if add_summary:
        tf.summary.scalar('autoencoder_cosine', out)
    return out

def code_autoencoder_mse_cosine(z, encoded_z, weight_factor=1.0, add_summary=True):
    return (
        code_autoencoder_mse(z, encoded_z, add_summary=add_summary) +
        code_autoencoder_cosine(z, encoded_z, add_summary=add_summary) * weight_factor
    )
