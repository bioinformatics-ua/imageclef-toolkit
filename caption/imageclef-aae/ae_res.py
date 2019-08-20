import tensorflow as tf
from tensorflow import keras
from tensorflow.contrib import gan as tfgan
from util import image_summaries_2level, random_hypersphere, upsample_2d, drift_loss, minibatch_stddev, pixelwise_feature_vector_norm, conv2d_sn, conv2d_t_sn, add_unit_norm_loss

arg_scope = tf.contrib.framework.arg_scope

N_CHANNELS = 3

class ResBlock(tf.keras.Model):

    def __init__(self, channels_out: int, ks: int = 3, activation='relu', batch_norm: bool = True):
        super(ResBlock, self).__init__()
        # 1. shortcut, [3, 3, 1], downsample, output = h/2 x w/2 x c_out
        # 2a. BN, ReLU, output = h x w x c_in
        # 2b. Conv, [3, 3, 1], output = h x w x c_out
        # 2c. BN, ReLU
        # 2d. Conv, [3, 3, 1], downsample, output = h/2 x w/2 x c_out
        # 3. addition
        self.shortcut = keras.layers.Conv2D(
            channels_out, ks, 2, padding='same', activation=None)
        self.batch_norm = batch_norm
        if batch_norm:
            [self.bn0, self.bn1, self.bn2] = [
                keras.layers.BatchNormalization() for i in range(3)]

        [self.act1, self.act2] = [
            keras.layers.Activation(activation) for i in range(2)]
        self.conv1 = keras.layers.Conv2D(
            channels_out, ks, 1, padding='same', activation=None)
        self.conv2 = keras.layers.Conv2D(
            channels_out, ks, 2, padding='same', activation=None)

    def call(self, inputs, training):
        assert len(inputs.shape.as_list()) == 4
        shortcut = self.shortcut(inputs)
        net = inputs
        if self.batch_norm:
            net = self.bn0(net, training=training)
        net = self.act1(net)
        net = self.conv1(net)
        if self.batch_norm:
            net = self.bn1(net, training=training)
        net = self.act2(net)
        net = self.conv2(net)
        if self.batch_norm:
            net = self.bn2(net, training=training)
        return shortcut + net


class ResTransposeBlock(tf.keras.Model):

    def __init__(self, channels_out: int, ks: int = 3, activation='relu', batch_norm: bool = True):
        super(ResTransposeBlock, self).__init__()
        # 1. shortcut ConvTranspose, [3, 3, 1], upsample, output = h*2 x w*2 x c_out
        # 2a. BN, ReLU, output = h x w x c_in
        # 2b. Conv, [3, 3, 1], output = h x w x c_out
        # 2c. BN, ReLU
        # 2d. ConvTranspose, [3, 3, 1], downsample, output = h*2 x w*2 x c_out
        # 3. addition
        self.shortcut = keras.layers.Conv2DTranspose(
            channels_out, ks, 2, padding='same', activation=None)
        self.batch_norm = batch_norm
        if batch_norm:
            [self.bn0, self.bn1, self.bn2] = [
                keras.layers.BatchNormalization() for i in range(3)]

        [self.act1, self.act2] = [
            keras.layers.Activation(activation) for i in range(2)]
        self.conv1 = keras.layers.Conv2D(
            channels_out, ks, 1, padding='same', activation=None)
        self.conv2 = keras.layers.Conv2DTranspose(
            channels_out, ks, 2, padding='same', activation=None)
        self.out = keras.layers.Add()

    def call(self, inputs, training):
        assert len(inputs.shape.as_list()) == 4
        shortcut = self.shortcut(inputs)
        if self.batch_norm:
            net = self.bn0(inputs, training=training)
        net = self.act1(net)
        net = self.conv1(net)
        if self.batch_norm:
            net = self.bn1(net, training=training)
        net = self.act2(net)
        net = self.conv2(net)
        if self.batch_norm:
            net = self.bn2(net, training=training)
        return self.out([shortcut, net])


class ResGanGenerator(tf.keras.Model):
    """ResNet19 generator"""

    def __init__(self, channels_out: int = N_CHANNELS, nlevels: int = 4,
                 fc_activation='relu', conv_activation='relu', last_activation='tanh',
                 batch_norm: bool = True):
        super(ResGanGenerator, self).__init__()
        self.fc0 = tf.keras.layers.Dense(512 * 4 * 4, activation=fc_activation)
        self.fc_reshape = tf.keras.layers.Reshape((4, 4, 512))
        self.blocks = [
            ResTransposeBlock(
                512 >> (i // 2), activation=conv_activation, batch_norm=batch_norm)
            for i in range(1, nlevels + 1)
        ]
        self.bn_last = keras.layers.BatchNormalization() if batch_norm else None
        self.act_last = keras.layers.Activation(conv_activation)
        self.convlast = keras.layers.Conv2DTranspose(
            channels_out, kernel_size=3, padding='same', activation=last_activation)

    def call(self, inputs, training):
        if isinstance(inputs, list):
            inputs = inputs[0]
        net = inputs
        net = self.fc0(net)
        net = self.fc_reshape(net)
        for resblock in self.blocks:
            net = resblock(net, training=training)
        return self.convlast(net)


class ResGanEncoder(tf.keras.Model):

    def __init__(self, channels_out: int = 512, nlevels: int = 4, conv_activation='relu', last_activation=None, batch_norm: bool = True, act_regularizer: bool = None):
        assert nlevels >= 4, "Only nlevels >= 4 are currently supported"
        super(ResGanEncoder, self).__init__()

        self.blocks = [
            ResBlock(64, activation=conv_activation, batch_norm=batch_norm),
            ResBlock(128, activation=conv_activation, batch_norm=batch_norm),
            ResBlock(256, activation=conv_activation, batch_norm=batch_norm),
            ResBlock(256, activation=conv_activation, batch_norm=batch_norm),
        ]
        for _ in range(4, nlevels):
            self.blocks.append(
                ResBlock(512, activation=conv_activation, batch_norm=batch_norm))
        self.blocks.append(
            ResBlock(
                channels_out, activation=conv_activation, batch_norm=batch_norm)
        )
        self.act_reg = act_regularizer
        self.act_last = keras.layers.Activation(last_activation)
        self.mp = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, inputs, training):
        if isinstance(inputs, list):
            inputs = inputs[0]
        net = inputs
        for resblock in self.blocks:
            net = resblock(net, training=training)
        if training and self.act_reg is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.act_reg(net))
        net = self.mp(net)
        net = self.act_last(net)
        return net


class ResGanDiscriminator(ResGanEncoder):

    def __init__(self, channels_hidden: int = 512,
                 nlevels: int = 4, conv_activation='relu',
                 batch_norm: bool = True):
        super(ResGanDiscriminator, self).__init__(channels_hidden,
                                                  nlevels, conv_activation=conv_activation, batch_norm=batch_norm)
        self.y_out = keras.layers.Dense(1, activation=None)

    def call(self, inputs, training):
        net = super(ResGanDiscriminator, self).call(inputs, training)
        return self.y_out(net, training=training)
