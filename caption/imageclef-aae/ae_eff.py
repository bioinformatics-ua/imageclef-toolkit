import efficientnet
from tensorflow import keras

class EfficientNetB0Encoder(tf.keras.Model):
        def __init__(self, nlevels: int = 4, last_activation=None, act_regularizer: bool = None):
        assert nlevels >= 4, "Only nlevels >= 4 are currently supported"
        super(EfficientNetB0Encoder, self).__init__()

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
