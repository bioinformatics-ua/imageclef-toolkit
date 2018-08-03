#!/usr/bin/env python3
"""Trainer of GANs with an auto-encoding process."""

from os import makedirs, path
from sys import exit
import tensorflow as tf
from tensorflow.python import debug as tf_debug  # pylint: disable=E0611
from tensorflow import saved_model
from tensorflow.contrib import gan as tfgan
from dataset_imagedir import get_dataset_dir
from util import image_summaries_generated, image_grid_summary, random_noise, upsample_2d, drift_loss, minibatch_stddev, pixelwise_feature_vector_norm
from ae import build_dcgan_generator, build_sndcgan_generator, build_sndcgan_encoder, build_encoder, build_1lvl_generator, build_dcgan_generator
from faae import build_faae_harness
from aae import build_code_discriminator, build_aae_harness
from aae_train import aegan_model, aegan_train_ops
from plain_gan import build_gan_harness, build_dcgan_discriminator, build_discriminator_1lvl, build_discriminator_2lvl
from AMSGrad import AMSGrad

tf.logging.set_verbosity(tf.logging.INFO)

# --------------------------- CONSTANTS ---------------------------

# Type: either "FAAE" (flipped-Adversarial Autoencoder), "AAE" (Adversarial Autoencoder), "VAEGAN" (VAE - GAN), or "GAN" (no encoder)
TYPE = "AAE"
# The aversarial training loss: "WASSERSTEIN" for GP-WGAN, "RELATIVISTIC_AVG" for RaSGAN, anything else is NS loss (`modified_loss`)
ADVERSARIAL_TRAINING = "RELATIVISTIC_AVG"
LOSS_TYPE_SUFFIX = {
    "WASSERSTEIN": "_W",
    "RELATIVISTIC_AVG": "_RaS"
}.get(ADVERSARIAL_TRAINING) or ""
# The run key to be used
KEY = "{}{}_RUN1".format(TYPE, LOSS_TYPE_SUFFIX)
# Training data set dir
DATA_DIR = "CaptionTraining2018"
# Testing data set dir
TEST_DATA_DIR = "CaptionTesting2018"
# TensorBoard log dir
LOG_DIR = "summaries/{}".format(KEY)
# Where to save the model in the end (`KEY` is appended automatically)
SAVED_MODEL_DIR = "saved_model"
# XLA optimization
XLA_JIT = False  # this might not work at all
# Whether to train the GAN
DO_TRAIN = True
# Whether to save the GAN's encoder and generator as a TF saved model
DO_SAVE = True
# Whether to evaluate the GAN in the end
DO_EVAL = False
# Debug mode
DEBUG = False
# Batch size
BATCH_SIZE = 32
# The number of dimensions of the prior/latent code
NOISE_DIMS = 1024
# The random distribution of the prior code (choices: "SPHERE", "UNIFORM", "NORMAL")
NOISE_FORMAT = "SPHERE"
# The real/generated image size in pixels
IMAGE_SIZE = 64
# The extra margin to consider in random cropping: image will be first
# resized to the size `IMAGE_SIZE + CROP_MARGIN_SIZE` 
CROP_MARGIN_SIZE = 8

# Number of channels in the images
N_CHANNELS = 3
# Estimated number of steps per epoch based on batch size and training data size
# (should be updated based on batch size and training set size)
STEPS_PER_EPOCH = 223859 // BATCH_SIZE
# Number of epochs to train
NUM_EPOCHS = 50
# Total number of steps to train
NUM_STEPS = STEPS_PER_EPOCH * NUM_EPOCHS
# Number of reconstruction steps per iteration
R_STEPS = 1
# Number of generator training steps per iteration
G_STEPS = 1
# Number of discriminator/critic training steps per iteration
D_STEPS = 1
# ---
# These constants point to network building functions with a custom prototype.
# More functions which can be assigned here are available in the modules `ae.py`, `aae.py` and `plain_gan.py`.
GENERATOR_FN = build_sndcgan_generator
DISCRIMINATOR_FN = build_code_discriminator
ENCODER_FN = build_sndcgan_encoder

# the number of levels of each network so that they can handle
# samples of the intended resolution (4 levels = 64x64, more levels means more resolution).
NUM_NETWORK_LEVELS = {
    16: 2,
    32: 3,
    64: 4,
    128: 5,
    256: 6,
    512: 7,
    1024: 8
}[IMAGE_SIZE]
# -----------------------------------------------------------------


if DO_SAVE and path.exists(SAVED_MODEL_DIR + "/" + KEY):
    print("Saved model directory {}/{} already exists! Please move or remove before continuing.".format(
        SAVED_MODEL_DIR, KEY))
    exit(-1)


def save(export_dir=None, generator_scope=None, encoder_scope=None):
    """Save the trained model to disk as a TensorFlow saved model."""
    read = tf.train.Saver()
    with tf.Session(graph=tf.get_default_graph()) as sess:
        read.restore(sess, tf.train.latest_checkpoint(LOG_DIR))

        export_dir = export_dir or (SAVED_MODEL_DIR + "/" + KEY)
        builder = saved_model.builder.SavedModelBuilder(export_dir)

        generator_scope = tf.variable_scope(
            generator_scope or "Generator", reuse=True)
        encoder_scope = tf.variable_scope(
            encoder_scope or "Encoder", reuse=True)

        defmap = {}

        if TYPE != "GAN":
            # basic encoder signature
            with encoder_scope:
                x_input = tf.placeholder(
                    tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS])
                encoded_z = ENCODER_FN(
                    x_input, NOISE_DIMS, mode='PREDICT')
            encode_signature_inputs = {
                "x": saved_model.utils.build_tensor_info(x_input)
            }
            encode_signature_outputs = {
                "z": saved_model.utils.build_tensor_info(encoded_z)
            }
            defmap['encode'] = saved_model.signature_def_utils.build_signature_def(
                encode_signature_inputs, encode_signature_outputs, 'encode')

        # basic generator signature
        with generator_scope:
            z_input = tf.placeholder(tf.float32, shape=[None, NOISE_DIMS])
            sample = GENERATOR_FN(z_input, nchannels=N_CHANNELS, nlevels=NUM_NETWORK_LEVELS, mode='PREDICT')
        generate_signature_inputs = {
            "z": saved_model.utils.build_tensor_info(z_input)
        }
        generate_signature_outputs = {
            "x": saved_model.utils.build_tensor_info(sample)
        }
        defmap['generate'] = saved_model.signature_def_utils.build_signature_def(
            generate_signature_inputs, generate_signature_outputs,
            'generate')

        builder.add_meta_graph_and_variables(
            sess, [saved_model.tag_constants.SERVING],
            signature_def_map=defmap)
        builder.save(as_text=False)

def gan_eval(gan_model, test_data_dir, num_batches=16):
    read = tf.train.Saver()

    def adapt_to_inception(x):
        # convert from grayscale to RGB
        if x.shape.as_list()[3] == 1:
            x = x[:, :, :, 0]
            x = tf.stack([x, x, x], axis=3)
        # resize
        if x.shape.as_list()[1:3] != [299, 299]:
            x = tf.image.resize_bilinear(x, [299, 299])
        return x

    with tf.Session(graph=tf.get_default_graph()) as sess:
        print("Evaluating GAN...")

        test_dataset = get_dataset_dir(test_data_dir, batch_size=BATCH_SIZE, nchannels=3, resizeto=299,
                                       normalize=True, shuffle=False)
        real_data = test_dataset.make_one_shot_iterator().get_next()
        
        generated_outputs = adapt_to_inception(gan_model.generated_data)
        batches = [sess.run(real_data) for _ in range(num_batches)]
        real_samples = tf.concat(batches, axis=0)

        read.restore(sess, tf.train.latest_checkpoint(LOG_DIR))
        batches = [sess.run(generated_outputs) for _ in range(num_batches)]
        gen_samples = tf.concat(batches, axis=0)

        inception = tfgan.eval.inception_score(gen_samples, num_batches=num_batches)
        inception_val = sess.run(inception)
        print("Inception score:", inception_val)

        fid = tfgan.eval.frechet_inception_distance(real_samples, gen_samples, num_batches=num_batches)
        fid_val = sess.run(fid)
        print("Frechet Inception Distance:", fid_val)
        return (inception_val, fid_val)

print("Loading up data set...")

# Set up the inputs
dataset = get_dataset_dir(DATA_DIR, batch_size=BATCH_SIZE, nchannels=N_CHANNELS, resizeto=IMAGE_SIZE + CROP_MARGIN_SIZE,
                      cropto=IMAGE_SIZE, normalize=True, shuffle=True)
image_input = dataset.make_one_shot_iterator().get_next()
image_input.set_shape([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS])
# Set up the prior codes
noise = random_noise([BATCH_SIZE, NOISE_DIMS], NOISE_FORMAT)
tf.contrib.layers.summarize_tensor(noise)

profile_path = 'profile/{}{}'.format('jit/' if XLA_JIT else '', KEY)
makedirs(profile_path, exist_ok=True)

if TYPE == 'AAE':
    (model, loss, rec_loss, train) = build_aae_harness(
        image_input, noise, GENERATOR_FN, DISCRIMINATOR_FN, ENCODER_FN,
        noise_format=NOISE_FORMAT, adversarial_training=ADVERSARIAL_TRAINING, no_trainer=not DO_TRAIN)
elif TYPE == 'FAAE':
    (model, loss, rec_loss, train) = build_faae_harness(
        image_input, noise, GENERATOR_FN, DISCRIMINATOR_FN, ENCODER_FN,
        noise_format=NOISE_FORMAT, adversarial_training=ADVERSARIAL_TRAINING, no_trainer=not DO_TRAIN)
elif TYPE == 'GAN':
    (model, loss, train) = build_gan_harness(
        image_input, noise, GENERATOR_FN, DISCRIMINATOR_FN,
        noise_format=NOISE_FORMAT, adversarial_training=ADVERSARIAL_TRAINING, no_trainer=not DO_TRAIN)
else:
    print("Invalid network type", TYPE)
    exit(-1)

if DO_TRAIN:
    writer = tf.summary.FileWriter(LOG_DIR)

    # console logging hook
    if TYPE == 'GAN':
        log_hook = tf.train.LoggingTensorHook({
            'generator_loss': loss.generator_loss,
            '{}_loss'.format(
                'critic' if ADVERSARIAL_TRAINING == 'WASSERSTEIN'
                else 'discriminator'): loss.discriminator_loss
        }, every_n_iter=10)
    else:
        log_hook = tf.train.LoggingTensorHook({
            'generator_loss': loss.generator_loss,
            '{}_loss'.format(
                'critic' if ADVERSARIAL_TRAINING == 'WASSERSTEIN'
                else 'discriminator'): loss.discriminator_loss,
            'reconstruction_loss': rec_loss
        }, every_n_iter=10)

    train_hooks = [
        log_hook,
        tf.train.StopAtStepHook(num_steps=NUM_STEPS),
    ]

    if DEBUG:
        train_hooks.append(
            tf_debug.TensorBoardDebugHook('localhost:6064')
        )

    r_steps = 0 if TYPE == "GAN" else R_STEPS

    def get_sequential_train_hooks(rec_steps: int = 1, disc_steps: int = 1, gen_steps: int = 1):
        """Returns a hooks function for sequential auto-encoding GAN training.

        Args:
        rec_steps: how many reconstruction steps to take
        disc_steps: how many discriminator training steps to take.
        gen_steps: how many generator steps to take

        Returns:
        A function that takes an AEGANTrainOps tuple and returns a list of hooks.
        """
        assert rec_steps >= 0
        assert disc_steps > 0
        assert gen_steps > 0
        def get_hooks(train_ops):
            discriminator_hook = tfgan.RunTrainOpsHook(train_ops.discriminator_train_op,
                                                    disc_steps)
            generator_hook = tfgan.RunTrainOpsHook(train_ops.generator_train_op,
                                                gen_steps)
            if rec_steps:
                reconstruction_hook = tfgan.RunTrainOpsHook(train_ops.rec_train_op,
                                                            rec_steps)
                return [reconstruction_hook, discriminator_hook, generator_hook]
            return [discriminator_hook, generator_hook]
        return get_hooks

    print("Training process begins: {} steps".format(NUM_STEPS))
    config = tf.ConfigProto()
    tfgan.gan_train(
        train,
        logdir=LOG_DIR,
        get_hooks_fn=get_sequential_train_hooks(r_steps, D_STEPS, G_STEPS),
        hooks=train_hooks,
        save_summaries_steps=100,
        save_checkpoint_secs=1200,
        config=config)

if DO_SAVE:
    save()

if DO_EVAL:
    if TEST_DATA_DIR is None:
        print("Testing set directory required for GAN evaluation")
    else:
    gan_eval(model, TEST_DATA_DIR)
