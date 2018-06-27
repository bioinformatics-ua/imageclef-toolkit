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
from ae import code_autoencoder_mse, code_autoencoder_mse_cosine, build_dcgan_generator, build_dcgan_encoder, build_encoder, build_1lvl_generator, build_dcgan_generator
from faae import build_faae_harness
from aae import build_code_discriminator, build_aae_harness
from aae_train import aegan_model, aegan_train_ops
from plain_gan import build_gan_harness, build_dcgan_discriminator, build_discriminator_1lvl, build_discriminator_2lvl
from AMSGrad import AMSGrad

tf.logging.set_verbosity(tf.logging.INFO)

# --------------------------- CONSTANTS ---------------------------

# Type: either "FAAE" (flipped-Adversarial Autoencoder), "AAE" (Adversarial Autoencoder), or "GAN" (no encoder)
TYPE = "GAN"
# The aversarial training loss ("WASSERSTEIN" for GP-WGAN, anything else is `modified_loss`)
ADVERSARIAL_TRAINING = "MODIFIED"
# The run key to be used
KEY = "{}{}_dcgan_bn_sphere_reg_RUN1".format(
    TYPE, '_W' if ADVERSARIAL_TRAINING == "WASSERSTEIN" else "")
# Training data set dir
DATA_DIR = "CaptionTraining2018"
# TensorBoard log dir
LOG_DIR = "summaries/{}".format(KEY)
# Where to save the model in the end (`KEY` is appended automatically)
SAVED_MODEL_DIR = "saved_model"
# XLA optimization
XLA_JIT = False  # this might not work at all
# Whether to just save the model and exit
ONLY_SAVE = False
# Debug mode
DEBUG = False
# Batch size
BATCH_SIZE = 32
# The number of dimensions of the prior/latent code
NOISE_DIMS = 512
# The random distribution of the prior code (choices: "SPHERE", "UNIFORM", "NORMAL")
NOISE_FORMAT = "SPHERE"
# The real/generated image size in pixels
IMAGE_SIZE = 64
# The extra margin to consider in random cropping: image will be first
# resized to the size `IMAGE_SIZE + CROP_MARGIN_SIZE` 
CROP_MARGIN_SIZE = 8

# Number of channels in the image
N_CHANNELS = 3
# Estimated number of epochs based on batch size and training data size
# (should be updated based on batch size and training set size)
STEPS_PER_EPOCH = 223859 // BATCH_SIZE
# Number of epochs to train
NUM_EPOCHS = 100
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
GENERATOR_FN = build_dcgan_generator
DISCRIMINATOR_FN = build_discriminator_1lvl
ENCODER_FN = build_encoder
# -----------------------------------------------------------------


if path.exists(SAVED_MODEL_DIR + "/" + KEY):
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
            sample = GENERATOR_FN(z_input, mode='PREDICT')
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


print("Loading up data set...")

# Set up the inputs
dataset = get_dataset_dir(DATA_DIR, batch_size=BATCH_SIZE, resizeto=IMAGE_SIZE + CROP_MARGIN_SIZE,
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
        NOISE_FORMAT, adversarial_training=ADVERSARIAL_TRAINING, no_trainer=ONLY_SAVE)
elif TYPE == 'FAAE':
    (model, loss, rec_loss, train) = build_faae_harness(
        image_input, noise, GENERATOR_FN, DISCRIMINATOR_FN, ENCODER_FN,
        NOISE_FORMAT, adversarial_training=ADVERSARIAL_TRAINING, no_trainer=ONLY_SAVE)
elif TYPE == 'GAN':
    (model, loss, train) = build_gan_harness(
        image_input, noise, GENERATOR_FN, DISCRIMINATOR_FN,
        NOISE_FORMAT, adversarial_training=ADVERSARIAL_TRAINING, no_trainer=ONLY_SAVE)
else:
    print("Invalid network type", TYPE)
    exit(-1)

if ONLY_SAVE:
    save()
    exit(0)

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

save()
