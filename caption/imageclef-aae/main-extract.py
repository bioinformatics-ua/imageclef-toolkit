#!/usr/bin/env python3
"""CLI program for extracting features out of a trained model."""

import argparse
from math import ceil
import os
import sys
import h5py as h5
import tensorflow as tf

from dataset_imagedir import get_dataset_dir_with_ids

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str,
                    help='Directory where the saved model is contained.')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--image_size', type=int, default=64)

parser.add_argument('--data_dir', default='CaptionTraining2018',
                    help='The directory containing the images to extract features.')
parser.add_argument('--out', '-o', default='features.h5',
                    help="The output file")
parser.add_argument('--out_file_list', '-f', default='file-list.txt',
                    help="The second output file (file names, in order)")
parser.add_argument('--no', action='store_true', default=False,
                    help="Don't extract features, just read model")
parser.add_argument('--debug', action='store_true', default=False,
                    help="Debug mode")

FLAGS = parser.parse_args()

with tf.Session(graph=tf.Graph()) as sess:
    graphdef = tf.saved_model.loader.load(
        sess, [tf.saved_model.tag_constants.SERVING], FLAGS.model_dir)

    signaturedef = graphdef.signature_def['encode']

    # 1. retrieve input and z tensor
    inputs = signaturedef.inputs
    xdef = inputs['x']

    outputs = signaturedef.outputs
    zdef = outputs['z']

    x = sess.graph.get_tensor_by_name(xdef.name)
    z = sess.graph.get_tensor_by_name(zdef.name)
    if FLAGS.debug:
        print('x:', x)
        print('z:', z)

    if FLAGS.no:
        sys.exit(0)

    dataset = get_dataset_dir_with_ids(
        FLAGS.data_dir, batch_size=FLAGS.batch_size, resizeto=FLAGS.image_size, cropto=None, shuffle=False, repeat=False)
    next_element = dataset.make_one_shot_iterator().get_next()

    n_samples = len(os.listdir(FLAGS.data_dir))
    n_batches = int(ceil(float(n_samples) / FLAGS.batch_size))
    id_list = open(FLAGS.out_file_list, mode='w')

    with h5.File(FLAGS.out, mode='w') as f:
        h5set = f.create_dataset('data', shape=[n_samples, z.shape[1]], dtype=float)
        h5ids = f.create_dataset('id', shape=[n_samples], dtype=h5.special_dtype(vlen=str))
        offset = 0
        i = 0
        j = 0
        while True:
            try:
                sys.stdout.write("\rFeature generation progress: {} of {}".format(i, n_batches))
                sys.stdout.flush()
                (ids, data) = sess.run(next_element)
                bsize = data.shape[0]
                z_features = sess.run(z, feed_dict={x: data})
                for l in ids:
                    l = l.decode('UTF-8')
                    id_list.write(l)
                    id_list.write("\n")
                    h5ids[j] = l
                    j += 1
                h5set[offset:offset + bsize] = z_features
                offset += bsize
                i += 1
            except tf.errors.OutOfRangeError:
                break

    id_list.close()
    print("\nFeatures saved in `{}`.".format(FLAGS.out))
