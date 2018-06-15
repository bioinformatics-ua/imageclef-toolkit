"""Module for fetching images from the ImageCLEF caption data set."""

import os
import tensorflow as tf

def get_dataset(filename="train_files.txt", batch_size=32, resizeto=None, cropto=None, normalize=True, shuffle=True,
                repeat=True, buffer=1024):
    dset = tf.data.TextLineDataset(filename)
    return _build_dataset(dset, batch_size, resizeto, cropto, normalize, shuffle, repeat, buffer)

def get_dataset_dir(dirpath="CaptionTraining2018", batch_size=32, resizeto=None, cropto=None, normalize=True,
                    shuffle=True, repeat=True, buffer=1024):
    dset = tf.data.Dataset.list_files(dirpath + "/*", shuffle=shuffle)
    return _build_dataset(dset, batch_size, resizeto, cropto, normalize, shuffle=False, repeat=repeat, buffer=buffer)

def get_dataset_dir_with_ids(dirpath="CaptionTraining2018", batch_size=32, resizeto=None, cropto=None, normalize=True,
                    shuffle=True, repeat=True, buffer=1024):
    return _build_dataset_with_ids(dirpath, extension='.png', batch_size=batch_size, resizeto=resizeto, cropto=cropto,
                                   normalize=normalize, shuffle=shuffle, repeat=repeat, buffer=buffer)

def _build_dataset(filepaths_dataset, batch_size=32, resizeto=None, cropto=None, normalize=True,
                  shuffle=True, repeat=True, buffer=1024):
    dset = filepaths_dataset
    if shuffle:
        if repeat:
            dset = dset.apply(tf.contrib.data.shuffle_and_repeat(buffer))
        else:
            dset = dset.shuffle(buffer, reshuffle_each_iteration=True)

    if isinstance(resizeto, int):
        resizeto = [resizeto, resizeto]

    if isinstance(cropto, int):
        cropto = [cropto, cropto, 3]

    if resizeto == cropto:
        cropto = None # ignore cropping

    dset = dset.map(lambda x: tf.image.decode_image(tf.read_file(x), channels=3))
    if resizeto:
        if cropto:
            dset = dset.map(lambda x: tf.image.resize_bilinear([x], resizeto)[0])
        else:
            dset = dset.apply(tf.contrib.data.map_and_batch(lambda x: tf.image.resize_bilinear([x], resizeto)[0], batch_size, num_parallel_batches=2))
    if cropto:
        dset = dset.apply(tf.contrib.data.map_and_batch(lambda x: tf.random_crop(x, cropto), batch_size, num_parallel_batches=2))
    if normalize:
        dset = dset.map(lambda x: tf.to_float(x) / 127.5 - 1.0)
    if buffer:
        dset = dset.prefetch(buffer)

    return dset

def _build_dataset_with_ids(dirpath, extension='.png', batch_size=32, resizeto=None, cropto=None, normalize=True,
                  shuffle=True, repeat=True, buffer=1024):
    ids = [
        f[:-4] for f in os.listdir(dirpath) if f.endswith(extension)
    ]
    dset_ids = tf.data.Dataset.from_tensor_slices(ids).batch(batch_size)
    filepaths = [
        os.path.join(dirpath, f + extension) for f in ids
    ]
    dset = tf.data.Dataset.from_tensor_slices(filepaths)

    dset = _build_dataset(dset, batch_size=batch_size, resizeto=resizeto, cropto=cropto, normalize=normalize,
                          shuffle=shuffle, repeat=repeat, buffer=buffer)
    # convert to dataset of pairs
    return tf.data.Dataset.zip((dset_ids, dset))
