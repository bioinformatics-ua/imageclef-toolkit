# Adversarial Auto-Encoding GANs

> Implementations of generative adversarial networks with an auto-encoding process.

This code reproduces an unsupervised representation learning approach for the ImageCLEF 2018 concept detection challenge. It was designed to support training on an arbitrary set of images, and contains three kinds of GANs:

- Flipped-Adversarial Autoencoder (FAAE)
- Adversarial Autoencoder (AAE)
- A plain Generative Adversarial Network (GAN), no encoder

## Requirements

- Python 3.6
- TensorFlow 1.7 or greater (tested up to 1.9)
- `h5py` for saving the extracted features

## Usage

[main.py](main.py) is the executable for training the GAN. Please see the various constants and functions declared at the beginning of the file to configure the networks and training processes to shape. By default, the program will fetch images from a directory named "CaptionTraining2018". Summaries and checkpoints will be produced automatically in "./summaries/". At the end of training, the full model is saved in "./saved_model".

The default hyperparameters, regularization methods and network components (generators, encoders and discriminator architectures used) in this repository may not be an exact match of those used in the original ImageCLEF 2018 submissions, but should enable users to attain a similar, potentially better representation quality.

[main-extract.py](main-extract.py) is a CLI application that fetches a TensorFlow saved model (as produced by the trainer in this project) and extracts the latent codes from multiple images, while saving them to an HDF5 file. Please run `python main-extract.py --help` for a more extensive documentation.
