#!/usr/bin/env python3

import encoding
import models

import absl
import matplotlib.pyplot as plt
import tensorflow as tf

absl.flags.DEFINE_boolean("train_quantum", True,
                          "Whether to train a Quantum CNN.")
absl.flags.DEFINE_boolean("train_classical", True,
                          "Whether to train a Classical CNN.")
FLAGS = absl.flags.FLAGS


def main(_):
    # Use the mnist handwritten digits dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if FLAGS.train_quantum:
        qcnn = models.QuantumCnn(x_train, y_train, x_test, y_test)
        qcnn.train()
        qcnn.plot()
    if FLAGS.train_classical:
        nn = models.ClassicalNn(x_train, y_train, x_test, y_test)
        nn.train()
        nn.plot()

    plt.show()


if __name__ == "__main__":
    absl.app.run(main)
