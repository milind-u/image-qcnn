import cirq
import tensorflow_quantum as tfq

import cv2 as cv
import numpy as np
import collections


class ImageEncoder:
    """ Filters images to just 3s and 6s and downsamples to 4 by 4. """

    IMAGE_LEN = 4

    def _filter_digits(self, x, y):
        # Filter to only 3s and 6s, and convert outputs to 0 or 1
        keep = ((y == 3) | (y == 6))
        x, y = x[keep], y[keep]
        for i in range(len(y)):
            y[i] = (0 if y[i] == 3 else 1)
        return x, y

    def encode_dataset(self, x_train, y_train, x_test, y_test):
        x_train_encoded, y_train_encoded = self._encode_images(
            x_train, y_train)
        x_test_encoded, y_test_encoded = self._encode_images(x_test, y_test)
        x_test_encoded, y_test_encoded = self._remove_train_test_duplicate(
            x_train_encoded, y_train_encoded, x_test_encoded, y_test_encoded)
        return (x_train_encoded, y_train_encoded), (x_test_encoded,
                                                    y_test_encoded)

    def _encode_images(self, x, y):
        x, y = self._filter_digits(x, y)

        # Resize the images to fit in a reasonable number of qubits
        x_resized = []
        for i in range(x.shape[0]):
            resized = cv.resize(
                x[i], (ImageEncoder.IMAGE_LEN, ImageEncoder.IMAGE_LEN))
            x_resized.append(resized)
        x_resized = np.array(x_resized)

        return x_resized, y

    def _remove_train_test_duplicate(self, x_train, y_train, x_test, y_test):
        # Removes test points that have the same input as train ones.
        # Needed because downsampling images causes duplicates

        initial_len = len(x_test)
        print(initial_len)
        
        i = 0
        while i < len(x_test):
            for x in x_train:
                if np.array_equal(x_test[i], x):
                    # Remove the test point because it's in train too
                    x_test = np.delete(x_test, i, axis=0)
                    y_test = np.delete(y_test, i)
                    i -= 1
                    break
            i += 1

        print("Removed %d duplicate train test inputs from the test datapoints" % (initial_len - len(x_test)))

        return x_test, y_test


class QuantumImageEncoder(ImageEncoder):
    """ On top of the classical encoding, encodes the processed images into qubits,
        using amplitude encoding. """

    # Total number of pixels in the square image
    INPUT_VALUES = ImageEncoder.IMAGE_LEN**2

    # Pad number of qubits to a power of 2
    NUM_QUBITS = int(2**np.ceil(np.log2(INPUT_VALUES)))

    def encode_dataset(self, x_train, y_train, x_test, y_test):
        (x_train_classical,
         y_train_classical), (x_test_classical,
                              y_test_classical) = super().encode_dataset(
                                  x_train, y_train, x_test, y_test)

        x_train_quantum, y_train_quantum = self._quantum_encode_images(
            x_train_classical, y_train_classical)
        x_test_quantum, y_test_quantum = self._quantum_encode_images(
            x_test_classical, y_test_classical)

        return (x_train_quantum, y_train_quantum), (x_test_quantum,
                                                    y_test_quantum)

    def _quantum_encode_images(self, x, y):
        x_quantum = []
        for image in x:
            circ = cirq.Circuit()
            qubits = cirq.GridQubit.rect(1, QuantumImageEncoder.NUM_QUBITS)

            count = 0
            qubit_index = 0

            for row in image:
                for pixel in row:
                    # Rotate qubits by the grayscale pixel values
                    circ += cirq.rx(pixel)(qubits[qubit_index])
                    count += 1
                    qubit_index += 1

            assert qubit_index == QuantumImageEncoder.INPUT_VALUES

            x_quantum.append(circ)

        return tfq.convert_to_tensor(x_quantum), y
