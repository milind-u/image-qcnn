import encoding

import abc

import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import cirq.contrib.svg


import sympy
import numpy as np

import matplotlib.pyplot as plt


class Model(abc.ABC):
    """ Abstract class that wraps a tf.keras.Model, encoding and plugging in the image data. """
    def __init__(self, x_train, y_train, x_test, y_test):
        # Encode the inputs
        encoder = self._make_encoder()
        (self.x_train, self.y_train), (self.x_test,
                                       self.y_test) = encoder.encode_dataset(
                                           x_train, y_train, x_test, y_test)

        # Will store history for the option to plot
        self.history = None

        # Compile the model
        self.model = self._make_model()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
            loss=tf.keras.losses.mse,
            metrics=[self._get_accuracy()])
        print(self.model.summary())

    @abc.abstractmethod
    def _make_encoder(self) -> encoding.ImageEncoder:
        """ Creates an encoder to encode the input images """
        pass

    @abc.abstractmethod
    def _make_model(self) -> tf.keras.Model:
        """ Builds an uncompiled model for image classification """
        pass

    @abc.abstractmethod
    def _get_accuracy(self):
        """ Returns the accuracy function for model fitting """
        pass

    def train(self):
        """ Trains the model with the encoded data """
        self.history = self.model.fit(x=self.x_train,
                                      y=self.y_train,
                                      batch_size=50,
                                      epochs=5,
                                      verbose=1,
                                      validation_data=(self.x_test,
                                                       self.y_test))

    def plot(self):
        """ Plots accuracy as a function of epochs. Must first run train() """
        plt.figure()
        plt.plot(self.history.history["accuracy"][1:], label="Training")
        plt.plot(self.history.history["val_accuracy"][1:], label="Test")
        plt.title("%s accuracy" % self.__class__.__name__)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()


class QuantumCnn(Model):
    """ Quantum CNN to classify images. """

    CONVOLUTION_PARAMS = 15
    POOL_PARAMS = 6
    # Number of tunable parameters per convolution and pool layer
    PARAMS_PER_LAYER = CONVOLUTION_PARAMS + POOL_PARAMS
    # Number of QCNN iterations
    NUM_LAYERS = int(np.log2(encoding.QuantumImageEncoder.NUM_QUBITS))
    # Total number of tunable paramters
    NUM_PARAMS = PARAMS_PER_LAYER * NUM_LAYERS

    def _make_encoder(self) -> encoding.ImageEncoder:
        return encoding.QuantumImageEncoder()

    def _one_qubit_unitary(self, q, symbols):
        # Rotating qubit about all 3 axes by symbol values
        return cirq.Circuit(
            cirq.X(q)**symbols[0],
            cirq.Y(q)**symbols[1],
            cirq.Z(q)**symbols[2])

    def _two_qubit_unitary(self, q1, q2, symbols):
        # Rotating both qubits about all 3 axes by symbol values
        circ = cirq.Circuit()
        circ += self._one_qubit_unitary(q1, symbols[:3])
        circ += self._one_qubit_unitary(q2, symbols[3:6])

        circ += cirq.ZZ(q1, q2)**symbols[6]
        circ += cirq.YY(q1, q2)**symbols[7]
        circ += cirq.XX(q1, q2)**symbols[8]

        circ += self._one_qubit_unitary(q1, symbols[9:12])
        circ += self._one_qubit_unitary(q2, symbols[12:])

        return circ

    def _two_qubit_pool(self, source, sink, symbols):
        # Do a pooling operation, reducing the entanglement from 2 qubits to 1
        circ = cirq.Circuit()

        sink_circ = self._one_qubit_unitary(sink, symbols[:3])
        source_circ = self._one_qubit_unitary(source, symbols[3:6])

        circ += sink_circ
        circ += source_circ
        circ += cirq.CX(source, sink)
        circ += sink_circ**-1

        return circ

    def _convolution_circuit(self, qubits, symbols):
        # Cascade of 2 qubit unitaries on all pairs of qubits
        circ = cirq.Circuit()
        for i in range(0, len(qubits) - 1, 2):
            circ += self._two_qubit_unitary(qubits[i], qubits[i + 1], symbols)

        for i in range(1, len(qubits), 2):
            j = (0 if i == len(qubits) - 1 else i + 1)
            circ += self._two_qubit_unitary(qubits[i], qubits[j], symbols)

        return circ

    def _pool_circuit(self, source_qubits, sink_qubits, symbols):
        # Pool information from two qubits into one
        circ = cirq.Circuit()
        for source, sink in zip(source_qubits, sink_qubits):
            circ += self._two_qubit_pool(source, sink, symbols)
        return circ

    def _make_model_circ(self, qubits):
        # Alternating convolution and pooling layers
        circ = cirq.Circuit()

        symbols = sympy.symbols("qconv0:%d" % QuantumCnn.NUM_PARAMS)

        # Offset into how many qubits have been destroyed by pooling
        qubit_offset = 0
        symbol_index = 0

        for i in range(QuantumCnn.NUM_LAYERS):
            # Apply convolution to qubits
            circ += self._convolution_circuit(
                qubits[qubit_offset:], symbols[symbol_index:symbol_index +
                                               QuantumCnn.CONVOLUTION_PARAMS])
            symbol_index += QuantumCnn.CONVOLUTION_PARAMS

            # Numer of qubits left
            num_qubits = encoding.QuantumImageEncoder.NUM_QUBITS - qubit_offset
            # Split between pooled qubits
            split_index = qubit_offset + num_qubits // 2
            # Pool first and second half of qubits
            circ += self._pool_circuit(
                qubits[qubit_offset:split_index], qubits[split_index:],
                symbols[symbol_index:symbol_index + QuantumCnn.POOL_PARAMS])
            symbol_index += QuantumCnn.POOL_PARAMS

            qubit_offset += num_qubits // 2

        assert qubit_offset == encoding.QuantumImageEncoder.NUM_QUBITS - 1

        return circ

    def _cluster_circuit(self, qubits):
        # Entangle all the qubits together
        circ = cirq.Circuit(cirq.H.on_each(qubits))
        for i in range(len(qubits)):
            j = (0 if i == len(qubits) - 1 else i + 1)
            circ += (cirq.CZ(qubits[i], qubits[j]))
        return circ

    def _make_model(self) -> tf.keras.Model:
        cluster = cirq.GridQubit.rect(1,
                                      encoding.QuantumImageEncoder.NUM_QUBITS)
        # Use the Z gate for readout
        operator = cirq.Z(cluster[-1])

        model_input = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        cluster_circ = tfq.layers.AddCircuit()(
            model_input, prepend=self._cluster_circuit(cluster))

        pqc = tfq.layers.PQC(self._make_model_circ(cluster),
                             operator)(cluster_circ)
        return tf.keras.Model(inputs=[model_input], outputs=[pqc])

    @tf.function
    def accuracy(self, y_actual, y_pred):
        # Custom accuracy mapping outputs to a boolean
        y_actual = tf.squeeze(y_actual)
        y_pred = tf.map_fn(lambda y: 1.0 if y >= 0.5 else 0.0, y_pred)
        return tf.keras.backend.mean(tf.keras.backend.equal(y_actual, y_pred))

    def _get_accuracy(self):
        return self.accuracy


class ClassicalNn(Model):
    """ A "fair" classical NN with around the same number of parameters as the quantum one,
        used to evaluate the QCNN.
        This has 91 parameters while the QCNN 84. """
    def _make_encoder(self) -> encoding.ImageEncoder:
        return encoding.ImageEncoder()

    def _make_model(self) -> tf.keras.Model:
        return tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(4, 4, 1)),
            tf.keras.layers.Dense(5, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

    def _get_accuracy(self):
        return "accuracy"
