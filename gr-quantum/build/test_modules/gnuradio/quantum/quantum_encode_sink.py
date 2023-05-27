#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2023 Mark Blashki.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import logging
import math
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pywt
from gnuradio import gr
from qiskit import (Aer, ClassicalRegister, QuantumCircuit, QuantumRegister,
                    transpile)
from qiskit.circuit.library import QFT


def norm(arr):
    amplitudes = [np.abs(a) * np.abs(a) for a in arr]
    return arr / np.sqrt(sum(amplitudes))

class quantum_encode_sink(gr.basic_block):
    """
    Encodes an input buffer into quantum form
    """
    def __init__(self, method='amplitude', num_qubits=4, shots=100):
        logging.basicConfig(level=logging.CRITICAL)
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.setLevel(logging.INFO)

        self.logger.info(f"Encoding using {method} method.")
        if(method == 'amplitude'):
            self.encode = self.encode_amplitude
            self.post_process = self.amplitude_post_process
        elif(method == 'angle'):
            self.encode = self.encode_angle
            self.post_process = self.mean_post_process
        elif(method == 'basis'):
            self.encode = self.encode_basis
            self.post_process = self.mean_post_process
        else:
            self.logger.exception(f"Cannot encode using method {method}")
            raise Exception(f"Cannot encode using method {method}")


        self.do_draw = False
        self.logger.info(f"Draw circuit (and quit)? {self.do_draw}")

        self.logger.info(f"Number of qubits = {num_qubits}")
        self.num_qubits = num_qubits

        self.buff_size = 2**self.num_qubits
        self.logger.info(f"Buffer size = {self.buff_size}")

        self.logger.info(f"Shots = {shots}")
        self.shots = shots

        self.simulator = Aer.get_backend('aer_simulator')
        self.simulator.set_options(precision='single')

        gr.basic_block.__init__(
            self,
            name="quantum_encode_sink",
            in_sig=[np.complex64],
            out_sig=[np.float32],
        )

    def general_work(self, input_items, output_items):
        self.consume(0, 1)

        # Get the buffer. input_items[0] is the newest
        in0 = np.array(input_items[0][:self.buff_size], dtype=np.complex128)

        self.classical = ClassicalRegister(self.num_qubits, "classical")
        self.quantum = QuantumRegister(self.num_qubits, "quantum")

        self.encode(in0)
        self.circuit = self.circuit.compose(
            QFT(num_qubits=self.num_qubits,
                approximation_degree=2))
        # self.circuit = self.circuit.compose(
        #     QFT(num_qubits=self.num_qubits,
        #         approximation_degree=2,
        #         inverse=True))
        # self.qft_rotations(self.num_qubits)

        self.circuit.measure(self.quantum, self.classical)
        self.circuit = transpile(self.circuit, self.simulator)

        # Run and get counts
        result = self.simulator.run(self.circuit, shots=self.shots, memory=True).result()
        counts = result.get_counts(self.circuit)
        raw_measurements = result.get_memory(self.circuit)

        self.logger.debug(raw_measurements)

        output_buffer = self.post_process(raw_measurements)

        num_output_items = len(output_buffer)

        output_items[0][:num_output_items] = output_buffer

        self.logger.debug(num_output_items)

        self.set_output_multiple(num_output_items)

        if (self.do_draw):
            self.draw()

        return num_output_items

    def draw(self):
        self.circuit.draw(output='mpl')
        plt.show(block=True)
        raise InterruptedError()

    def amplitude_post_process(self, raw_measurements):
        # Convert binary measurements to decimal values
        decimal_values = [int(binary, 2) for binary in raw_measurements]

        # Count the occurrences of each decimal value
        state_counts = Counter(decimal_values)

        # Calculate the total number of measurements
        total_measurements = len(decimal_values)

        # Normalize the frequencies to obtain probabilities
        probabilities = {state: count / total_measurements for state, count in state_counts.items()}

        # Calculate the amplitudes by taking the square root of the probabilities
        amplitudes = {state: np.sqrt(prob) for state, prob in probabilities.items()}

        # Convert amplitudes to a numpy array of dtype float32
        return np.array(list(amplitudes.values()), dtype=np.float32)

    def mean_post_process(self, raw_measurements):
        output = [int(binary, 2) for binary in raw_measurements]
        self.logger.debug(output)
        return np.array([np.float32(np.mean(output))])

    def encode_basis(self, arr):
        self.circuit = QuantumCircuit(self.quantum,
                                      self.classical,
                                      name=f"Basis Encoding")
        # Perform delta encoding
        delta_encoded = np.zeros_like(arr, dtype=np.int32)

        # First sample remains unchanged
        delta_encoded[0] = np.linalg.norm(arr[0]) + 1
        for i in range(1, len(arr)):
            # Compute delta between consecutive samples
            delta_encoded[i] = np.linalg.norm(arr[i] - arr[i - 1])

        self.logger.debug(delta_encoded)

        # Convert delta-encoded samples to binary representation
        binary_encoded = np.zeros_like(delta_encoded, dtype=np.uint8)
        for i, sample in enumerate(delta_encoded):
            # Binary encoding
            binary_encoded[i] = int(np.real(sample) > 0) * 2 + int(np.imag(sample) > 0)

        self.logger.debug(binary_encoded)

        self.logger.debug("delta_encoded state")
        self.logger.debug(binary_encoded)

        state = np.array(binary_encoded, dtype=np.longdouble) / np.linalg.norm(binary_encoded)
        self.circuit.initialize(state, self.circuit.qubits)

    def encode_amplitude(self, arr):
        self.circuit = QuantumCircuit(self.quantum,
                                      self.classical,
                                      name=f"Amplitude Encoding")
        state = arr / np.linalg.norm(arr)
        self.circuit.initialize(state, self.circuit.qubits)

    def encode_angle(self, arr):
        self.circuit = QuantumCircuit(self.quantum,
                                      self.classical,
                                      name=f"Angle Encoding")

        # Convert array to double, because wavelet transform can't do complexes
        cA, cD = norm(pywt.dwt([np.abs(a) for a in arr], 'haar'))

        # log the wavelet
        self.logger.debug("Wavelets:")
        self.logger.debug(cA)
        self.logger.debug(cD)

        # encode the angle
        for i in range(self.num_qubits):
            self.circuit.ry(cA[i]*2*math.pi, i)
            self.circuit.rz(cD[i]*2*math.pi, i)


    def qft_rotations(self, n):
        if n == 0: # Exit function if circuit is empty
            return self.circuit
        n -= 1 # Indexes start from 0
        self.circuit.h(n) # Apply the H-gate to the most significant qubit
        for qubit in range(n):
            # For each less significant qubit, we need to do a
            # smaller-angled controlled rotation:
            self.circuit.cp(math.pi/2**(n-qubit), qubit, n)
