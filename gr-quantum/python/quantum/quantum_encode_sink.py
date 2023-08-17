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


class quantum_encode_sink(gr.basic_block):
    """
    Encodes an input buffer into quantum form
    """
    def __init__(self, method='amplitude', num_qubits=4, shots=100):
        logging.basicConfig(level=logging.CRITICAL)
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.setLevel(logging.INFO)

        self.logger.info(f"Encoding using {method} method.")
        self.method = method
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
                approximation_degree=0, inverse=True))
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
        self.circuit.decompose(reps=8).draw(output='mpl')
        self.logger.info(f"Circuit depth = {self.circuit.depth()}")
        # plt.show(block=True)
        plt.savefig(
            f"/mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/circuit - {self.method}"
        )
        raise InterruptedError()

    def amplitude_post_process(self, raw_measurements):
        num_bits = len(raw_measurements[0])

        num_samples = len(raw_measurements)
        bit_averages = []

        for i in range(num_bits):
            bit_sum = 0
            for j in range(num_samples):
                bit_sum += int(raw_measurements[j][i])

            bit_average = bit_sum / num_samples
            bit_averages.append(bit_average)

        # Convert amplitudes to a numpy array of dtype float32
        # return np.array(bit_averages, dtype=np.float32)
        return np.array([bit_averages[0]])

    def mean_post_process(self, raw_measurements):
        output = [int(binary, 2) for binary in raw_measurements]
        self.logger.debug(output)
        return np.array([np.float32(np.mean(output))])

    def encode_basis(self, arr):
        self.circuit = QuantumCircuit(self.quantum,
                                      self.classical,
                                      name=f"Basis Encoding")
        # Perform delta encoding
        binary_encoded = np.zeros_like(arr, dtype=np.int32)

        # First sample starts at 0
        binary_encoded[0] = 1
        for i in range(1, len(arr)):
            # Compute up/down delta between consecutive samples
            binary_encoded[i] = int(np.real(arr[i] - arr[i - 1]) <= 0)

        self.logger.debug("binary_encoded state")
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
        cA, cD = pywt.dwt([np.abs(a) for a in arr], 'haar')
        cA = cA / np.linalg.norm(cA)
        cD = cD / np.linalg.norm(cD)

        # log the wavelet
        self.logger.debug("Wavelets:")
        self.logger.debug(cA)
        self.logger.debug(cD)

        # encode the angle
        for i in range(self.num_qubits):
            self.circuit.rx(cA[i]*math.pi, i)
            self.circuit.ry(cD[i]*math.pi, i)

    def qft_rotations(self, n):
        if n == 0: return self.circuit
        n -= 1
        self.circuit.h(n)
        for qubit in range(n):
            self.circuit.cp(math.pi/2**(n-qubit), qubit, n)
