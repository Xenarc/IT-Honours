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

import matplotlib.pyplot as plt
import numpy as np
from gnuradio import gr
from qiskit import (Aer, ClassicalRegister, QuantumCircuit, QuantumRegister,
                    transpile)
from qiskit.circuit.library import QFT

# def norm(arr):
#     amplitudes = [np.abs(a) * np.abs(a) for a in arr]
#     return arr / np.sqrt(sum(amplitudes))
# return arr / np.linalg.norm(arr, ord=1)
# return arr / np.linalg.norm(arr)



class quantum_encode_sink(gr.basic_block):
    """
    Encodes an input buffer into quantum form
    """
    def __init__(self, method='amplitude', num_qubits=4, shots=100):
        logging.basicConfig(level=logging.CRITICAL)
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.setLevel(logging.INFO)

        if(method == 'amplitude'): self.encode = self.encode_amplitude
        elif(method == 'angle'): self.encode = self.encode_angle
        else:
            self.logger.exception(f"Cannot encode using method {method}")
            raise Exception(f"Cannot encode using method {method}")

        self.logger.info(f"Encoding using {method} method.")

        self.num_qubits = num_qubits
        self.logger.info(f"Number of qubits = {self.num_qubits}")

        self.buff_size = 2**self.num_qubits
        self.logger.info(f"Buffer size = {self.buff_size}")

        self.shots = shots
        self.logger.info(f"Shots = {self.shots}")

        self.simulator = Aer.get_backend('aer_simulator')
        self.simulator.set_options(precision='single')


        # self.decimation = int(self.buff_size)

        gr.basic_block.__init__(
            self,
            name="quantum_encode_sink",
            in_sig=[np.complex64],
            out_sig=[np.float32],
            # decim=self.decimation
        )

    def general_work(self, input_items, output_items):
        if(len(input_items[0]) <= self.buff_size): return 0
        self.consume(0, self.buff_size)
        self.produce(0, 1)

        in0 = np.array(input_items[0][:self.buff_size], dtype=np.complex128)

        self.encode(in0)
        self.circuit += QFT(num_qubits=self.num_qubits, approximation_degree=2)
        self.circuit += QFT(num_qubits=self.num_qubits, approximation_degree=2, inverse=true)
        # self.qft_rotations(self.num_qubits)

        self.circuit.measure(self.quantum, self.classical)
        self.circuit = transpile(self.circuit, self.simulator)

        # Run and get counts
        result = self.simulator.run(self.circuit, shots=self.shots).result()
        counts = result.get_counts(self.circuit)

        most_probable_value = 0 # highest count
        most_probable_key = next(iter(counts)) # First / any key
        for key, value in counts.items():
            if value > most_probable_value:
                most_probable_key = key
                most_probable_value = value

        output_items[0][:] = float(most_probable_value / self.shots)

        # output_items[0][:] = int(most_probable_value, 2)
        # self.draw()

        return 1

    def draw(self):
        self.circuit.draw(output='mpl')
        plt.show(block=True)
        raise InterruptedError()

    def encode_amplitude(self, arr):
        self.classical = ClassicalRegister(self.num_qubits, "classical")
        self.quantum = QuantumRegister(self.num_qubits, "quantum")

        self.circuit = QuantumCircuit(self.quantum, self.classical, name=f"Amplitude Encoding")
        state = arr / np.sqrt(np.sum(arr * np.conj(arr)))
        self.circuit.initialize(state, self.circuit.qubits)

    def encode_angle(self, arr):
        self.circuit = QuantumCircuit(self.num_qubits, name=f"Angle Encoding")
        state = norm(arr)
        self.circuit.initialize(state, self.circuit.qubits)

    def qft_rotations(self, n):
        if n == 0: # Exit function if circuit is empty
            return self.circuit
        n -= 1 # Indexes start from 0
        self.circuit.h(n) # Apply the H-gate to the most significant qubit
        for qubit in range(n):
            # For each less significant qubit, we need to do a
            # smaller-angled controlled rotation:
            self.circuit.cp(math.pi/2**(n-qubit), qubit, n)
