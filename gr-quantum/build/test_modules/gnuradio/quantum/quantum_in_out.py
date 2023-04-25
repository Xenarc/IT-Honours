import logging

import numpy as np
from gnuradio import gr
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


class quantum_in_out(gr.sync_block):
    """
    docstring for block quantum_in_out
    """
    def __init__(self):        
        gr.sync_block.__init__(self,
            name="quantum_in_out",
            in_sig=[np.complex64],
            out_sig=[np.complex64])
        logging.info("Initialising quantum block")
        
        # Use Aer's AerSimulator
        self.simulator = AerSimulator(precision)
        # Create a Quantum Circuit acting on the q register
        self.circuit = QuantumCircuit(2, 2)
        # Add a H gate on qubit 0
        self.circuit.h(0)
        # Add a CX (CNOT) gate on control qubit 0 and target qubit 1
        self.circuit.cx(0, 1)
        # Map the quantum measurement to the classical bits
        self.circuit.measure([0, 1], [0, 1])


    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]
        encode(in0)
        out[:] = in0
        return len(output_items[0])
