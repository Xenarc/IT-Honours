import logging

import numpy as np
from qiskit import QuantumCircuit


class amplitude_encoder:
  def __init__(self, num_qubits):
    self.circ = QuantumCircuit(num_qubits)

  def encode_array(self, arr):
    magnitude = np.sqrt(sum([x*x for x in arr]))
    state = (1 / magnitude) * np.array(arr)

    self.circ.initialize(state, [0, 1])
    self.circ.draw(output='mpl')

        # # create figure (will only create new window if needed)
        # plt.figure()
        # # Generate plot1
        # plt.plot(range(10, 20))
        # # Show the plot in non-blocking mode
        # plt.show(block=False)

        # # Finally block main thread until all plots are closed
        # plt.show()
        

