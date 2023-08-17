import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from qiskit import Aer, QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT
from qiskit.visualization import plot_histogram
import math
import sys
from collections import Counter
from qiskit.extensions import UnitaryGate

class DefaultSignalGenerator:
    def __init__(self, sampling_rate, duration, frequency):
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.frequency = frequency
        
    def generate(self):
        t = np.linspace(0.0, self.duration, int(self.sampling_rate * self.duration), endpoint=False)
        arr = np.sin(2 * np.pi * self.frequency * t)
        return arr
class QuantumSineWaveEncoder:
    def __init__(self, signal_generator, num_qubits, shots, sampling_rate, duration, frequency):
        self.num_qubits = num_qubits
        self.shots = shots
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.frequency = frequency
        
        self.logger = logging.getLogger('main')
        self.logger.setLevel(logging.INFO)
        
        self.signal_generator = signal_generator
        
        self.quantum = QuantumRegister(self.num_qubits, "quantum")
        self.classical = ClassicalRegister(self.num_qubits, "classical")
        self.simulator = Aer.get_backend('aer_simulator')
        self.simulator.set_options(precision='single')
        
        self.circuit = QuantumCircuit(self.quantum, self.classical, name="Amplitude Encoding")
        
    def get_signal(self):
        signal = self.signal_generator.generate()[:2 ** self.num_qubits]
        return signal / np.linalg.norm(signal) # normalise signal
        
    def run_circuit(self):
        state = self.get_signal()
        self.circuit.initialize(state, self.circuit.qubits)
        
        dim = 2 * self.num_qubits
        
        transition_matrix = np.zeros((dim, dim), dtype=int)
        transition_matrix[2 * self.num_qubits - 1, 0] = 1
        transition_matrix[0, 1] = 1
        transition_matrix[1:2 * self.num_qubits - 1, 2:2 * self.num_qubits - 0] = np.identity(2 * self.num_qubits - 2)
        print(transition_matrix)
        # Create a QuantumCircuit with a custom UnitaryGate
        custom_gate = UnitaryGate(transition_matrix)
        self.circuit.append(custom_gate, range(self.num_qubits))
        
        self.circuit.measure(self.quantum, self.classical)
        self.circuit = transpile(self.circuit, self.simulator)
        
        result = self.simulator.run(self.circuit, shots=self.shots, memory=True).result()
        raw_measurements = result.get_memory(self.circuit)
        for measurement in raw_measurements:
            self.logger.info(measurement)
        
        return raw_measurements

    def plot_histogram(self, raw_measurements):
        unique, counts = np.unique(raw_measurements, return_counts=True)
        plt.bar(unique[len(unique)//2:], counts[len(counts)//2:]+counts[::-1][len(counts)//2:])
        plt.xlabel('Measurement Outcomes')
        plt.ylabel('Counts')
        plt.title('Measurement Outcomes Histogram')
        plt.xticks(unique)
        plt.show()
        
def run_and_plot_single_run_histogram(SignalGenerator, num_qubits, shots, sampling_rate, duration, frequencies):
    results = []

    for frequency in frequencies:
        signal_generator = SignalGenerator(sampling_rate, duration, frequency)
        qse = QuantumSineWaveEncoder(signal_generator, num_qubits, shots, sampling_rate, duration, frequency)
        raw_measurements = qse.run_circuit()
        qse.plot_histogram(raw_measurements)
        
def experiment_1():
  """
  Generates a heatmap from a pulse signal, using lots of samples
  """
  num_qubits = 8
  shots = 20000
  sampling_rate = 43000
  duration = 1.0
  starting_freq = 440.0  # Starting frequency in Hz
  interval = 1000.0  # Frequency interval in Hz
  num_frequencies = 1

  frequencies = np.arange(starting_freq, starting_freq + interval * num_frequencies, interval)

  signal_generator = DefaultSignalGenerator
  
  run_and_plot_single_run_histogram(signal_generator, num_qubits, shots, sampling_rate, duration, frequencies)


if __name__ == "__main__":
  experiment_1()
  
# import numpy as np
# from qiskit import QuantumCircuit, transpile, Aer, assemble
# from qiskit.extensions import UnitaryGate



# # Define the quantum circuit for edge detection
# def edge_detection_circuit(time_series, n_qubits):
#     dim = 2 * n_qubits
#     time_series = time_series[:dim]
#     qc = QuantumCircuit(n_qubits, n_qubits)

#     qc.initialize(time_series / np.linalg.norm(time_series), n_qubits)

#     transition_matrix = np.zeros((dim, dim), dtype=int)
#     transition_matrix[2 * n_qubits - 1, 0] = 1
#     transition_matrix[0, 1] = 1
#     transition_matrix[1:2 * n_qubits - 1, 2:2 * n_qubits - 0] = np.identity(2 * n_qubits - 2)
#     print(transition_matrix)
#     # Create a QuantumCircuit with a custom UnitaryGate
#     custom_gate = UnitaryGate(transition_matrix)
#     qc.append(custom_gate, range(n_qubtits))

#     return qc

# # Simulating the circuit
# n_qubits = 3  # Number of qubits
# time_series = [0.2, 0.5, 0.8, 0.5, 0.2, 0.5, 0.8, 0.5]  # Example amplitude-encoded time series
# circuit = edge_detection_circuit(time_series, n_qubits)

# simulator = Aer.get_backend('statevector_simulator')
# job = assemble(transpile(circuit, simulator), shots=1)
# result = simulator.run(job).result()
# statevector = result.get_statevector()

# # Edge detection results
# # edges = [np.abs(statevector[i]) ** 2 for i in range(2 ** n_qubits)]
# print("Edge detection results:", edges)
