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

class DefaultSignalGenerator:
    def __init__(self, sampling_rate, duration, frequency):
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.frequency = frequency
        
    def generate(self):
        t = np.linspace(0.0, self.duration, int(self.sampling_rate * self.duration), endpoint=False)
        arr = np.sin(2 * np.pi * self.frequency * t)
        return arr

class MultiToneSignalGenerator:
    def __init__(self, sampling_rate, duration, frequency):
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.frequency = frequency
        
    def generate(self):
        t = np.linspace(0.0, self.duration, int(self.sampling_rate * self.duration), endpoint=False)
        freq1 = np.sin(2 * np.pi * self.frequency * t)
        freq2 = np.sin(2 * np.pi * self.frequency*2 * t)
        arr = freq1 + freq2
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
        
        self.circuit = self.circuit.compose(
            QFT(num_qubits=self.num_qubits, approximation_degree=0, inverse=False))
        self.circuit.measure(self.quantum, self.classical)
        self.circuit = transpile(self.circuit, self.simulator)
        
        result = self.simulator.run(self.circuit, shots=self.shots, memory=True).result()
        raw_measurements = result.get_memory(self.circuit)
        for measurement in raw_measurements:
            self.logger.info(measurement)
        
        return raw_measurements

    def plot_histogram(self, raw_measurements):
        unique, counts = np.unique(raw_measurements, return_counts=True)
        plt.bar(unique, counts)
        plt.xlabel('Measurement Outcomes')
        plt.ylabel('Counts')
        plt.title('Measurement Outcomes Histogram')
        plt.xticks(unique)
        plt.show()

def run_and_plot_heatmap(SignalGenerator, num_qubits, shots, sampling_rate, duration, frequencies):
    results = []

    for frequency in frequencies:
        signal_generator = SignalGenerator(sampling_rate, duration, frequency)
        qse = QuantumSineWaveEncoder(signal_generator, num_qubits, shots, sampling_rate, duration, frequency)
        raw_measurements = qse.run_circuit()
        results.append((frequency, raw_measurements))

    measurement_counts = {}
    for frequency, raw_measurements in results:
        for measurement in raw_measurements:
            decimal_measurement = int(measurement, 2)
            if frequency not in measurement_counts:
                measurement_counts[frequency] = Counter()
            measurement_counts[frequency][decimal_measurement] += 1

    frequency_values = [freq for freq, _ in results]
    measurement_values = set()
    for counts in measurement_counts.values():
        measurement_values.update(counts.keys())

    heatmap_data = np.zeros((len(frequency_values), len(measurement_values)))
    for i, freq in enumerate(frequency_values):
        for j, meas in enumerate(measurement_values):
            heatmap_data[i, j] = measurement_counts[freq][meas]

    plt.imshow(heatmap_data, interpolation='nearest', cmap='viridis', norm=LogNorm(vmin=1, vmax=heatmap_data.max()))
    plt.colorbar(label='Counts')
    plt.xticks(np.arange(len(measurement_values)), list(measurement_values))
    plt.yticks(np.arange(len(frequency_values)), frequency_values)
    plt.xlabel('Measurement Outcomes (Decimal)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Measurement Outcomes Heatmap')
    plt.show()
    
def run_and_plot_multiple_runs_scatter(SignalGenerator, num_qubits, shots, sampling_rate, duration, frequencies):
    results = []

    for frequency in frequencies:
        signal_generator = SignalGenerator(sampling_rate, duration, frequency)
        qse = QuantumSineWaveEncoder(signal_generator, num_qubits, shots, sampling_rate, duration, frequency)
        raw_measurements = qse.run_circuit()
        results.append((frequency, raw_measurements))

    for frequency, raw_measurements in results:
        unique, counts = np.unique(raw_measurements, return_counts=True)
        plt.scatter([frequency] * len(unique), counts, label=f"Frequency: {frequency} Hz")

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Counts')
    plt.title('Measurement Outcomes Scatter')
    plt.legend()
    plt.show()

def experiment_1():
  """
  Generates a heatmap
  """
  num_qubits = 8
  shots = 2000
  sampling_rate = 44000
  duration = 1.0
  starting_freq = 440.0  # Starting frequency in Hz
  interval = 1000.0  # Frequency interval in Hz
  num_frequencies = 50

  frequencies = np.arange(starting_freq, starting_freq + interval * num_frequencies, interval)

  signal_generator = DefaultSignalGenerator
  
  run_and_plot_heatmap(signal_generator, num_qubits, shots, sampling_rate, duration, frequencies)

def experiment_2():
  """
  Generates sequential histograms
  """
  num_qubits = 5
  shots = 1000
  sampling_rate = 10000
  duration = 1.0
  starting_freq = 440.0  # Starting frequency in Hz
  interval = 220.0  # Frequency interval in Hz
  num_frequencies = 20

  frequencies = np.arange(starting_freq, starting_freq + interval * num_frequencies, interval)

  signal_generator = DefaultSignalGenerator

  run_and_plot_multiple_runs_scatter(signal_generator, num_qubits, shots, sampling_rate, duration, frequencies)

def experiment_3():
  """
  Generates a heatmap from a multitone signal
  """
  num_qubits = 8
  shots = 2000
  sampling_rate = 44000
  duration = 1.0
  starting_freq = 440.0  # Starting frequency in Hz
  interval = 1000.0  # Frequency interval in Hz
  num_frequencies = 50

  frequencies = np.arange(starting_freq, starting_freq + interval * num_frequencies, interval)

  signal_generator = MultiToneSignalGenerator
  
  run_and_plot_heatmap(signal_generator, num_qubits, shots, sampling_rate, duration, frequencies)

def experiment_4():
  """
  Generates a heatmap from a multitone signal, using lots of samples
  """
  num_qubits = 8
  shots = 20000
  sampling_rate = 44000
  duration = 1.0
  starting_freq = 440.0  # Starting frequency in Hz
  interval = 1000.0  # Frequency interval in Hz
  num_frequencies = 50

  frequencies = np.arange(starting_freq, starting_freq + interval * num_frequencies, interval)

  signal_generator = MultiToneSignalGenerator
  
  run_and_plot_heatmap(signal_generator, num_qubits, shots, sampling_rate, duration, frequencies)

if __name__ == "__main__":
  experiment_3()
  