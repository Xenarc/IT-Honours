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
from scipy.signal import find_peaks, savgol_filter

class DefaultSignalGenerator:
    def __init__(self, sampling_rate, duration, frequency):
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.frequency = frequency
        
    def generate(self):
        t = np.linspace(0.0, self.duration, int(self.sampling_rate * self.duration), endpoint=False)
        arr = np.sin(2 * np.pi * self.frequency * t)
        return arr

class SteppedSignalGenerator:
    def __init__(self, sampling_rate, duration, frequency):
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.frequency = frequency
        
    def generate(self):
        t = np.linspace(0.0, self.duration, int(self.sampling_rate * self.duration), endpoint=False)
        arr = np.zeros(len(t))
        pulse_start = int(len(t) * 0.25)
        pulse_end = int(len(t) * 0.55)
        arr[pulse_start:pulse_end] = np.sin(2 * np.pi * self.frequency * t[pulse_start:pulse_end])
        plt.plot(t, arr, label='Array Values')
        plt.show()
        return arr

class SteppedAmplitudeSignalGenerator:
    def __init__(self, sampling_rate, duration, frequency):
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.frequency = frequency
        
    def generate(self):
        t = np.linspace(0.0, self.duration, int(self.sampling_rate * self.duration), endpoint=False)
        arr = np.zeros(len(t))
        pulse_start = int(len(t) * 0.25)
        pulse_end = int(len(t) * 0.55)
        arr[pulse_start:pulse_end] = 1
        # plt.plot(t, arr, label='Array Values')
        # plt.show()
        return arr

class SteppedAmplitudeSignalGeneratorWithNoise:
    def __init__(self, sampling_rate, duration, frequency):
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.frequency = frequency
        
    def generate(self, noise_level=1):
        t = np.linspace(0.0, self.duration, int(self.sampling_rate * self.duration), endpoint=False)
        arr = np.zeros(len(t))
        pulse_start = int(len(t) * 0.25)
        pulse_end = int(len(t) * 0.55)
        arr[pulse_start:pulse_end] = 1
        
        # Generate random noise
        noise = noise_level * np.random.randn(len(t))
        
        # Add noise to the signal
        noisy_signal = arr + noise
        
        window_size = 3
        weights = np.arange(1, window_size + 1)
        noisy_signal = np.convolve(noisy_signal, weights / np.sum(weights), mode='same')
        
        # plt.plot(t, noisy_signal, label='Noisy Signal')
        # plt.show()
        
        return noisy_signal

class SteppedAmplitudeSignalGeneratorWithSmallNoise:
    def __init__(self, sampling_rate, duration, frequency):
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.frequency = frequency
        
    def generate(self, noise_level=0.4):
        t = np.linspace(0.0, self.duration, int(self.sampling_rate * self.duration), endpoint=False)
        arr = np.zeros(len(t))
        pulse_start = int(len(t) * 0.25)
        pulse_end = int(len(t) * 0.55)
        arr[pulse_start:pulse_end] = 1
        
        # Generate random noise
        noise = noise_level * np.random.randn(len(t))
        
        # Add noise to the signal
        noisy_signal = arr + noise
        
        window_size = 3
        weights = np.arange(1, window_size + 1)
        noisy_signal = np.convolve(noisy_signal, weights / np.sum(weights), mode='same')
        
        plt.plot(t, noisy_signal, label='Noisy Signal')
        plt.show()
        
        return noisy_signal

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
        
    def get_signal(self, num_qubits):
        signal = self.signal_generator.generate()[:2 ** num_qubits]
        return signal / np.linalg.norm(signal) # normalise signal
        
    def run_circuit(self):
        signal = self.get_signal(self.num_qubits-1)
        state = np.kron(np.array([1, 0]), signal)
        self.circuit.initialize(state, range(0, self.num_qubits))
        
        # print(transition_matrix)
        # Create a QuantumCircuit with a custom UnitaryGate
        custom_gate = UnitaryGate(self.qSobel(self.num_qubits))
        
        self.circuit.h(0)
        # self.circuit = self.circuit.compose(
        #     QFT(num_qubits=self.num_qubits, approximation_degree=0, inverse=False))
        # self.circuit.append(custom_gate, range(self.num_qubits))
        # self.circuit = self.circuit.compose(
        #     QFT(num_qubits=self.num_qubits, approximation_degree=0, inverse=True))
        self.circuit.unitary(self.qSobel(self.num_qubits), range(self.num_qubits))
        self.circuit.h(0)
        
        self.circuit.measure(self.quantum, self.classical)
        self.circuit = transpile(self.circuit, self.simulator)
        
        result = self.simulator.run(self.circuit, shots=self.shots, memory=True).result()
        raw_measurements = result.get_memory(self.circuit)
        for measurement in raw_measurements:
            self.logger.info(measurement)
        
        # print([measurement[1] for measurement in raw_measurements])
        # raw_measurements = [measurement for measurement in raw_measurements if measurement[0] == '1']
        
        return raw_measurements, signal
      
    def qSobel(self, n_qubits):
      transition_matrix = np.roll(np.identity(2**n_qubits), 1, axis=1)
      return transition_matrix
        
    def detect_pulses(self, time_series, threshold_percentage, min_pulse_width):
      max_value = max(time_series)
      threshold = threshold_percentage * max_value
      print("threshold: ", threshold)
      
      pulses = []
      pulse_start = None
      
      for i, value in enumerate(time_series):
        if value >= threshold:
          if pulse_start is None:
            pulse_start = i
        elif pulse_start is not None:
          pulse_end = i - 1
          pulse_width = pulse_end - pulse_start + 1
          if pulse_width >= min_pulse_width:
            pulses.append((pulse_start, pulse_end))
          pulse_start = None
    
      if pulse_start is not None:
        pulse_end = len(time_series) - 1
        pulse_width = pulse_end - pulse_start + 1
        if pulse_width >= min_pulse_width:
          pulses.append((pulse_start, pulse_end))
      
      return pulses

    
    def plot_histogram(self, raw_measurements, fold):
        unique, counts = np.unique(raw_measurements, return_counts=True)
        probabilities = counts / np.sum(counts)
        x_values = np.arange(len(probabilities))

        # Create a stem plot
        plt.stem(x_values, probabilities, basefmt=" ", use_line_collection=True)
        plt.xlabel('Measurement Outcome')
        plt.ylabel('Probability')
        plt.title('Stem Plot of Measurement Probabilities')
      
    def pulse_graph(self, raw_measurements, input_signal):
      unique, counts = np.unique(raw_measurements, return_counts=True)
      probabilities = counts / np.sum(counts)
      
      decimal_measurements = [int(measurement, 2) for measurement in raw_measurements]
      counts = Counter(decimal_measurements)
      total_shots = self.shots
      probabilities = {outcome: count / total_shots for outcome, count in counts.items()}
      probabilities = [probabilities.get(i, 0) ** 0.5 for i in range(2 ** self.num_qubits)]
    
      
      print(self.detect_pulses(probabilities, 0.1, 15))
      
      
      
      x_values = np.arange(len(probabilities))
      
      window_size = 3
      probabilities = np.convolve(probabilities, np.ones(window_size) / window_size, mode='same')
      
      constant = 2
      peaks, _ = find_peaks(probabilities, distance=constant)  # Adjust distance as needed

      # Create pulse widths
      pulse_widths = np.diff(peaks)  # Calculate the difference between peak indices

      # Create denoised signal using pulse widths
      denoised_signal = np.zeros_like(probabilities)
      for i, width in enumerate(pulse_widths):
          x_start = peaks[i]
          x_end = peaks[i] + width
          denoised_signal[x_start:x_end] = probabilities[x_start]

      # Plot the original probabilities
      plt.step(np.arange(len(probabilities)), probabilities, where='mid', label='Original Probabilities')

      # Overlay the pulse widths as vertical lines
      for i, width in enumerate(pulse_widths):
          x_position = peaks[i] + width // constant
          # plt.axvline(x=x_position, color='red', linestyle='--', alpha=0.5, label='Pulse Width')

      # Create the figure and the primary y-axis for probabilities and denoised steps
      fig, ax1 = plt.subplots()

      # Plot the original probabilities on the primary y-axis
      ax1.step(np.arange(len(probabilities)), probabilities, where='mid', label='Original Probabilities', color='tab:blue')
      ax1.step(np.arange(len(denoised_signal)), denoised_signal, where='mid', color='green', linestyle='--', label='Denoised Steps')
      ax1.set_xlabel('Time Period')
      ax1.set_ylabel('Probabilities / Denoised Steps', color='tab:blue')
      ax1.tick_params(axis='y', labelcolor='tab:blue')

      # Create the secondary y-axis for the input signal
      ax2 = ax1.twinx()
      ax2.plot(np.arange(len(input_signal)), input_signal, color='tab:orange', label='Input Signal', alpha=0.7)
      ax2.set_ylabel('Input Signal', color='tab:orange')
      ax2.tick_params(axis='y', labelcolor='tab:orange')

      # Combine the legends for both y-axes
      lines, labels = ax1.get_legend_handles_labels()
      lines2, labels2 = ax2.get_legend_handles_labels()
      ax2.legend(lines + lines2, labels + labels2, loc='upper left')

      plt.title('Quantized Pulses Overlay with Denoised Steps and Input Signal')
      plt.show()

def run_and_plot_single_run_histogram_nbr_samples(SignalGenerator, num_qubits, shots, nbrSamples, frequencies, fold=True):
  duration = 1
  sampling_rate = nbrSamples / duration
  run_and_plot_single_run_histogram(SignalGenerator, num_qubits, shots, sampling_rate, duration, frequencies, fold)
        
def run_and_plot_single_run_histogram(SignalGenerator, num_qubits, shots, sampling_rate, duration, frequencies, fold=True):
  results = []

  for frequency in frequencies:
    signal_generator = SignalGenerator(sampling_rate, duration, frequency)
    qse = QuantumSineWaveEncoder(signal_generator, num_qubits, shots, sampling_rate, duration, frequency)
    raw_measurements, input_signal = qse.run_circuit()
    qse.pulse_graph(raw_measurements, input_signal)
    # qse.plot_histogram(raw_measurements, fold)
        
def run_and_plot_single_run_heatmap_nbr_samples(SignalGenerator, num_qubits, shots, nbrSamples, frequencies, fold=True):
  results = []
  duration = 1
  sampling_rate = nbrSamples / duration

  for frequency in frequencies:
    signal_generator = SignalGenerator(sampling_rate, duration, frequency)
    qse = QuantumSineWaveEncoder(signal_generator, num_qubits, shots, sampling_rate, duration, frequency)
    raw_measurements, input_signal = qse.run_circuit()
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
        
def experiment_1():
  print("Experiment 1")
  """
  Generates a heatmap from a normal signal
  """
  num_qubits = 4
  shots = 20000
  sampling_rate = 43000
  duration = 1.0
  starting_freq = 440.0  # Starting frequency in Hz
  interval = 1000.0  # Frequency interval in Hz
  num_frequencies = 1

  frequencies = np.arange(starting_freq, starting_freq + interval * num_frequencies, interval)

  signal_generator = DefaultSignalGenerator
  
  run_and_plot_single_run_histogram(signal_generator, num_qubits, shots, sampling_rate, duration, frequencies)
        
def experiment_2():
  print("Experiment 2")
  """
  Generates a heatmap from a pulse signal
  """
  num_qubits = 7
  shots = 20000
  nbrSamples = 2 ** num_qubits
  starting_freq = 440.0  # Starting frequency in Hz
  interval = 1000.0  # Frequency interval in Hz
  num_frequencies = 1

  frequencies = np.arange(starting_freq, starting_freq + interval * num_frequencies, interval)

  signal_generator = SteppedSignalGenerator
  
  run_and_plot_single_run_histogram_nbr_samples(signal_generator, num_qubits, shots, nbrSamples, frequencies, False)

def experiment_3():
  print("Experiment 3")
  """
  Generates a heatmap from a pulse signal
  """
  num_qubits = 8
  shots = 20000
  nbrSamples = 2 ** num_qubits
  starting_freq = 4400.0  # Starting frequency in Hz
  interval = 1000.0  # Frequency interval in Hz
  num_frequencies = 1

  frequencies = np.arange(starting_freq, starting_freq + interval * num_frequencies, interval)

  signal_generator = SteppedAmplitudeSignalGenerator
  
  run_and_plot_single_run_histogram_nbr_samples(signal_generator, num_qubits, shots, nbrSamples, frequencies, False)

def experiment_4():
  print("Experiment 4")
  """
  Generates a heatmap from a pulse signal
  """
  num_qubits = 10
  shots = 20000
  nbrSamples = 2 ** num_qubits
  starting_freq = 4400.0  # Starting frequency in Hz
  interval = 1000.0  # Frequency interval in Hz
  num_frequencies = 1

  frequencies = np.arange(starting_freq, starting_freq + interval * num_frequencies, interval)

  signal_generator = SteppedAmplitudeSignalGeneratorWithNoise
  
  run_and_plot_single_run_histogram_nbr_samples(signal_generator, num_qubits, shots, nbrSamples, frequencies, False)

def experiment_5():
  print("Experiment 5")
  """
  Generates a heatmap from a pulse signal
  """
  num_qubits = 11
  shots = 20000
  nbrSamples = 2 ** num_qubits
  starting_freq = 4400.0  # Starting frequency in Hz
  interval = 1000.0  # Frequency interval in Hz
  num_frequencies = 1

  frequencies = np.arange(starting_freq, starting_freq + interval * num_frequencies, interval)

  signal_generator = SteppedAmplitudeSignalGeneratorWithSmallNoise
  
  run_and_plot_single_run_histogram_nbr_samples(signal_generator, num_qubits, shots, nbrSamples, frequencies, False)

def experiment_6():
  print("Experiment 6")
  """
  Generates a heatmap from a pulse signal
  """
  num_qubits = 10
  shots = 20000
  nbrSamples = 2 ** num_qubits
  starting_freq = 4400.0  # Starting frequency in Hz
  interval = 1000.0  # Frequency interval in Hz
  num_frequencies = 1

  frequencies = np.arange(starting_freq, starting_freq + interval * num_frequencies, interval)

  signal_generator = SteppedAmplitudeSignalGeneratorWithNoise
  
  run_and_plot_single_run_heatmap_nbr_samples(signal_generator, num_qubits, shots, nbrSamples, frequencies)

if __name__ == "__main__":
  # experiment_1()
  # experiment_2()
  # experiment_3()
  # experiment_4()
  experiment_5()
