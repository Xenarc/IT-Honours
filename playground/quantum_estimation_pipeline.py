import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from qiskit import Aer, QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT
from qiskit.visualization import plot_histogram
import math
import sys
import random
from collections import Counter
import uuid

class DefaultSignalGenerator:
    def __init__(self, num_samples, frequency, pulse_duration, toa, do_plot, noise_level=0.0):
        self.num_samples = num_samples
        self.frequency = frequency
        self.pulse_duration = pulse_duration
        self.toa = toa
        self.do_plot = do_plot
        self.noise_level = noise_level

    def generate(self, amplitude=False):
        t = np.linspace(0, self.num_samples, self.num_samples)
        signal = np.zeros(len(t))
        pulse_start = self.toa
        pulse_end = self.toa + self.pulse_duration
        if amplitude:
            signal[pulse_start:pulse_end] = 1
        else:
            signal[pulse_start:pulse_end] = np.sin(2 * np.pi * self.frequency * t[pulse_start:pulse_end])
        signal = signal + np.random.normal(0, self.noise_level, size=len(signal))
        if self.do_plot:
            plt.plot(t, signal, label='Signal')
            plt.savefig(str(uuid.uuid4()) + ' signal.png')
            plt.show()
        return signal


class HistogramOutputProcessor:

    def __init__(self,
                 output_processor=lambda raw_measurements: raw_measurements):
        self.output_processor = output_processor

    def __call__(self, raw_measurements):
        unique, counts = np.unique(raw_measurements, return_counts=True)

        # Convert binary strings to decimal integers
        decimal_unique = [int(binary_str, 2) for binary_str in unique]

        plt.bar(decimal_unique, counts)
        plt.xlabel('Measurement Outcomes (Decimal)')
        plt.ylabel('Counts')
        plt.title('Measurement Outcomes Histogram')
        plt.xticks(decimal_unique)
        plt.savefig(str(uuid.uuid4()) + ' histogram.png')
        plt.show()

        return self.output_processor(raw_measurements)

class HeatMapOutputProcessor:

    def __init__(self,
                 output_processor=lambda raw_measurements: raw_measurements):
        self.output_processor = output_processor

    def __call__(self, raw_measurements):
        # Count measurement outcomes
        measurement_counts = Counter(raw_measurements)

        # Create the heatmap
        measurement_values = set(measurement_counts.keys())
        heatmap_data = np.zeros(len(measurement_values))
        for j, meas in enumerate(measurement_values):
            heatmap_data[j] = measurement_counts[meas]

        # Plot the heatmap
        plt.imshow(
            [heatmap_data],
            interpolation='nearest',
            cmap='jet',
            aspect='auto',
            #    norm=LogNorm(vmin=1, vmax=heatmap_data.max())
        )
        plt.colorbar(label='Counts')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('Measurement Outcomes (Decimal)')
        plt.title(f'Measurement Outcomes Heatmap')
        plt.savefig(str(uuid.uuid4()) + ' heatmap.png')
        plt.show()

        return self.output_processor(raw_measurements)

class PulseDetectorOutputProcessor:
    def __init__(self, shots, num_qubits, threshold_percentage, min_pulse_width):
        self.num_qubits = num_qubits
        self.shots = shots
        self.threshold_percentage = threshold_percentage
        self.min_pulse_width = min_pulse_width


    def get_probabilities(raw_measurements, shots, num_qubits):
        unique, counts = np.unique(raw_measurements, return_counts=True)
        probabilities = counts / np.sum(counts)

        decimal_measurements = [
            int(measurement, 2) for measurement in raw_measurements
        ]
        counts = Counter(decimal_measurements)
        total_shots = shots
        probabilities = {
            outcome: count / total_shots
            for outcome, count in counts.items()
        }
        probabilities = [
            probabilities.get(i, 0)**0.5 for i in range(2**num_qubits)
        ]
        return probabilities

    def detect_pulses(self, time_series, threshold_percentage, min_pulse_width):
        max_value = max(time_series)
        threshold = threshold_percentage * max_value
        # print("threshold: ", threshold)

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

    def __call__(self, raw_measurements):
        probabilities = PulseDetectorOutputProcessor.get_probabilities(raw_measurements, self.shots, self.num_qubits)
        pulses = self.detect_pulses(probabilities, self.threshold_percentage, self.min_pulse_width)
        return raw_measurements, pulses

class AggregatorOutputProcessor:
    def __init__(self, processor1, processor2):
        self.processor1 = processor1
        self.processor2 = processor2

    def __call__(self, raw_measurements_list):
        processed_measurements1 = self.processor1(raw_measurements_list[0])
        processed_measurements2 = self.processor2(raw_measurements_list[1])
        # Perform some aggregation logic here, like summing the measurements
        aggregated_measurements = processed_measurements1 + processed_measurements2
        return aggregated_measurements

class PassThroughOutputProcessor:
    def __init__(self):
        pass

    def __call__(self, raw_measurements):
        return raw_measurements

class QuantumSineWaveEncoder:
    def __init__(self, num_qubits, num_ancilliary, shots):
        self.num_qubits = num_qubits
        self.num_ancilliary = num_ancilliary
        self.total_qubits = num_ancilliary + num_qubits
        self.shots = shots

        self.quantum = QuantumRegister(self.total_qubits, "quantum")
        self.classical = ClassicalRegister(self.total_qubits, "classical")
        self.simulator = Aer.get_backend('aer_simulator')
        self.simulator.set_options(precision='single')

        self.circuit = QuantumCircuit(self.quantum, self.classical, name="Quantum Signal Processor")

    def run_circuit(self, circuit, signal, output_processor):
        state = signal / np.linalg.norm(signal) # intiialise a state from the signal

        for i in range(self.num_ancilliary):
            state = np.kron(np.array([1, 0]), state) # initialise the ancilliary qubits to |0>

        self.circuit.initialize(state, self.circuit.qubits)
        circuit()
        self.circuit.measure(self.quantum, self.classical)

        self.circuit = transpile(self.circuit, self.simulator)
        result = self.simulator.run(self.circuit, shots=self.shots, memory=True).result()

        raw_measurements = result.get_memory(self.circuit)
        return output_processor(raw_measurements)

    def frequencyEstimationCircuit(self):
        self.circuit = self.circuit.compose(
            QFT(num_qubits=self.total_qubits,
                approximation_degree=0,
                inverse=False))

    def pulseDetection(self):
        transition_matrix = np.roll(np.identity(2**self.total_qubits), 1, axis=1)
        self.circuit.h(0)
        self.circuit.unitary(transition_matrix, range(self.total_qubits))
        self.circuit.h(0)


    def expand_state_vector_from_list(self, input_list, desired_length):
        """
        Transform a flexible-length list to a fixed length by cutting or duplicating elements.

        This function takes a flexible-length list and transforms it to a fixed length
        as specified by the desired length. If the input list is longer than the desired
        length, it will be cut to match the desired length. If the input list is shorter,
        it will be duplicated and concatenated to reach the desired length.

        Parameters:
            input_list (list): The flexible-length list to be transformed.
            desired_length (int): The desired fixed length for the transformed list.

        Returns:
            list: The transformed list with the desired fixed length.
        """

        transformed_list = input_list[:] # copy input

        while desired_length > len(transformed_list):
            transformed_list = np.concatenate([transformed_list, transformed_list])

        return transformed_list[:desired_length]

def experiment_1():
    """Constant Frequency - no pulse, no noise. Plots Histogram"""
    num_qubits = 8
    num_ancilliary = 1
    shots = 20000
    num_samples = 2**(num_qubits)
    frequency = 1000  # Starting frequency in Hz
    pulse_duration = num_samples
    toa = 0
    do_plot = False

    signal_generator = DefaultSignalGenerator(num_samples, frequency,
                                              pulse_duration, toa, do_plot)
    signal = signal_generator.generate()
    output_processor = HistogramOutputProcessor()

    qse = QuantumSineWaveEncoder(output_processor, num_qubits, num_ancilliary,
                                 shots)
    raw_measurements = qse.run_circuit(qse.frequencyEstimationCircuit, signal)

def experiment_2():
    """Constant Frequency - no pulse, no noise. Plots Heatmap"""
    num_qubits = 9
    shots = 20000
    num_samples = 2**(num_qubits - 1)
    frequency = 10000  # Starting frequency in Hz
    pulse_duration = num_samples
    toa = 0
    do_plot = True

    signal_generator = DefaultSignalGenerator(num_samples, frequency,
                                              pulse_duration, toa, do_plot)
    signal = signal_generator.generate()
    output_processor = HeatMapOutputProcessor()

    qse = QuantumSineWaveEncoder(output_processor, num_qubits, shots)
    raw_measurements = qse.run_circuit(qse.frequencyEstimationCircuit, signal)

def experiment_3():
    """"""
    num_qubits = 8
    num_ancilliary = 1
    shots = 20000
    num_samples = 2**(num_qubits)
    frequency = 200  # Starting frequency in Hz
    pulse_duration = int(num_samples * 0.1)
    toa = int(0.1*num_samples)
    threshold_percentage = 0.3
    min_pulse_width = 15
    noise_level = 0.2
    do_plot = True

    total_qubits = num_ancilliary + num_qubits

    signal_generator = DefaultSignalGenerator(num_samples, frequency, pulse_duration,
                                    toa, do_plot, noise_level)
    amplitude_signal = signal_generator.generate(amplitude=True)
    frequency_signal = signal_generator.generate(amplitude=False)


    qse = QuantumSineWaveEncoder(num_qubits, num_ancilliary, shots)

    pulseOutputProcessor = HeatMapOutputProcessor(HistogramOutputProcessor(
        PulseDetectorOutputProcessor(shots, total_qubits, threshold_percentage,
                                     min_pulse_width)))

    raw_measurements, pulses = qse.run_circuit(qse.pulseDetection,
                                               amplitude_signal,
                                               pulseOutputProcessor)

    print(pulses)
    print((toa, toa+pulse_duration))
    qse = QuantumSineWaveEncoder(total_qubits, 0, shots)

    frequencyOutputProcessor = HeatMapOutputProcessor(HistogramOutputProcessor())

    for start, end in pulses:
        frequency_state_vector = qse.expand_state_vector_from_list(frequency_signal[start:end], 2**total_qubits)

        raw_measurements = qse.run_circuit(
            qse.frequencyEstimationCircuit,
            frequency_state_vector,
            frequencyOutputProcessor)
        frequency_vector = PulseDetectorOutputProcessor.get_probabilities(raw_measurements, shots, total_qubits)

def experiment_4(frequency=100, pd=25):
    """"""
    num_qubits = 9
    num_ancilliary = 1
    shots = 20000
    num_samples = 2**(num_qubits)
    frequency = frequency  # Starting frequency in Hz
    pulse_duration = pd
    toa = random.randint(0, num_samples - pd-2)
    threshold_percentage = 0.3
    min_pulse_width = 15
    noise_level = 0.2
    do_plot = False

    total_qubits = num_ancilliary + num_qubits

    signal_generator = DefaultSignalGenerator(num_samples, frequency, pulse_duration,
                                    toa, do_plot, noise_level)
    amplitude_signal = signal_generator.generate(amplitude=True)
    frequency_signal = signal_generator.generate(amplitude=False)


    qse = QuantumSineWaveEncoder(num_qubits, num_ancilliary, shots)

    pulseOutputProcessor = PulseDetectorOutputProcessor(
            shots, total_qubits, threshold_percentage, min_pulse_width)

    raw_measurements, pulses = qse.run_circuit(qse.pulseDetection,
                                               amplitude_signal,
                                               pulseOutputProcessor)

    # print(pulses)
    qse = QuantumSineWaveEncoder(total_qubits, 0, shots)

    frequencyOutputProcessor = PassThroughOutputProcessor()

    frequencies = []
    if len(pulses) > 0:
        start = pulses[0][0]
        end = pulses[0][1]
        frequency_state_vector = qse.expand_state_vector_from_list(frequency_signal[start:end], 2**total_qubits)

        raw_measurements = qse.run_circuit(
            qse.frequencyEstimationCircuit,
            frequency_state_vector,
            frequencyOutputProcessor)
        frequency_measurement = PulseDetectorOutputProcessor.get_probabilities(raw_measurements, shots, total_qubits)

        sorted_indices = np.argsort(frequency_measurement)
        frequencies = [sorted_indices[-2], sorted_indices[-1]]
        frequencies.sort()
    # print(frequencies)
    return pulses[0] if len(pulses) > 0 else [0, 0], frequencies[0] if len(frequencies) > 0 else 0

def experiment_5():
    results = {}
    for frequency in range(128, 256, 2):
        pulses, frequencies = experiment_4(frequency, 25)
        measured_pd = (pulses[1] - pulses[0])
        measured_frequency = frequencies
        results[frequency] = [measured_pd, measured_frequency]
    x = [expected for expected, measurement in results.items()]
    y = [measurement[1] if measurement != None else 0 for expected, measurement in results.items()]
    plt.scatter(x, y)
    plt.savefig(str(uuid.uuid4()) + 'experiment5.png')
    plt.show()

def experiment_6():
    results = {}
    for pd in range(15, 100, 1):
        pulses, frequencies = experiment_4(100, pd)
        measured_pd = (pulses[1] - pulses[0])
        measured_frequency = frequencies
        results[pd] = [measured_pd, measured_frequency]
    x = [expected for expected, measurement in results.items()]
    y = [measurement[0] if measurement != None else 0 for expected, measurement in results.items()]
    plt.scatter(x, y)
    plt.savefig(str(uuid.uuid4()) + 'experiment6.png')
    plt.show()
if __name__ == '__main__':
    experiment_3()
    experiment_5()
    experiment_6()
