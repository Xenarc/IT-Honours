#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# Author: mblashki
# GNU Radio version: 3.10.1.1

from packaging.version import Version as StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio.filter import firdes
import sip
from gnuradio import analog
from gnuradio import blocks
from gnuradio import filter
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import quantum


def snipfcn_data_process_and_display(self):
    import matplotlib.pyplot as plt
    import numpy as np

    ###### load

    quantum = np.fromfile(open("/mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/quantum.bin"), dtype=np.float32)
    raw = np.fromfile(open("/mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/raw.bin"), dtype=np.float32)

    # Stats

    print(len(quantum))
    print(len(raw))

    ###### store

    min_length = min(len(quantum), len(raw))
    data = np.column_stack((raw[:min_length], quantum[:min_length]))
    np.savetxt("/mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/data.csv", data, delimiter=",", fmt="%.8f")
    ###### Display

    # Load the data from the CSV file
    data = np.genfromtxt('/mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/data.csv', delimiter=',')

    # Extract the columns
    raw = data[:, 0]
    quantum = data[:, 1]

    # Create the x-axis values
    x = np.arange(len(raw))

    # Plot the data
    plt.plot(x, raw, label='Raw')
    plt.plot(x, quantum, label='Quantum')

    # Set the chart title and labels
    plt.title('Raw and Quantum Data')
    plt.xlabel('Index')
    plt.ylabel('Value')

    # Show a legend
    plt.legend()

    # Display the chart
    plt.show()


def snippets_main_after_stop(tb):
    snipfcn_data_process_and_display(tb)

from gnuradio import qtgui

class noise(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Not titled yet")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "noise")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.max_freq = max_freq = 150_000_000
        self.samp_rate = samp_rate = max_freq*2
        self.SNR = SNR = 1

        ##################################################
        # Blocks
        ##################################################
        self.quantum_quantum_encode_sink_0 = quantum.quantum_encode_sink('basis', 8, 250)
        self.qtgui_time_sink_x_1_0 = qtgui.time_sink_f(
            512, #size
            samp_rate, #samp_rate
            "", #name
            2, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_1_0.set_update_time(0.1)
        self.qtgui_time_sink_x_1_0.set_y_axis(-1, 1)

        self.qtgui_time_sink_x_1_0.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_1_0.enable_tags(True)
        self.qtgui_time_sink_x_1_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_1_0.enable_autoscale(True)
        self.qtgui_time_sink_x_1_0.enable_grid(True)
        self.qtgui_time_sink_x_1_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_1_0.enable_control_panel(True)
        self.qtgui_time_sink_x_1_0.enable_stem_plot(False)


        labels = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(2):
            if len(labels[i]) == 0:
                self.qtgui_time_sink_x_1_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_time_sink_x_1_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_1_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_1_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_1_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_1_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_1_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_1_0_win = sip.wrapinstance(self.qtgui_time_sink_x_1_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_1_0_win)
        self.low_pass_filter_0 = filter.fir_filter_ccf(
            1,
            firdes.low_pass(
                1,
                samp_rate,
                150_000_000,
                1_000_000,
                window.WIN_HAMMING,
                6.76))
        self.blocks_throttle_0_1 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_file_sink_0_0 = blocks.file_sink(gr.sizeof_float*1, '/mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/raw.bin', False)
        self.blocks_file_sink_0_0.set_unbuffered(False)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_float*1, '/mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/quantum.bin', False)
        self.blocks_file_sink_0.set_unbuffered(False)
        self.blocks_complex_to_mag_0 = blocks.complex_to_mag(1)
        self.analog_noise_source_x_0 = analog.noise_source_c(analog.GR_GAUSSIAN, 1/SNR, 0)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_noise_source_x_0, 0), (self.low_pass_filter_0, 0))
        self.connect((self.blocks_complex_to_mag_0, 0), (self.blocks_file_sink_0_0, 0))
        self.connect((self.blocks_complex_to_mag_0, 0), (self.qtgui_time_sink_x_1_0, 1))
        self.connect((self.blocks_throttle_0_1, 0), (self.blocks_complex_to_mag_0, 0))
        self.connect((self.blocks_throttle_0_1, 0), (self.quantum_quantum_encode_sink_0, 0))
        self.connect((self.low_pass_filter_0, 0), (self.blocks_throttle_0_1, 0))
        self.connect((self.quantum_quantum_encode_sink_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.quantum_quantum_encode_sink_0, 0), (self.qtgui_time_sink_x_1_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "noise")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()
        snippets_main_after_stop(self)
        event.accept()

    def get_max_freq(self):
        return self.max_freq

    def set_max_freq(self, max_freq):
        self.max_freq = max_freq
        self.set_samp_rate(self.max_freq*2)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle_0_1.set_sample_rate(self.samp_rate)
        self.low_pass_filter_0.set_taps(firdes.low_pass(1, self.samp_rate, 150_000_000, 1_000_000, window.WIN_HAMMING, 6.76))
        self.qtgui_time_sink_x_1_0.set_samp_rate(self.samp_rate)

    def get_SNR(self):
        return self.SNR

    def set_SNR(self, SNR):
        self.SNR = SNR
        self.analog_noise_source_x_0.set_amplitude(1/self.SNR)




def main(top_block_cls=noise, options=None):

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        snippets_main_after_stop(tb)
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
