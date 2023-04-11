# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: CW Radar
# GNU Radio version: 3.10.1.1

from gnuradio import analog
from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal







class CW_Radar(gr.hier_block2, Qt.QWidget):
    def __init__(self, CW_Radar_pulse_repetition_interval=1200, CW_Radar_pulse_width=700):
        gr.hier_block2.__init__(
            self, "CW Radar",
                gr.io_signature(0, 0, 0),
                gr.io_signature(1, 1, gr.sizeof_gr_complex*1),
        )

        Qt.QWidget.__init__(self)
        self.top_layout = Qt.QVBoxLayout()
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)
        self.setLayout(self.top_layout)

        ##################################################
        # Parameters
        ##################################################
        self.CW_Radar_pulse_repetition_interval = CW_Radar_pulse_repetition_interval
        self.CW_Radar_pulse_width = CW_Radar_pulse_width

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 32000

        ##################################################
        # Blocks
        ##################################################
        self.blocks_vector_source_x_0_0 = blocks.vector_source_c([1 for i in range(1, CW_Radar_pulse_width)] + [0 for i in range(1, (CW_Radar_pulse_repetition_interval))], True, 1, [])
        self.analog_sig_source_x_0_0 = analog.sig_source_c(samp_rate, analog.GR_SIN_WAVE, 1_000_000, 1, 0, 0)
        self.CW_Radar_Multiply = blocks.multiply_vcc(1)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.CW_Radar_Multiply, 0), (self, 0))
        self.connect((self.analog_sig_source_x_0_0, 0), (self.CW_Radar_Multiply, 0))
        self.connect((self.blocks_vector_source_x_0_0, 0), (self.CW_Radar_Multiply, 1))


    def get_CW_Radar_pulse_repetition_interval(self):
        return self.CW_Radar_pulse_repetition_interval

    def set_CW_Radar_pulse_repetition_interval(self, CW_Radar_pulse_repetition_interval):
        self.CW_Radar_pulse_repetition_interval = CW_Radar_pulse_repetition_interval
        self.blocks_vector_source_x_0_0.set_data([1 for i in range(1, self.CW_Radar_pulse_width)] + [0 for i in range(1, (self.CW_Radar_pulse_repetition_interval))], [])

    def get_CW_Radar_pulse_width(self):
        return self.CW_Radar_pulse_width

    def set_CW_Radar_pulse_width(self, CW_Radar_pulse_width):
        self.CW_Radar_pulse_width = CW_Radar_pulse_width
        self.blocks_vector_source_x_0_0.set_data([1 for i in range(1, self.CW_Radar_pulse_width)] + [0 for i in range(1, (self.CW_Radar_pulse_repetition_interval))], [])

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.analog_sig_source_x_0_0.set_sampling_freq(self.samp_rate)

