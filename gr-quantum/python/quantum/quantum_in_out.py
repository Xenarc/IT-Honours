#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2023 Mark Blashki.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np
from gnuradio import gr

class quantum_in_out(gr.sync_block):
    """
    docstring for block quantum_in_out
    """
    def __init__(self):        
        gr.sync_block.__init__(self,
            name="quantum_in_out",
            in_sig=[ np.complex64 ],
            out_sig=[ np.complex64 ])


    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]
        # TODO: Add signal processing here
        out[:] = in0
        return len(output_items[0])
