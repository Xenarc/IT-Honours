#!/usr/bin/env bash
cd ./gr-quantum/build/
cmake ..
make
make install
ldconfig
