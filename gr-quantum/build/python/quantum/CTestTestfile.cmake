# CMake generated Testfile for 
# Source directory: /mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/gr-quantum/python/quantum
# Build directory: /mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/gr-quantum/build/python/quantum
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(qa_quantum_encode_sink "/usr/bin/sh" "qa_quantum_encode_sink_test.sh")
set_tests_properties(qa_quantum_encode_sink PROPERTIES  _BACKTRACE_TRIPLES "/usr/lib/x86_64-linux-gnu/cmake/gnuradio/GrTest.cmake;116;add_test;/mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/gr-quantum/python/quantum/CMakeLists.txt;43;GR_ADD_TEST;/mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/gr-quantum/python/quantum/CMakeLists.txt;0;")
subdirs("bindings")
