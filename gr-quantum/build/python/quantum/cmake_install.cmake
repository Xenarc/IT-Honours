# Install script for directory: /mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/gr-quantum/python/quantum

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3.10/dist-packages/gnuradio/quantum" TYPE FILE FILES
    "/mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/gr-quantum/python/quantum/__init__.py"
    "/mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/gr-quantum/python/quantum/quantum_in_out.py"
    "/mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/gr-quantum/python/quantum/quantum_encode_sink.py"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3.10/dist-packages/gnuradio/quantum" TYPE FILE FILES
    "/mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/gr-quantum/build/python/quantum/__init__.pyc"
    "/mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/gr-quantum/build/python/quantum/quantum_in_out.pyc"
    "/mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/gr-quantum/build/python/quantum/quantum_encode_sink.pyc"
    "/mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/gr-quantum/build/python/quantum/__init__.pyo"
    "/mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/gr-quantum/build/python/quantum/quantum_in_out.pyo"
    "/mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/gr-quantum/build/python/quantum/quantum_encode_sink.pyo"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/gr-quantum/build/python/quantum/bindings/cmake_install.cmake")

endif()

