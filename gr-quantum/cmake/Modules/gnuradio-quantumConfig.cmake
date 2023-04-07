find_package(PkgConfig)

PKG_CHECK_MODULES(PC_GR_QUANTUM gnuradio-quantum)

FIND_PATH(
    GR_QUANTUM_INCLUDE_DIRS
    NAMES gnuradio/quantum/api.h
    HINTS $ENV{QUANTUM_DIR}/include
        ${PC_QUANTUM_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    GR_QUANTUM_LIBRARIES
    NAMES gnuradio-quantum
    HINTS $ENV{QUANTUM_DIR}/lib
        ${PC_QUANTUM_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/gnuradio-quantumTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GR_QUANTUM DEFAULT_MSG GR_QUANTUM_LIBRARIES GR_QUANTUM_INCLUDE_DIRS)
MARK_AS_ADVANCED(GR_QUANTUM_LIBRARIES GR_QUANTUM_INCLUDE_DIRS)
