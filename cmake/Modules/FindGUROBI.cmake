find_path(GUROBI_INCLUDE_DIRS
    NAMES gurobi_c.h
    HINTS $ENV{GUROBI_HOME} ${GUROBI_DIR}
    PATH_SUFFIXES include)
mark_as_advanced(GUROBI_INCLUDE_DIRS)

find_library(GUROBI_LIBRARY
    NAMES gurobi gurobi100 gurobi110 gurobi120 gurobi130
    HINTS $ENV{GUROBI_HOME} ${GUROBI_DIR}
    PATH_SUFFIXES lib)

# if (MSVC)
    set(_MSVC_YEAR "2017")

    string(FIND "${CMAKE_MSVC_RUNTIME_LIBRARY}" "MultiThreadedDLL" _isDLL)
    if (_isDLL GREATER -1)
        set(_RUNTIME_SUFFIX "md")
    else()
        set(_RUNTIME_SUFFIX "mt")
    endif()

    message(STATUS "gurobi_c++${_RUNTIME_SUFFIX}${_MSVC_YEAR}")

    find_LIBRARY(GUROBI_CXX_LIBRARY
        NAMES
            gurobi_c++${_RUNTIME_SUFFIX}${_MSVC_YEAR}
        HINTS "${GUROBI_DIR}" $ENV{GUROBI_HOME}
        PATH_SUFFIXES lib
        DOC "Gurobi C++ release library matching runtime"
        CACHE FILEPATH "Gurobi C++ release library"
    )

    find_library(GUROBI_CXX_DEBUG_LIBRARY
        NAMES
            gurobi_c++${_RUNTIME_SUFFIX}d${_MSVC_YEAR}
        HINTS "${GUROBI_DIR}" $ENV{GUROBI_HOME}
        PATH_SUFFIXES lib
        DOC "Gurobi C++ debug library matching runtime"
        CACHE FILEPATH "Gurobi C++ debug library"
    )
# endif()

message(STATUS "FindGUROBI: GUROBI_HOME = '$ENV{GUROBI_HOME}'")
message(STATUS "FindGUROBI: GUROBI_INCLUDE_DIRS = ${GUROBI_INCLUDE_DIRS}")
message(STATUS "FindGUROBI: GUROBI_CXX_LIBRARY   = ${GUROBI_CXX_LIBRARY}")
message(STATUS "FindGUROBI: GUROBI_CXX_DEBUG     = ${GUROBI_CXX_DEBUG_LIBRARY}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GUROBI
    REQUIRED_VARS GUROBI_INCLUDE_DIRS GUROBI_LIBRARY GUROBI_CXX_LIBRARY)
