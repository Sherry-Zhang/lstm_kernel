IF (APPLE)
    cmake_minimum_required(VERSION 3.4)
ELSE()
    cmake_minimum_required(VERSION 2.8)
ENDIF()

PROJECT(sru)
IF (NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")
ELSE()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
    add_definitions(-DAPPLE)
ENDIF()

OPTION(WITH_OMP "compile intel-sru with openmp." ON)
if(NOT WITH_OMP)
    add_definitions(-DCTC_DISABLE_OMP)
endif()

INCLUDE_DIRECTORIES(include)

ADD_LIBRARY(lstm SHARED src/lstm.cpp)
TARGET_LINK_LIBRARIES(lstm mkl_rt)

ADD_EXECUTABLE(test_lstm tests/test_lstm.cpp)
TARGET_LINK_LIBRARIES(test_lstm lstm)
