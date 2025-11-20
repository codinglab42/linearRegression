cmake_minimum_required(VERSION 3.15)
project(MLAlgorithmsPython VERSION 1.0.0 LANGUAGES CXX)

# Trova Python e pybind11
find_package(Python REQUIRED COMPONENTS Development.Module)
find_package(pybind11 REQUIRED)

# Includi il progetto C++ core
add_subdirectory(../cpp-core cpp_core)

# Crea modulo Python
pybind11_add_module(ml_algorithms
    src/wrapper.cpp
)

# Link alla libreria core e pybind11
target_link_libraries(ml_algorithms 
    PRIVATE 
    ml_algorithms_core
    pybind11::headers
)

# Include directories
target_include_directories(ml_algorithms 
    PRIVATE 
    ../cpp-core/include
    ${Python_INCLUDE_DIRS}
)

# Imposta proprietà del modulo
set_target_properties(ml_algorithms PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)