cmake_minimum_required(VERSION 3.15)
project(LinearRegression LANGUAGES CXX)

# Imposta standard C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export dei comandi
SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Sub directory di progetto
add_subdirectory(ml_algorithms)
add_subdirectory(pyBindAlgorithms)
