cmake_minimum_required(VERSION 3.15)
project(MLAlgorithmsCore VERSION 1.0.0 LANGUAGES CXX)

# Imposta standard C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Directory include
target_include_directories(ml_algorithms_core 
    PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Crea libreria statica
add_library(ml_algorithms_core STATIC
    src/algorithms.cpp
    src/linear_regression.cpp
    # Aggiungi altri file sorgente qui
)

# Test eseguibile (opzionale)
enable_testing()
add_executable(test_algorithms tests/test_linear.cpp)
target_link_libraries(test_algorithms ml_algorithms_core)

add_test(NAME test_linear_regression COMMAND test_algorithms)

# Installazione (opzionale)
install(TARGETS ml_algorithms_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)