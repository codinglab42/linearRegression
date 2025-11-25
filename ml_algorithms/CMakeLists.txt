cmake_minimum_required(VERSION 3.15)
project(MLAlgorithms VERSION 1.0.0 LANGUAGES CXX)

# Imposta standard C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export dei comandi
SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# === PYBIND11 INSTALLATO VIA PIP / PYENV ===
# Dì a CMake quale Python usare
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Recupera il path ai file CMake di pybind11 (installato via pip)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")


# Trova Eigen e pybind11
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)

# === LISTA DEI FILE SORGENTE CONDIVISI ===
set(ML_SOURCES
    src/algorithms.cpp
    #src/linear_algebra.cpp
    #src/cost_functions.cpp
    #src/gradient_descent.cpp
    # Aggiungi altri file sorgente qui
)

set(ML_HEADERS
    include/algorithms.h
    #include/linear_algebra.h
    #include/cost_functions.h
    #include/gradient_descent.h
    # Aggiungi altri header qui
)

# === OPCIONE 1: LIBRERIA STATICA (per C++ puro) ===
add_library(ml_algorithms_static STATIC
    ${ML_SOURCES}
)

target_include_directories(ml_algorithms_static 
    PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(ml_algorithms_static PUBLIC Eigen3::Eigen)

# Rinomina l'output per chiarezza
set_target_properties(ml_algorithms_static PROPERTIES
    OUTPUT_NAME ml_algorithms
    CLEAN_DIRECT_OUTPUT 1
)

# === OPCIONE 2: LIBRERIA DINAMICA (SHARED) ===
add_library(ml_algorithms_shared SHARED
    ${ML_SOURCES}
)

target_include_directories(ml_algorithms_shared 
    PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(ml_algorithms_shared PUBLIC Eigen3::Eigen)

# Rinomina l'output shared library
set_target_properties(ml_algorithms_shared PROPERTIES
    OUTPUT_NAME ml_algorithms_shared
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
    CLEAN_DIRECT_OUTPUT 1
)

# === OPCIONE 3: MODULO PYTHON CON PYTHON11 ===
#set(PYTHON_MODULE_SOURCES
#    python_bindings/src_cpp/bindAlgorithms.cpp  # File wrapper per Python
#)

pybind11_add_module(ml_algorithms_py
#    ${PYTHON_MODULE_SOURCES}
    ../python-bindings/src_cpp/bindAlgorithms.cpp  # File wrapper per Python
)

# Il modulo Python linka contro la libreria statica
target_link_libraries(ml_algorithms_py 
    PRIVATE 
    ml_algorithms_static  # Usa la statica per il modulo Python
    Eigen3::Eigen
)

target_include_directories(ml_algorithms_py 
    PRIVATE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# === INSTALLAZIONE (OPZIONALE) ===
# Installa le librerie
install(TARGETS ml_algorithms_static ml_algorithms_shared
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${ML_HEADERS}
    DESTINATION include/ml_algorithms
)

# === TARGET PER COMPILARE TUTTO ===
add_custom_target(build_all
    DEPENDS ml_algorithms_static ml_algorithms_shared ml_algorithms_py
    COMMENT "Building all targets: static lib, shared lib, and Python module"
)

# === CONFIGURAZIONE PER DEBUG/RELEASE ===
# Imposta flags di debug
target_compile_definitions(ml_algorithms_static PRIVATE $<$<CONFIG:Debug>:DEBUG=1>)
target_compile_definitions(ml_algorithms_shared PRIVATE $<$<CONFIG:Debug>:DEBUG=1>)
target_compile_definitions(ml_algorithms_py PRIVATE $<$<CONFIG:Debug>:DEBUG=1>)

# Ottimizzazioni per release
target_compile_options(ml_algorithms_static PRIVATE $<$<CONFIG:Release>:-O3>)
target_compile_options(ml_algorithms_shared PRIVATE $<$<CONFIG:Release>:-O3>)
target_compile_options(ml_algorithms_py PRIVATE $<$<CONFIG:Release>:-O3>)
