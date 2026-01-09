# Libreria C+++
project(ml_algorithms LANGUAGES CXX)


# Trova Eigen
find_package(Eigen3 REQUIRED)

# === LISTA DEI FILE SORGENTE CONDIVISI ===
set(ML_SOURCES
    src/algorithms.cpp
    src/linear_regression_one_var.cpp
    src/linear_regression_multi_var.cpp
    src/logistic_regression.cpp
    # Aggiungi altri file sorgente qui
)

set(ML_HEADERS
    include/algorithms.h
    include/linear_regression_one_var.h
    include/src/linear_regression_multi_var.h
    # Aggiungi altri header qui
)

# === LIBRERIA STATICA ===
add_library(ml_algorithms_static STATIC
    ${ML_SOURCES}
)

target_include_directories(ml_algorithms_static 
    PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(ml_algorithms_static PUBLIC Eigen3::Eigen)

set_target_properties(ml_algorithms_static PROPERTIES 
    OUTPUT_NAME "ml_algorithms_static"
)

# === LIBRERIA DINAMICA (SHARED) ===
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
    OUTPUT_NAME "ml_algorithms_shared"
)

# === PER TUTTE LE LIBRERIE ===
set_target_properties(ml_algorithms_static ml_algorithms_shared PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
    CLEAN_DIRECT_OUTPUT 1
)