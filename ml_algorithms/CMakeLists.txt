# Libreria C+++
project(ml_algorithms LANGUAGES CXX)


# Trova Eigen
find_package(Eigen3 REQUIRED)

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

# === LIBRERIA STATICA ===
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
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME ml_algorithms
    CLEAN_DIRECT_OUTPUT 1
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
    OUTPUT_NAME ml_algorithms_shared
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
    CLEAN_DIRECT_OUTPUT 1
)
