cmake_minimum_required(VERSION 3.15)
project(LinearRegression LANGUAGES CXX)

# Imposta standard C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export dei comandi
SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# Directory per output delle librerie
# ===== CONFIGURAZIONE OUTPUT =====
# Directory di output per TUTTE le librerie
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(LIB_OUTPUT_DIR ${CMAKE_BINARY_DIR}/lib)
set(BIN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIB_OUTPUT_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LIB_OUTPUT_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BIN_OUTPUT_DIR})

# Crea directory lib nel progetto
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)


# Sub directory di progetto
add_subdirectory(ml_algorithms)
add_subdirectory(pyBindAlgorithms)

# Messaggio informativo
message(STATUS "Librerie saranno create in: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "Librerie copiate in: ${CMAKE_SOURCE_DIR}/lib")

# ===== POST-BUILD: COPIA IN PROJECT LIB/ =====
# Questa macro copierà automaticamente tutte le librerie
macro(copy_libraries_to_project)
    add_custom_target(copy_all_libs ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/lib
        COMMAND ${CMAKE_COMMAND} -E echo "Copia librerie in ${CMAKE_SOURCE_DIR}/lib/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIB_OUTPUT_DIR}/*.a ${CMAKE_SOURCE_DIR}/lib/
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIB_OUTPUT_DIR}/*.so ${CMAKE_SOURCE_DIR}/lib/
        COMMAND ${CMAKE_COMMAND} -E echo "Librerie copiate!"
        DEPENDS ml_algorithms_static ml_algorithms_shared pymlalgorithms
        COMMENT "Copia tutte le librerie nella directory del progetto"
    )
endmacro()

copy_libraries_to_project()
