cmake_minimum_required(VERSION 3.15)

project(parakeet_cpp_py LANGUAGES CXX)

set(WHISPER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp")
set(PYBIND11_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11")
set(WHISPER_BUILD_DIR "${WHISPER_SOURCE_DIR}/build")
set(PARAKEET_CLI "${WHISPER_BUILD_DIR}/bin/parakeet-cli")
set(PARAKEET_MODEL "${WHISPER_SOURCE_DIR}/models/ggml-parakeet-tdt-0.6b-v3-q8_0.bin")

if(NOT EXISTS "${WHISPER_SOURCE_DIR}/CMakeLists.txt" OR NOT EXISTS "${PYBIND11_SOURCE_DIR}/CMakeLists.txt")
    find_package(Git REQUIRED)
    message(STATUS "Initializing pinned git submodules")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" submodule update --init --recursive
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE GIT_SUBMODULE_RESULT
        OUTPUT_VARIABLE GIT_SUBMODULE_OUTPUT
        ERROR_VARIABLE GIT_SUBMODULE_ERROR
    )
    if(NOT GIT_SUBMODULE_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to initialize git submodules:\n${GIT_SUBMODULE_OUTPUT}${GIT_SUBMODULE_ERROR}")
    endif()
endif()

if(NOT EXISTS "${WHISPER_SOURCE_DIR}/CMakeLists.txt" OR NOT EXISTS "${PYBIND11_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Pinned submodules are unavailable after initialization.")
endif()

# Apply local patches to whisper.cpp before building
set(PATCH_FILE "${CMAKE_CURRENT_SOURCE_DIR}/patches/0001-parakeet-windowed-streaming.patch")
if(EXISTS "${PATCH_FILE}")
    message(STATUS "Applying patch: ${PATCH_FILE}")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply "${PATCH_FILE}"
        WORKING_DIRECTORY "${WHISPER_SOURCE_DIR}"
        RESULT_VARIABLE PATCH_RESULT
        OUTPUT_VARIABLE PATCH_OUTPUT
        ERROR_VARIABLE PATCH_ERROR
    )
    if(NOT PATCH_RESULT EQUAL 0)
        # Patch may already be applied — check with --check
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" apply --check "${PATCH_FILE}"
            WORKING_DIRECTORY "${WHISPER_SOURCE_DIR}"
            RESULT_VARIABLE PATCH_CHECK_RESULT
        )
        if(NOT PATCH_CHECK_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to apply patch:\n${PATCH_OUTPUT}${PATCH_ERROR}")
        else()
            message(STATUS "Patch already applied, skipping.")
        endif()
    endif()
endif()

find_program(HF_EXECUTABLE hf REQUIRED)

add_custom_target(build_parakeet_cli
    COMMAND "${CMAKE_COMMAND}" -B build -S .
    COMMAND "${CMAKE_COMMAND}" --build build --target parakeet-cli -j 12
    BYPRODUCTS "${PARAKEET_CLI}"
    WORKING_DIRECTORY "${WHISPER_SOURCE_DIR}"
    COMMENT "Building the pinned parakeet-cli"
    VERBATIM
)

add_custom_command(
    OUTPUT "${PARAKEET_MODEL}"
    COMMAND "${HF_EXECUTABLE}" download ggml-org/parakeet-GGUF ggml-parakeet-tdt-0.6b-v3-q8_0.bin --local-dir models
    WORKING_DIRECTORY "${WHISPER_SOURCE_DIR}"
    DEPENDS build_parakeet_cli
    COMMENT "Downloading the pinned Parakeet Q8 model"
    VERBATIM
)

add_custom_target(download_parakeet_model DEPENDS "${PARAKEET_MODEL}")

add_subdirectory(third_party/pybind11 EXCLUDE_FROM_ALL)

pybind11_add_module(_parakeet_cpp src/parakeet_cpp/bindings.cpp)
target_compile_features(_parakeet_cpp PRIVATE cxx_std_11)
add_dependencies(_parakeet_cpp download_parakeet_model)

install(TARGETS _parakeet_cpp LIBRARY DESTINATION parakeet_cpp)
