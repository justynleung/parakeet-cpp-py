cmake_minimum_required(VERSION 3.15)

project(parakeet_cpp_py LANGUAGES C CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(WHISPER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp")
set(PYBIND11_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11")

if(NOT EXISTS "${WHISPER_SOURCE_DIR}/CMakeLists.txt" OR NOT EXISTS "${PYBIND11_SOURCE_DIR}/CMakeLists.txt")
    find_package(Git REQUIRED)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" submodule update --init --recursive
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE GIT_SUBMODULE_RESULT
    )
    if(NOT GIT_SUBMODULE_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to initialize pinned git submodules")
    endif()
endif()

if(NOT EXISTS "${WHISPER_SOURCE_DIR}/CMakeLists.txt" OR NOT EXISTS "${PYBIND11_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Pinned submodules are unavailable after initialization.")
endif()

# Build only the library.  The extension uses common-whisper's in-process audio
# decoder, so there is no dependency on whisper.cpp's parakeet-cli executable.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_COMMON_FFMPEG OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/whisper.cpp EXCLUDE_FROM_ALL)

add_library(parakeet_audio_decoder STATIC
    third_party/whisper.cpp/examples/common-whisper.cpp
)
target_include_directories(parakeet_audio_decoder PUBLIC
    "${WHISPER_SOURCE_DIR}/examples"
    "${WHISPER_SOURCE_DIR}/include"
)
target_link_libraries(parakeet_audio_decoder PRIVATE whisper)
set_target_properties(parakeet_audio_decoder PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(third_party/pybind11 EXCLUDE_FROM_ALL)

pybind11_add_module(_parakeet_cpp src/parakeet_cpp/bindings.cpp)
target_compile_features(_parakeet_cpp PRIVATE cxx_std_11)
target_include_directories(_parakeet_cpp PRIVATE "${WHISPER_SOURCE_DIR}/examples")
target_link_libraries(_parakeet_cpp PRIVATE parakeet parakeet_audio_decoder)

install(TARGETS _parakeet_cpp LIBRARY DESTINATION parakeet_cpp)
