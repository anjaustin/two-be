cmake_minimum_required(VERSION 3.14)
project(bitswitch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(STATUS "Building for ARM64 (Jetson)")
    set(ARCH_FLAGS -march=armv8.2-a+fp16)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    message(STATUS "Building for ARM32")
    set(ARCH_FLAGS -mfpu=neon -mfloat-abi=hard)
else()
    message(STATUS "Building for x86_64")
    set(ARCH_FLAGS -mavx2 -mfma)
endif()

add_library(bitswitch SHARED bitswitch.cpp)
target_compile_options(bitswitch PRIVATE 
    ${ARCH_FLAGS}
    -ffast-math 
    -funroll-loops
)

install(TARGETS bitswitch LIBRARY DESTINATION lib)
install(FILES bitswitch.h DESTINATION include)
