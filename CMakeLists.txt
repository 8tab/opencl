cmake_minimum_required(VERSION 2.8)
project(opencl)

list(APPEND libraries common)
list(APPEND libraries OpenCL)
set(pwd "${CMAKE_CURRENT_LIST_DIR}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I${pwd}/common -Wall -Wextra -Werror -Wno-deprecated-declarations -Wsign-compare -Wconversion")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${pwd}/common -std=c++11 -Wall -Wextra -Werror -Wno-deprecated-declarations -Wsign-compare -Wconversion")


add_subdirectory(common)
add_subdirectory(platform_info)
