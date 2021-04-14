cmake_minimum_required (VERSION 3.10)

#set(CMAKE_CXX_COMPILER pgc++)
set(CMAKE_CXX_COMPILER nvc++)

find_package(OpenACC REQUIRED)

project (enum_sort_acc LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

set(OpenACC_CXX_FLAGS -fast -acc=host,gpu -gpu=cc60,lineinfo -Minfo=accel)
#set(OpenACC_CXX_FLAGS -g -O0 -acc=host,gpu -gpu=cc60 -Minfo=accel)

add_executable(prefix_sum_acc main.cpp prefix_sum_acc.cpp prefix_sum_acc.hpp)

target_compile_features(prefix_sum_acc PRIVATE cxx_std_17)
target_compile_options(prefix_sum_acc PRIVATE ${OpenACC_CXX_FLAGS})
target_link_libraries(prefix_sum_acc PRIVATE ${OpenACC_CXX_FLAGS} -lm)
