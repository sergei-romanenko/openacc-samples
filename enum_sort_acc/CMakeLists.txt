cmake_minimum_required (VERSION 3.10)

#set(CMAKE_CXX_COMPILER pgc++)
set(CMAKE_CXX_COMPILER nvc++)

find_package(OpenACC REQUIRED)

project (enum_sort_acc LANGUAGES CXX)

set(OpenACC_CXX_FLAGS -fast -acc=host,gpu -gpu=cc60,lineinfo -Minfo=accel)
#set(OpenACC_CXX_FLAGS -g -O0 -acc=host,gpu -gpu=cc60 -Minfo=accel)

add_executable(enum_sort_acc main.cpp enum_sort_acc.cpp)

target_compile_features(enum_sort_acc PRIVATE cxx_std_17)
target_compile_options(enum_sort_acc PRIVATE ${OpenACC_CXX_FLAGS})
target_link_libraries(enum_sort_acc PRIVATE ${OpenACC_CXX_FLAGS} -lm)
