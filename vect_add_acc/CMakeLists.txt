cmake_minimum_required (VERSION 3.10)

#set(CMAKE_CXX_COMPILER pgc++)
set(CMAKE_CXX_COMPILER nvc++)

find_package(OpenACC REQUIRED)


project (vect_add_acc LANGUAGES CXX)

set(OpenACC_CXX_FLAGS -fast -acc=gpu -gpu=cc60,lineinfo -Minfo=all,intensity,ccff)

add_executable(vect_add_acc main.cpp vect_add_acc.cpp)

target_compile_features(vect_add_acc PRIVATE cxx_std_17)
target_compile_options(vect_add_acc PRIVATE ${OpenACC_CXX_FLAGS})
target_link_libraries(vect_add_acc PRIVATE ${OpenACC_CXX_FLAGS} -lm)
