PROJECT_ROOT = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

#CC= gcc -O3 -fopenacc
CC = nvcc -fast -acc=host,gpu -gpu=cc60 -Minfo=accel -cuda
#CXX = g++ -O3 -fopenacc
CXX = nvc++ -fast -acc=host,gpu -gpu=cc60 -Minfo=accel -cuda -std=c++17
#CXX = nvc++ -g -O0 -acc=host,gpu -gpu=cc60 -Minfo=accel -cuda -std=c++17

CUDAC=nvcc
#CUDAFLAGS = -g -O0  -gencode arch=compute_60,code=sm_60
CUDAFLAGS = -g -O3  -gencode arch=compute_60,code=sm_60

SRC_DIR = src
BUILD_DIR = build

SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRC_FILES))
LIBS =

all: $(BUILD_DIR)/prefix_sum_acc

#
# Compile
#

$(BUILD_DIR)/prefix_sum_acc: $(OBJ_FILES)
	$(CXX) -o $@ $^ $(LIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) -c -o $@ $<

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(CUDAC) $(CUDAFLAGS) -c -o $@ $<

#
# Clean out object files and the executable.
#
clean:
	rm -fr $(BUILD_DIR)/*.o $(BUILD_DIR)/arith_acc

# Dependency Rules

$(BUILD_DIR)/main.o: $(SRC_DIR)/prefix_sum_acc.hpp
$(BUILD_DIR)/prefix_sum_acc.o: $(SRC_DIR)/prefix_sum_acc.hpp
