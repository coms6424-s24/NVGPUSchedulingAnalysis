CC = g++
NVCC ?= nvcc
CFLAGS = -fPIC -std=c++17
LDFLAGS = -lcuda -I/usr/local/cuda/include
RPATH = -Wl,-rpath,.
NVCC_RPATH = -Xlinker -rpath -Xlinker .

.PHONY: clean tests

all: gpu_sm_masking.so

gpu_sm_masking.so: gpu_sm_masking.cpp gpu_sm_masking.hpp
	$(CC) $< -shared -o $@ $(CFLAGS) $(LDFLAGS)

matrix_mul_partitioning: matrix_mul_partitioning.cu gpu_sm_masking.so
	$(NVCC) $< -o $@ -g -L. ./gpu_sm_masking.so $(LDFLAGS) $(NVCC_RPATH)

matrix_mul_test: matrix_mul_test.cu gpu_sm_masking.so
	$(NVCC) $< -o $@ -g -L. ./gpu_sm_masking.so $(LDFLAGS) $(NVCC_RPATH)

matrix_mul_partitioning_gpc: matrix_mul_partitioning_gpc.cu gpu_sm_masking.so
	$(NVCC) $< -o $@ -g -L. ./gpu_sm_masking.so $(LDFLAGS) $(NVCC_RPATH)

matrix_mul_resource_constraint: matrix_mul_resource_constraint.cu gpu_sm_masking.so
	$(NVCC) $< -o $@ -g -L. ./gpu_sm_masking.so $(LDFLAGS) $(NVCC_RPATH)

tests:  matrix_mul_partitioning matrix_mul_test \
 		matrix_mul_partitioning_gpc matrix_mul_resource_constraint

clean:  
	rm -f gpu_sm_masking.so  matrix_mul_partitioning \
		matrix_mul_test matrix_mul_partitioning_gpc matrix_mul_resource_constraint
