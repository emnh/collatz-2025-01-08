
CPPFLAGS = -g -O3 -march=native -mtune=native -flto -mavx2 -I /usr/local/cuda-12.6/targets/x86_64-linux/include

all: collatz_perftest collatz_perftest_parallel collatz_perftest_updated_64bit cuda_test

cuda_test: cuda_test.cu
	nvcc -g -O3 -arch=sm_75 cuda_test.cu -o cuda_test
