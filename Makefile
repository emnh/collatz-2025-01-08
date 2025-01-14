
CPPFLAGS = -g -O3 -march=native -mtune=native -flto -mavx2

all: collatz_perftest collatz_perftest_parallel collatz_perftest_updated_64bit cuda_test


