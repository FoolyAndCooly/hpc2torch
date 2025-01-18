.PHONY : build clean test

TYPE ?= Release

build:
	mkdir -p build
	cmake -Bbuild -DCMAKE_CUDA_FLAGS="-G" -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc -DCMAKE_BUILD_TYPE=$(TYPE) -DUSE_CUDA=ON
	make -j -C build

clean:
	rm -rf build

test:
	python3 test/gather.py --device cuda
