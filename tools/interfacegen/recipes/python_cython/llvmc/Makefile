# TODO: Replace by CMake file

SHELL=/usr/bin/bash

C_HEADERS ?= $(shell find /opt/rocm/llvm/include/llvm-c/ -name "*.h")
STATIC_LIBS ?= $(shell find /opt/rocm/llvm/lib -name "libLLVM*.a") 

llvm-c.h: $(C_HEADERS)
	printf "" > $@
	for f in $(C_HEADERS); do echo "#include \"$$f\"" >> $@; done

libLLVM.so: $(STATIC_LIBS)
	gcc -shared -Wl,--whole-archive $^ -Wl,--no-whole-archive -o $@
