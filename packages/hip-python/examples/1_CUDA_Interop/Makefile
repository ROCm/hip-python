# AMD_COPYRIGHT

# [literalinclude-begin]
.PHONY: build run

build:
	python3 setup.py build_ext --inplace
run: build
	python3 -c "import ccuda_stream"
clean:
	rm -rf *.so *.c build/