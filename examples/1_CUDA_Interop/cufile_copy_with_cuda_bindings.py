# MIT License
#
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

"""File copy with the cuda-python ``cuda.bindings.cufile`` API.

Companion to :mod:`hipfile_copy` showing the same file-copy workflow
against the modern ``cuda.bindings.cufile`` import path. On AMD GPUs the
``hip-python-interop`` package supplies the binding (layered on hipFILE); on
NVIDIA the official ``cuda-python`` package does. The Python source is identical
in both environments.

End-to-end demo:
* create a 2 MiB random input file in a temporary directory,
* allocate a device buffer with ``cuda.bindings.runtime`` (cudaMalloc),
* register it with the cuFile driver,
* register the input and output file handles (``O_DIRECT`` file descriptors),
* read the input through GPU memory and write it back out,
* compare SHA256 hashes of the two files.

cuFile issues its I/O with ``O_DIRECT``, which requires an ``O_DIRECT``-capable
filesystem; set ``HIPFILE_TMPDIR`` to such a mount if the default temp dir is
tmpfs.
"""

# [literalinclude-begin]
import hashlib
import os
import pathlib
import tempfile

from cuda.bindings import runtime
from cuda.bindings import cufile


def cuda_check(call_result):
    """Unwrap the ``(error, *results)`` tuple returned by cuda.bindings.runtime."""
    if isinstance(call_result, tuple):
        err = call_result[0]
        result = call_result[1:]
        if len(result) == 1:
            result = result[0]
    else:
        err = call_result
        result = ()
    if (
        isinstance(err, runtime.cudaError_t)
        and err != runtime.cudaError_t.cudaSuccess
    ):
        raise RuntimeError(str(err))
    return result


def _register_handle(fd):
    """Build a cuFile file descriptor for ``fd`` and register it."""
    descr = cufile.Descr()
    descr.type = cufile.FileHandleType.OPAQUE_FD
    descr.handle.fd = fd
    # Keep the Descr alive until the handle is registered (handle_register
    # copies out of it); returning it lets the caller hold the reference.
    return cufile.handle_register(descr.ptr), descr


print(f"cuFile Version: {cufile.get_version()}")

# cuFile issues its reads/writes with O_DIRECT, so the scratch files must live
# on an O_DIRECT-capable filesystem. Default to the system temp dir and allow
# an override via HIPFILE_TMPDIR (e.g. point at an ext4 mount if the default
# temp dir is tmpfs).
scratch_dir = os.environ.get("HIPFILE_TMPDIR") or None

# 2 MiB, block-aligned so the O_DIRECT transfers are valid.
size = 2 * 1024 * 1024

with tempfile.TemporaryDirectory(dir=scratch_dir) as tmp_dir:
    input_path = pathlib.Path(tmp_dir) / "random_2MiB.bin"
    output_path = pathlib.Path(tmp_dir) / "output.bin"

    # Create the random input up front so the example is fully self-contained.
    input_path.write_bytes(os.urandom(size))

    print(f"Driver Use Count Before: {cufile.use_count()}")

    # Allocate the device bounce buffer via the cuda-bindings runtime API.
    dev_ptr = int(cuda_check(runtime.cudaMalloc(size)))
    print(f"Buffer located at: {dev_ptr} | {hex(dev_ptr)}")

    cufile.driver_open()
    fd_input = fd_output = None
    fh_input = fh_output = None
    buffer_registered = False
    try:
        print(f"Driver Use Count After: {cufile.use_count()}")

        cufile.buf_register(dev_ptr, size, 0)
        buffer_registered = True

        fd_input = os.open(input_path, os.O_RDWR | os.O_DIRECT | os.O_CREAT)
        fd_output = os.open(
            output_path, os.O_RDWR | os.O_DIRECT | os.O_CREAT | os.O_TRUNC
        )

        fh_input, _descr_in = _register_handle(fd_input)
        fh_output, _descr_out = _register_handle(fd_output)

        print(f"Transferring {size} bytes...")
        bytes_read = cufile.read(fh_input, dev_ptr, size, 0, 0)
        print(f"Bytes Read: {bytes_read}")
        bytes_written = cufile.write(fh_output, dev_ptr, size, 0, 0)
        print(f"Bytes Written: {bytes_written}")
    finally:
        if fh_output is not None:
            cufile.handle_deregister(fh_output)
        if fh_input is not None:
            cufile.handle_deregister(fh_input)
        if buffer_registered:
            cufile.buf_deregister(dev_ptr)
        if fd_output is not None:
            os.close(fd_output)
        if fd_input is not None:
            os.close(fd_input)
        cufile.driver_close()
        cuda_check(runtime.cudaFree(dev_ptr))

    with open(input_path, "br") as file_in:
        hash_in = hashlib.sha256()
        chunk = file_in.read(1 * 1024 * 1024)
        while len(chunk) != 0:
            hash_in.update(chunk)
            chunk = file_in.read(1 * 1024 * 1024)
        print(f"Input File Hash: {hash_in.hexdigest()}")

    with open(output_path, "br") as file_out:
        hash_out = hashlib.sha256()
        chunk = file_out.read(1 * 1024 * 1024)
        while len(chunk) != 0:
            hash_out.update(chunk)
            chunk = file_out.read(1 * 1024 * 1024)
        print(f"Output File Hash: {hash_out.hexdigest()}")

    assert hash_in.hexdigest() == hash_out.hexdigest(), "file hashes differ"
    print("ok")
