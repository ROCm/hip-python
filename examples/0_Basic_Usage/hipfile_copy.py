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

# Ported from ROCm/rocm-systems projects/hipfile/python/main.py (commit
# cbbf349092, "[hipFile] Polish the python project prior to early release on
# PyPI", #5089) — original author Riley Dixon <riley.dixon@amd.com>. The
# port re-routes imports from `hipfile.*` to `rocm.hipfile.*` and replaces
# the upstream `hipfile.hipMalloc` ctypes-based hack with hip-python's
# native `rocm.bindings.hip.hipMalloc` / `hipFree`.

"""Copy a file via GPU memory using hipFile.

End-to-end demo:
* create a 2 MiB random input file in a temporary directory,
* allocate a device buffer with hipMalloc,
* register it with the hipFile driver,
* open + register the input and output file handles,
* read the input through GPU memory and write to the output,
* compare SHA256 hashes of the two files.

The scratch files are created and removed during the run, so no
pre-existing fixture is needed. hipFile issues its I/O with
``O_DIRECT``, which requires an ``O_DIRECT``-capable filesystem; set
``HIPFILE_TMPDIR`` to such a mount if the default temp dir is tmpfs.
"""

__author__ = (
    "Riley Dixon <riley.dixon@amd.com> (original); "
    "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com> (port)"
)

# [literalinclude-begin]
import hashlib
import os
import pathlib
import tempfile

from rocm.bindings.hip import hipMalloc, hipFree

from rocm.hipfile import (
    Driver,
    FileHandle,
    Buffer,
    FileHandleType,
    get_version,
)

hipfile_version = get_version()
print(f"hipFile Version: {hipfile_version}")

# hipFile issues its reads/writes with O_DIRECT, so the scratch files must
# live on an O_DIRECT-capable filesystem. Default to the system temp dir and
# allow an override via HIPFILE_TMPDIR (e.g. point at an ext4 mount if the
# default temp dir is tmpfs).
scratch_dir = os.environ.get("HIPFILE_TMPDIR") or None

# 2 MiB, block-aligned so the O_DIRECT transfers are valid.
size = 2 * 1024 * 1024

with tempfile.TemporaryDirectory(dir=scratch_dir) as tmp_dir:
    input_path = pathlib.Path(tmp_dir) / "random_2MiB.bin"
    output_path = pathlib.Path(tmp_dir) / "output.bin"

    # Create the random input up front instead of relying on a pre-existing
    # file, so the example is fully self-contained.
    input_path.write_bytes(os.urandom(size))

    print(f"Driver Use Count Before: {Driver.use_count()}")

    # rocm.bindings.hip.hipMalloc returns (err, DeviceArray). Unpack the
    # error-tuple shape the auto-generated wrapper uses, then take the
    # device pointer from the array's __int__.
    err, dev_array = hipMalloc(size)
    assert int(err) == 0, f"hipMalloc failed: {err}"
    buffer_ptr = int(dev_array)
    print(f"Buffer located at: {buffer_ptr} | {hex(buffer_ptr)}")

    with Driver() as hipfile_driver:
        print(f"Driver Use Count After: {hipfile_driver.use_count()}")
        with Buffer(buffer_ptr, size, 0) as registered_buffer:
            with FileHandle(
                input_path,
                os.O_RDWR | os.O_DIRECT | os.O_CREAT,
                handle_type=FileHandleType.OPAQUE_FD,
            ) as fh_input:
                with FileHandle(
                    output_path,
                    os.O_RDWR | os.O_DIRECT | os.O_CREAT | os.O_TRUNC,
                ) as fh_output:
                    print(f"Transferring {size} bytes...")
                    bytes_read = fh_input.read(registered_buffer, size, 0, 0)
                    print(f"Bytes Read: {bytes_read}")
                    bytes_written = fh_output.write(registered_buffer, size, 0, 0)
                    print(f"Bytes Written: {bytes_written}")

    free_err = hipFree(dev_array)
    assert int(free_err) == 0, f"hipFree failed: {free_err}"

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
