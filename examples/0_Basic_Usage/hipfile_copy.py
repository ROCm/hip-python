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
* allocate a device buffer with hipMalloc,
* register it with the hipFile driver,
* open + register an input and output file handle,
* read the input through GPU memory and write to the output,
* compare SHA256 hashes of the two files.

Run with the input/output paths overridden via env vars
``HIPFILE_INPUT`` / ``HIPFILE_OUTPUT`` (the defaults assume the
hipfile in-tree test fixtures at ``/mnt/ais/ext4/``).
"""

__author__ = (
    "Riley Dixon <riley.dixon@amd.com> (original); "
    "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com> (port)"
)

# [literalinclude-begin]
import hashlib
import os
import pathlib

from rocm.bindings.hip import hipMalloc, hipFree

from rocm.hipfile import (
    Driver,
    FileHandle,
    Buffer,
    FileHandleType,
    get_version,
)

hipfile_version = get_version()

input_path = pathlib.Path(
    os.environ.get("HIPFILE_INPUT", "/mnt/ais/ext4/random_2MiB.bin")
)
output_path = pathlib.Path(
    os.environ.get("HIPFILE_OUTPUT", "/mnt/ais/ext4/output.bin")
)

print(f"hipFile Version: {hipfile_version}")
print(f"Driver Use Count Before: {Driver.use_count()}")

# Cap each I/O at the Linux-kernel single-call ceiling. Larger requests
# are silently truncated by the kernel.
size = min(input_path.stat().st_size, 2 * 1024 * 1024 * 1024 - 4 * 1024)

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
