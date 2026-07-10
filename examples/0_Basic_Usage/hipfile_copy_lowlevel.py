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

# Low-level counterpart of hipfile_copy.py. Where that example uses the
# high-level ``rocm.hipfile`` context managers, this one calls the
# auto-generated ``rocm.bindings.hipfile`` functions directly, so it doubles as
# a reference for the raw (retval, error) tuple shapes and the manual resource
# teardown the wrappers otherwise hide.

"""Copy a file via GPU memory using the low-level ``rocm.bindings.hipfile`` API.

Same end-to-end flow as ``hipfile_copy.py``, but driving the auto-generated
bindings directly instead of the :py:mod:`rocm.hipfile` wrappers:

* ``hipFileDriverOpen`` / ``hipFileDriverClose`` bracket the driver,
* ``hipFileBufRegister`` / ``hipFileBufDeregister`` register the GPU buffer,
* ``hipFileHandleRegister`` / ``hipFileHandleDeregister`` register the files,
* ``hipFileRead`` / ``hipFileWrite`` move bytes through GPU memory.

This surfaces the raw return shapes. Every generated wrapper returns a tuple,
so error-only calls hand back a 1-tuple ``(hipFileError,)`` (the struct
carries ``.err`` / ``.hip_drv_err``), ``hipFileGetVersion`` returns
``(hipFileError, major, minor, patch)``, and ``hipFileRead`` / ``hipFileWrite``
return ``(retval, errno, hip_drv_err)``. It also shows the ``hipFileDescr``
union handling that the high-level classes encapsulate.

The scratch files are created and removed during the run, so no pre-existing
fixture is needed. hipFile issues its I/O with ``O_DIRECT``, which requires an
``O_DIRECT``-capable filesystem; set ``HIPFILE_TMPDIR`` to such a mount if the
default temp dir is tmpfs.
"""

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

# [literalinclude-begin]
import hashlib
import os
import pathlib
import stat
import tempfile

from rocm.bindings.hip import hipMalloc, hipFree

from rocm.bindings.hipfile import (
    hipFileBufDeregister,
    hipFileBufRegister,
    hipFileDescr,
    hipFileDriverClose,
    hipFileDriverOpen,
    hipFileFileHandleType,
    hipFileGetVersion,
    hipFileHandleDeregister,
    hipFileHandleRegister,
    hipFileOpError,
    hipFileRead,
    hipFileUseCount,
    hipFileWrite,
)


_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH


def check(err, what):
    """Raise on a non-success ``hipFileError`` return."""
    if err.err != hipFileOpError.hipFileSuccess:
        raise RuntimeError(
            f"{what} failed: {hipFileOpError(err.err)!r} "
            f"(hip_drv_err={int(err.hip_drv_err)})"
        )


def check_io(result, what):
    """Interpret a ``(retval, errno, hip_drv_err)`` read/write result."""
    n, err_no, hip_drv_err = result
    if n == -1:
        raise OSError(err_no, os.strerror(err_no))
    if n < -1:
        raise RuntimeError(
            f"{what} failed: {hipFileOpError(-n)!r} (hip_drv_err={hip_drv_err})"
        )
    return n


def register_file(path, flags):
    """``os.open`` + ``hipFileHandleRegister``; returns ``(fd, handle)``."""
    fd = os.open(path, flags, _FILE_MODE)
    # The generated hipFileDescr wrapper is fully settable: the constructor
    # zero-inits the buffer (fs_ops stays NULL), `.type` takes a
    # hipFileFileHandleType, and `.handle.fd` writes the POSIX union member
    # in place — no ctypes struct needed.
    descr = hipFileDescr()
    descr.type = hipFileFileHandleType.hipFileHandleTypeOpaqueFD
    descr.handle.fd = fd
    err, fh = hipFileHandleRegister(descr)
    try:
        check(err, "hipFileHandleRegister")
    except Exception:
        os.close(fd)
        raise
    return fd, fh


version_err, major, minor, patch = hipFileGetVersion()
check(version_err, "hipFileGetVersion")
print(f"hipFile Version: {major}.{minor}.{patch}")

# hipFile issues its reads/writes with O_DIRECT, so the scratch files must live
# on an O_DIRECT-capable filesystem. Default to the system temp dir; override
# via HIPFILE_TMPDIR (e.g. an ext4 mount) if the default temp dir is tmpfs.
scratch_dir = os.environ.get("HIPFILE_TMPDIR") or None

# 2 MiB, block-aligned so the O_DIRECT transfers are valid.
size = 2 * 1024 * 1024

with tempfile.TemporaryDirectory(dir=scratch_dir) as tmp_dir:
    input_path = pathlib.Path(tmp_dir) / "random_2MiB.bin"
    output_path = pathlib.Path(tmp_dir) / "output.bin"

    # Create the random input up front so the example is fully self-contained.
    input_path.write_bytes(os.urandom(size))

    # rocm.bindings.hip.hipMalloc returns (err, DeviceArray). Take the device
    # pointer from the array's __int__.
    err, dev_array = hipMalloc(size)
    assert int(err) == 0, f"hipMalloc failed: {err}"
    buffer_ptr = int(dev_array)
    print(f"Buffer located at: {buffer_ptr} | {hex(buffer_ptr)}")

    try:
        # Error-only calls come back as a 1-tuple ``(hipFileError,)``.
        (open_err,) = hipFileDriverOpen()
        check(open_err, "hipFileDriverOpen")
        # hipFileUseCount returns int64_t, also wrapped as a 1-tuple.
        (use_count,) = hipFileUseCount()
        print(f"Driver Use Count: {use_count}")
        try:
            (reg_err,) = hipFileBufRegister(buffer_ptr, size, 0)
            check(reg_err, "hipFileBufRegister")
            try:
                in_fd, in_fh = register_file(
                    input_path, os.O_RDWR | os.O_DIRECT | os.O_CREAT
                )
                try:
                    out_fd, out_fh = register_file(
                        output_path,
                        os.O_RDWR | os.O_DIRECT | os.O_CREAT | os.O_TRUNC,
                    )
                    try:
                        print(f"Transferring {size} bytes...")
                        bytes_read = check_io(
                            hipFileRead(in_fh, buffer_ptr, size, 0, 0),
                            "hipFileRead",
                        )
                        print(f"Bytes Read: {bytes_read}")
                        bytes_written = check_io(
                            hipFileWrite(out_fh, buffer_ptr, size, 0, 0),
                            "hipFileWrite",
                        )
                        print(f"Bytes Written: {bytes_written}")
                    finally:
                        hipFileHandleDeregister(out_fh)
                        os.close(out_fd)
                finally:
                    hipFileHandleDeregister(in_fh)
                    os.close(in_fd)
            finally:
                (dereg_err,) = hipFileBufDeregister(buffer_ptr)
                check(dereg_err, "hipFileBufDeregister")
        finally:
            (close_err,) = hipFileDriverClose()
            check(close_err, "hipFileDriverClose")
    finally:
        # hipFree returns only an error, wrapped as a 1-tuple.
        (free_err,) = hipFree(dev_array)
        assert int(free_err) == 0, f"hipFree failed: {free_err}"

    # Hash both files to confirm the GPU round-trip preserved the bytes.
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
