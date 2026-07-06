# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
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

"""Annotate code ranges through the NVTX (``nvtx``) Python API.

On AMD GPUs the ``hip-python-interop`` package supplies an ``nvtx``
compatibility shim backed by ROCTX (``rocm.bindings.roctx``); elsewhere the
upstream ``nvtx`` package does. The Python source below is identical in both
environments, which is the whole point of the shim: code that already annotates
ranges with ``nvtx`` keeps working on AMD hardware unchanged. Run it under a
ROCm-aware profiler (``rocprof-compute``) to see the ranges;
without a profiler attached the annotations are cheap no-ops.
"""

# [literalinclude-begin]
import time

import nvtx


# `annotate` works as a decorator. `color`/`domain` are accepted for
# source compatibility but have no effect on ROCTX (message only).
@nvtx.annotate("work", color="green", domain="demo")
def do_work():
    time.sleep(0.01)


def main():
    print(f"nvtx enabled: {nvtx.enabled()}")

    do_work()

    # `annotate` also works as a context manager.
    with nvtx.annotate("phase", color="blue"):
        # An instantaneous marker.
        nvtx.mark("checkpoint")
        time.sleep(0.01)

    # Manually nested (per-thread) range.
    nvtx.push_range("manual")
    time.sleep(0.01)
    nvtx.pop_range()

    # Process range (may be started/stopped on different threads).
    range_id = nvtx.start_range("process")
    time.sleep(0.01)
    nvtx.end_range(range_id)

    # Automatic function annotation for a code region.
    profiler = nvtx.Profile()
    profiler.enable()
    do_work()
    profiler.disable()

    # NOTE: counters have no ROCTX equivalent and are no-ops in this shim.
    domain = nvtx.get_domain("demo")
    counter = domain.get_counter("loss", float)
    counter.sample(0.42)  # recorded on NVIDIA; no-op on ROCm

    print("ok")


if __name__ == "__main__":
    main()
