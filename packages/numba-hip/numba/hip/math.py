# MIT License
#
# Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

import math

from numba.core import imputils
import numba.core.typing.templates as typing_templates

from numba.hip import hipdevicelib

typing_registry = typing_templates.Registry()
impl_registry = imputils.Registry()

for name, mathobj in vars(math).items():
    stub = hipdevicelib.stubs.get(name, None)
    if stub != None:
        # register signatures
        typing_registry.register_global(
            val=mathobj,
            typ=typing_templates.make_concrete_template(
                name=f"Hip_math_{name}", key=mathobj, signatures=stub._signatures_
            ),
        )
        # register code generators
        for callgen, numba_parm_types in stub._call_generators_:
            impl_registry.lower(mathobj, *numba_parm_types)(callgen)

# clean up
for k in list(globals().keys()):
    if k not in ("typing_registry", "impl_registry"):
        del globals()[k]
del k
