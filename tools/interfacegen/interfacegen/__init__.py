# MIT License
# 
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
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

__author__ = "Advanced Micro Devices, Inc."

from . import cparser
from . import tree
from . import typehandler
from . import cython
from . import control
from . import doxyparser
from . import support
from . import treefactory

# configure logging
def disable_logging():
    """Disables the logger. Initializes it if it doesn't exist.
    """
    import logging
    logging.getLogger("interfacegen").disabled = True

disable_logging() # init and disable per default

def enable_logging(level = None):
    """Enables the logger for this package.

    Args:
        level:
            The log level. Log output is filtered accordingly.
            Defaults to None which implies logging.INFO.

    Note:
        This configured logger can be retrieved via
        `logging.getLogger("interfacegen")` anywhere.

    Returns:
        The logger used by the application.
    """

    import sys
    import logging
    logger = logging.getLogger("interfacegen")
    logger.disabled = False
    logger.setLevel(logging.INFO if level == None else level)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[%(levelname)s][%(pathname)s:%(lineno)s]%(message)s"))
    logger.addHandler(handler)
    return logger