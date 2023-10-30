import os

from .config import DATALAYOUT, AMDGCN_BC_DATALAYOUT, TRIPLE

# Allow user to use "NUMBA_USE_LIBHLC" env-var to use cmdline HLC.
# if os.environ.get('NUMBA_USE_LIBHLC', '').lower() in ['1', 'yes', 'true']:
#     from numba.roc.hlc import libhlc as hlc