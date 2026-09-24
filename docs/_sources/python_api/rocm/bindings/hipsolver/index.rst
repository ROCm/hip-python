rocm.bindings.hipsolver
=======================

.. py:module:: rocm.bindings.hipsolver


Attributes
----------

.. autoapisummary::

   rocm.bindings.hipsolver.hipsolverVersionMajor
   rocm.bindings.hipsolver.hipsolverVersionMinor
   rocm.bindings.hipsolver.hipsolverVersionPatch
   rocm.bindings.hipsolver.hipsolverOperation_t
   rocm.bindings.hipsolver.hipsolverFillMode_t
   rocm.bindings.hipsolver.hipsolverSideMode_t


Classes
-------

.. autoapisummary::

   rocm.bindings.hipsolver.hipsolverStatus_t
   rocm.bindings.hipsolver.hipsolverEigMode_t
   rocm.bindings.hipsolver.hipsolverEigType_t
   rocm.bindings.hipsolver.hipsolverEigRange_t
   rocm.bindings.hipsolver.hipsolverDeterministicMode_t
   rocm.bindings.hipsolver.hipsolverAlgMode_t
   rocm.bindings.hipsolver.hipsolverDnFunction_t
   rocm.bindings.hipsolver.hipsolverRfFactorization_t
   rocm.bindings.hipsolver.hipsolverRfMatrixFormat_t
   rocm.bindings.hipsolver.hipsolverRfNumericBoostReport_t
   rocm.bindings.hipsolver.hipsolverRfResetValuesFastMode_t
   rocm.bindings.hipsolver.hipsolverRfTriangularSolve_t
   rocm.bindings.hipsolver.hipsolverRfUnitDiagonal_t


Functions
---------

.. autoapisummary::

   rocm.bindings.hipsolver.has_symbol
   rocm.bindings.hipsolver.hipsolverCreate
   rocm.bindings.hipsolver.hipsolverDestroy
   rocm.bindings.hipsolver.hipsolverSetStream
   rocm.bindings.hipsolver.hipsolverGetStream
   rocm.bindings.hipsolver.hipsolverSetDeterministicMode
   rocm.bindings.hipsolver.hipsolverGetDeterministicMode
   rocm.bindings.hipsolver.hipsolverCreateGesvdjInfo
   rocm.bindings.hipsolver.hipsolverDestroyGesvdjInfo
   rocm.bindings.hipsolver.hipsolverXgesvdjSetMaxSweeps
   rocm.bindings.hipsolver.hipsolverXgesvdjSetSortEig
   rocm.bindings.hipsolver.hipsolverXgesvdjSetTolerance
   rocm.bindings.hipsolver.hipsolverXgesvdjGetResidual
   rocm.bindings.hipsolver.hipsolverXgesvdjGetSweeps
   rocm.bindings.hipsolver.hipsolverCreateSyevjInfo
   rocm.bindings.hipsolver.hipsolverDestroySyevjInfo
   rocm.bindings.hipsolver.hipsolverXsyevjSetMaxSweeps
   rocm.bindings.hipsolver.hipsolverXsyevjSetSortEig
   rocm.bindings.hipsolver.hipsolverXsyevjSetTolerance
   rocm.bindings.hipsolver.hipsolverXsyevjGetResidual
   rocm.bindings.hipsolver.hipsolverXsyevjGetSweeps
   rocm.bindings.hipsolver.hipsolverSorgbr_bufferSize
   rocm.bindings.hipsolver.hipsolverDorgbr_bufferSize
   rocm.bindings.hipsolver.hipsolverCungbr_bufferSize
   rocm.bindings.hipsolver.hipsolverZungbr_bufferSize
   rocm.bindings.hipsolver.hipsolverSorgbr
   rocm.bindings.hipsolver.hipsolverDorgbr
   rocm.bindings.hipsolver.hipsolverCungbr
   rocm.bindings.hipsolver.hipsolverZungbr
   rocm.bindings.hipsolver.hipsolverSorgqr_bufferSize
   rocm.bindings.hipsolver.hipsolverDorgqr_bufferSize
   rocm.bindings.hipsolver.hipsolverCungqr_bufferSize
   rocm.bindings.hipsolver.hipsolverZungqr_bufferSize
   rocm.bindings.hipsolver.hipsolverSorgqr
   rocm.bindings.hipsolver.hipsolverDorgqr
   rocm.bindings.hipsolver.hipsolverCungqr
   rocm.bindings.hipsolver.hipsolverZungqr
   rocm.bindings.hipsolver.hipsolverSorgtr_bufferSize
   rocm.bindings.hipsolver.hipsolverDorgtr_bufferSize
   rocm.bindings.hipsolver.hipsolverCungtr_bufferSize
   rocm.bindings.hipsolver.hipsolverZungtr_bufferSize
   rocm.bindings.hipsolver.hipsolverSorgtr
   rocm.bindings.hipsolver.hipsolverDorgtr
   rocm.bindings.hipsolver.hipsolverCungtr
   rocm.bindings.hipsolver.hipsolverZungtr
   rocm.bindings.hipsolver.hipsolverSormqr_bufferSize
   rocm.bindings.hipsolver.hipsolverDormqr_bufferSize
   rocm.bindings.hipsolver.hipsolverCunmqr_bufferSize
   rocm.bindings.hipsolver.hipsolverZunmqr_bufferSize
   rocm.bindings.hipsolver.hipsolverSormqr
   rocm.bindings.hipsolver.hipsolverDormqr
   rocm.bindings.hipsolver.hipsolverCunmqr
   rocm.bindings.hipsolver.hipsolverZunmqr
   rocm.bindings.hipsolver.hipsolverSormtr_bufferSize
   rocm.bindings.hipsolver.hipsolverDormtr_bufferSize
   rocm.bindings.hipsolver.hipsolverCunmtr_bufferSize
   rocm.bindings.hipsolver.hipsolverZunmtr_bufferSize
   rocm.bindings.hipsolver.hipsolverSormtr
   rocm.bindings.hipsolver.hipsolverDormtr
   rocm.bindings.hipsolver.hipsolverCunmtr
   rocm.bindings.hipsolver.hipsolverZunmtr
   rocm.bindings.hipsolver.hipsolverSgebrd_bufferSize
   rocm.bindings.hipsolver.hipsolverDgebrd_bufferSize
   rocm.bindings.hipsolver.hipsolverCgebrd_bufferSize
   rocm.bindings.hipsolver.hipsolverZgebrd_bufferSize
   rocm.bindings.hipsolver.hipsolverSgebrd
   rocm.bindings.hipsolver.hipsolverDgebrd
   rocm.bindings.hipsolver.hipsolverCgebrd
   rocm.bindings.hipsolver.hipsolverZgebrd
   rocm.bindings.hipsolver.hipsolverSSgels_bufferSize
   rocm.bindings.hipsolver.hipsolverDDgels_bufferSize
   rocm.bindings.hipsolver.hipsolverCCgels_bufferSize
   rocm.bindings.hipsolver.hipsolverZZgels_bufferSize
   rocm.bindings.hipsolver.hipsolverSSgels
   rocm.bindings.hipsolver.hipsolverDDgels
   rocm.bindings.hipsolver.hipsolverCCgels
   rocm.bindings.hipsolver.hipsolverZZgels
   rocm.bindings.hipsolver.hipsolverSgeqrf_bufferSize
   rocm.bindings.hipsolver.hipsolverDgeqrf_bufferSize
   rocm.bindings.hipsolver.hipsolverCgeqrf_bufferSize
   rocm.bindings.hipsolver.hipsolverZgeqrf_bufferSize
   rocm.bindings.hipsolver.hipsolverSgeqrf
   rocm.bindings.hipsolver.hipsolverDgeqrf
   rocm.bindings.hipsolver.hipsolverCgeqrf
   rocm.bindings.hipsolver.hipsolverZgeqrf
   rocm.bindings.hipsolver.hipsolverSSgesv_bufferSize
   rocm.bindings.hipsolver.hipsolverDDgesv_bufferSize
   rocm.bindings.hipsolver.hipsolverCCgesv_bufferSize
   rocm.bindings.hipsolver.hipsolverZZgesv_bufferSize
   rocm.bindings.hipsolver.hipsolverSSgesv
   rocm.bindings.hipsolver.hipsolverDDgesv
   rocm.bindings.hipsolver.hipsolverCCgesv
   rocm.bindings.hipsolver.hipsolverZZgesv
   rocm.bindings.hipsolver.hipsolverSgesvd_bufferSize
   rocm.bindings.hipsolver.hipsolverDgesvd_bufferSize
   rocm.bindings.hipsolver.hipsolverCgesvd_bufferSize
   rocm.bindings.hipsolver.hipsolverZgesvd_bufferSize
   rocm.bindings.hipsolver.hipsolverSgesvd
   rocm.bindings.hipsolver.hipsolverDgesvd
   rocm.bindings.hipsolver.hipsolverCgesvd
   rocm.bindings.hipsolver.hipsolverZgesvd
   rocm.bindings.hipsolver.hipsolverSgesvdj_bufferSize
   rocm.bindings.hipsolver.hipsolverDgesvdj_bufferSize
   rocm.bindings.hipsolver.hipsolverCgesvdj_bufferSize
   rocm.bindings.hipsolver.hipsolverZgesvdj_bufferSize
   rocm.bindings.hipsolver.hipsolverSgesvdj
   rocm.bindings.hipsolver.hipsolverDgesvdj
   rocm.bindings.hipsolver.hipsolverCgesvdj
   rocm.bindings.hipsolver.hipsolverZgesvdj
   rocm.bindings.hipsolver.hipsolverSgesvdjBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverDgesvdjBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverCgesvdjBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverZgesvdjBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverSgesvdjBatched
   rocm.bindings.hipsolver.hipsolverDgesvdjBatched
   rocm.bindings.hipsolver.hipsolverCgesvdjBatched
   rocm.bindings.hipsolver.hipsolverZgesvdjBatched
   rocm.bindings.hipsolver.hipsolverSgetrf_bufferSize
   rocm.bindings.hipsolver.hipsolverDgetrf_bufferSize
   rocm.bindings.hipsolver.hipsolverCgetrf_bufferSize
   rocm.bindings.hipsolver.hipsolverZgetrf_bufferSize
   rocm.bindings.hipsolver.hipsolverSgetrf
   rocm.bindings.hipsolver.hipsolverDgetrf
   rocm.bindings.hipsolver.hipsolverCgetrf
   rocm.bindings.hipsolver.hipsolverZgetrf
   rocm.bindings.hipsolver.hipsolverSgetrs_bufferSize
   rocm.bindings.hipsolver.hipsolverDgetrs_bufferSize
   rocm.bindings.hipsolver.hipsolverCgetrs_bufferSize
   rocm.bindings.hipsolver.hipsolverZgetrs_bufferSize
   rocm.bindings.hipsolver.hipsolverSgetrs
   rocm.bindings.hipsolver.hipsolverDgetrs
   rocm.bindings.hipsolver.hipsolverCgetrs
   rocm.bindings.hipsolver.hipsolverZgetrs
   rocm.bindings.hipsolver.hipsolverSpotrf_bufferSize
   rocm.bindings.hipsolver.hipsolverDpotrf_bufferSize
   rocm.bindings.hipsolver.hipsolverCpotrf_bufferSize
   rocm.bindings.hipsolver.hipsolverZpotrf_bufferSize
   rocm.bindings.hipsolver.hipsolverSpotrf
   rocm.bindings.hipsolver.hipsolverDpotrf
   rocm.bindings.hipsolver.hipsolverCpotrf
   rocm.bindings.hipsolver.hipsolverZpotrf
   rocm.bindings.hipsolver.hipsolverSpotrfBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverDpotrfBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverCpotrfBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverZpotrfBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverSpotrfBatched
   rocm.bindings.hipsolver.hipsolverDpotrfBatched
   rocm.bindings.hipsolver.hipsolverCpotrfBatched
   rocm.bindings.hipsolver.hipsolverZpotrfBatched
   rocm.bindings.hipsolver.hipsolverSpotri_bufferSize
   rocm.bindings.hipsolver.hipsolverDpotri_bufferSize
   rocm.bindings.hipsolver.hipsolverCpotri_bufferSize
   rocm.bindings.hipsolver.hipsolverZpotri_bufferSize
   rocm.bindings.hipsolver.hipsolverSpotri
   rocm.bindings.hipsolver.hipsolverDpotri
   rocm.bindings.hipsolver.hipsolverCpotri
   rocm.bindings.hipsolver.hipsolverZpotri
   rocm.bindings.hipsolver.hipsolverSpotrs_bufferSize
   rocm.bindings.hipsolver.hipsolverDpotrs_bufferSize
   rocm.bindings.hipsolver.hipsolverCpotrs_bufferSize
   rocm.bindings.hipsolver.hipsolverZpotrs_bufferSize
   rocm.bindings.hipsolver.hipsolverSpotrs
   rocm.bindings.hipsolver.hipsolverDpotrs
   rocm.bindings.hipsolver.hipsolverCpotrs
   rocm.bindings.hipsolver.hipsolverZpotrs
   rocm.bindings.hipsolver.hipsolverSpotrsBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverDpotrsBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverCpotrsBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverZpotrsBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverSpotrsBatched
   rocm.bindings.hipsolver.hipsolverDpotrsBatched
   rocm.bindings.hipsolver.hipsolverCpotrsBatched
   rocm.bindings.hipsolver.hipsolverZpotrsBatched
   rocm.bindings.hipsolver.hipsolverSsyevd_bufferSize
   rocm.bindings.hipsolver.hipsolverDsyevd_bufferSize
   rocm.bindings.hipsolver.hipsolverCheevd_bufferSize
   rocm.bindings.hipsolver.hipsolverZheevd_bufferSize
   rocm.bindings.hipsolver.hipsolverSsyevd
   rocm.bindings.hipsolver.hipsolverDsyevd
   rocm.bindings.hipsolver.hipsolverCheevd
   rocm.bindings.hipsolver.hipsolverZheevd
   rocm.bindings.hipsolver.hipsolverSsyevdx_bufferSize
   rocm.bindings.hipsolver.hipsolverDsyevdx_bufferSize
   rocm.bindings.hipsolver.hipsolverCheevdx_bufferSize
   rocm.bindings.hipsolver.hipsolverZheevdx_bufferSize
   rocm.bindings.hipsolver.hipsolverSsyevdx
   rocm.bindings.hipsolver.hipsolverDsyevdx
   rocm.bindings.hipsolver.hipsolverCheevdx
   rocm.bindings.hipsolver.hipsolverZheevdx
   rocm.bindings.hipsolver.hipsolverSsyevj_bufferSize
   rocm.bindings.hipsolver.hipsolverDsyevj_bufferSize
   rocm.bindings.hipsolver.hipsolverCheevj_bufferSize
   rocm.bindings.hipsolver.hipsolverZheevj_bufferSize
   rocm.bindings.hipsolver.hipsolverSsyevj
   rocm.bindings.hipsolver.hipsolverDsyevj
   rocm.bindings.hipsolver.hipsolverCheevj
   rocm.bindings.hipsolver.hipsolverZheevj
   rocm.bindings.hipsolver.hipsolverSsyevjBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverDsyevjBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverCheevjBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverZheevjBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverSsyevjBatched
   rocm.bindings.hipsolver.hipsolverDsyevjBatched
   rocm.bindings.hipsolver.hipsolverCheevjBatched
   rocm.bindings.hipsolver.hipsolverZheevjBatched
   rocm.bindings.hipsolver.hipsolverSsygvd_bufferSize
   rocm.bindings.hipsolver.hipsolverDsygvd_bufferSize
   rocm.bindings.hipsolver.hipsolverChegvd_bufferSize
   rocm.bindings.hipsolver.hipsolverZhegvd_bufferSize
   rocm.bindings.hipsolver.hipsolverSsygvd
   rocm.bindings.hipsolver.hipsolverDsygvd
   rocm.bindings.hipsolver.hipsolverChegvd
   rocm.bindings.hipsolver.hipsolverZhegvd
   rocm.bindings.hipsolver.hipsolverSsygvdx_bufferSize
   rocm.bindings.hipsolver.hipsolverDsygvdx_bufferSize
   rocm.bindings.hipsolver.hipsolverChegvdx_bufferSize
   rocm.bindings.hipsolver.hipsolverZhegvdx_bufferSize
   rocm.bindings.hipsolver.hipsolverSsygvdx
   rocm.bindings.hipsolver.hipsolverDsygvdx
   rocm.bindings.hipsolver.hipsolverChegvdx
   rocm.bindings.hipsolver.hipsolverZhegvdx
   rocm.bindings.hipsolver.hipsolverSsygvj_bufferSize
   rocm.bindings.hipsolver.hipsolverDsygvj_bufferSize
   rocm.bindings.hipsolver.hipsolverChegvj_bufferSize
   rocm.bindings.hipsolver.hipsolverZhegvj_bufferSize
   rocm.bindings.hipsolver.hipsolverSsygvj
   rocm.bindings.hipsolver.hipsolverDsygvj
   rocm.bindings.hipsolver.hipsolverChegvj
   rocm.bindings.hipsolver.hipsolverZhegvj
   rocm.bindings.hipsolver.hipsolverSsytrd_bufferSize
   rocm.bindings.hipsolver.hipsolverDsytrd_bufferSize
   rocm.bindings.hipsolver.hipsolverChetrd_bufferSize
   rocm.bindings.hipsolver.hipsolverZhetrd_bufferSize
   rocm.bindings.hipsolver.hipsolverSsytrd
   rocm.bindings.hipsolver.hipsolverDsytrd
   rocm.bindings.hipsolver.hipsolverChetrd
   rocm.bindings.hipsolver.hipsolverZhetrd
   rocm.bindings.hipsolver.hipsolverSsytrf_bufferSize
   rocm.bindings.hipsolver.hipsolverDsytrf_bufferSize
   rocm.bindings.hipsolver.hipsolverCsytrf_bufferSize
   rocm.bindings.hipsolver.hipsolverZsytrf_bufferSize
   rocm.bindings.hipsolver.hipsolverSsytrf
   rocm.bindings.hipsolver.hipsolverDsytrf
   rocm.bindings.hipsolver.hipsolverCsytrf
   rocm.bindings.hipsolver.hipsolverZsytrf
   rocm.bindings.hipsolver.hipsolverDnCreate
   rocm.bindings.hipsolver.hipsolverDnDestroy
   rocm.bindings.hipsolver.hipsolverDnSetStream
   rocm.bindings.hipsolver.hipsolverDnGetStream
   rocm.bindings.hipsolver.hipsolverDnSetDeterministicMode
   rocm.bindings.hipsolver.hipsolverDnGetDeterministicMode
   rocm.bindings.hipsolver.hipsolverDnCreateGesvdjInfo
   rocm.bindings.hipsolver.hipsolverDnDestroyGesvdjInfo
   rocm.bindings.hipsolver.hipsolverDnXgesvdjSetMaxSweeps
   rocm.bindings.hipsolver.hipsolverDnXgesvdjSetSortEig
   rocm.bindings.hipsolver.hipsolverDnXgesvdjSetTolerance
   rocm.bindings.hipsolver.hipsolverDnXgesvdjGetResidual
   rocm.bindings.hipsolver.hipsolverDnXgesvdjGetSweeps
   rocm.bindings.hipsolver.hipsolverDnCreateSyevjInfo
   rocm.bindings.hipsolver.hipsolverDnDestroySyevjInfo
   rocm.bindings.hipsolver.hipsolverDnXsyevjSetMaxSweeps
   rocm.bindings.hipsolver.hipsolverDnXsyevjSetSortEig
   rocm.bindings.hipsolver.hipsolverDnXsyevjSetTolerance
   rocm.bindings.hipsolver.hipsolverDnXsyevjGetResidual
   rocm.bindings.hipsolver.hipsolverDnXsyevjGetSweeps
   rocm.bindings.hipsolver.hipsolverDnSorgbr_bufferSize
   rocm.bindings.hipsolver.hipsolverDnDorgbr_bufferSize
   rocm.bindings.hipsolver.hipsolverDnCungbr_bufferSize
   rocm.bindings.hipsolver.hipsolverDnZungbr_bufferSize
   rocm.bindings.hipsolver.hipsolverDnSorgbr
   rocm.bindings.hipsolver.hipsolverDnDorgbr
   rocm.bindings.hipsolver.hipsolverDnCungbr
   rocm.bindings.hipsolver.hipsolverDnZungbr
   rocm.bindings.hipsolver.hipsolverDnSorgqr_bufferSize
   rocm.bindings.hipsolver.hipsolverDnDorgqr_bufferSize
   rocm.bindings.hipsolver.hipsolverDnCungqr_bufferSize
   rocm.bindings.hipsolver.hipsolverDnZungqr_bufferSize
   rocm.bindings.hipsolver.hipsolverDnSorgqr
   rocm.bindings.hipsolver.hipsolverDnDorgqr
   rocm.bindings.hipsolver.hipsolverDnCungqr
   rocm.bindings.hipsolver.hipsolverDnZungqr
   rocm.bindings.hipsolver.hipsolverDnSorgtr_bufferSize
   rocm.bindings.hipsolver.hipsolverDnDorgtr_bufferSize
   rocm.bindings.hipsolver.hipsolverDnCungtr_bufferSize
   rocm.bindings.hipsolver.hipsolverDnZungtr_bufferSize
   rocm.bindings.hipsolver.hipsolverDnSorgtr
   rocm.bindings.hipsolver.hipsolverDnDorgtr
   rocm.bindings.hipsolver.hipsolverDnCungtr
   rocm.bindings.hipsolver.hipsolverDnZungtr
   rocm.bindings.hipsolver.hipsolverDnSormqr_bufferSize
   rocm.bindings.hipsolver.hipsolverDnDormqr_bufferSize
   rocm.bindings.hipsolver.hipsolverDnCunmqr_bufferSize
   rocm.bindings.hipsolver.hipsolverDnZunmqr_bufferSize
   rocm.bindings.hipsolver.hipsolverDnSormqr
   rocm.bindings.hipsolver.hipsolverDnDormqr
   rocm.bindings.hipsolver.hipsolverDnCunmqr
   rocm.bindings.hipsolver.hipsolverDnZunmqr
   rocm.bindings.hipsolver.hipsolverDnSormtr_bufferSize
   rocm.bindings.hipsolver.hipsolverDnDormtr_bufferSize
   rocm.bindings.hipsolver.hipsolverDnCunmtr_bufferSize
   rocm.bindings.hipsolver.hipsolverDnZunmtr_bufferSize
   rocm.bindings.hipsolver.hipsolverDnSormtr
   rocm.bindings.hipsolver.hipsolverDnDormtr
   rocm.bindings.hipsolver.hipsolverDnCunmtr
   rocm.bindings.hipsolver.hipsolverDnZunmtr
   rocm.bindings.hipsolver.hipsolverDnSgebrd_bufferSize
   rocm.bindings.hipsolver.hipsolverDnDgebrd_bufferSize
   rocm.bindings.hipsolver.hipsolverDnCgebrd_bufferSize
   rocm.bindings.hipsolver.hipsolverDnZgebrd_bufferSize
   rocm.bindings.hipsolver.hipsolverDnSgebrd
   rocm.bindings.hipsolver.hipsolverDnDgebrd
   rocm.bindings.hipsolver.hipsolverDnCgebrd
   rocm.bindings.hipsolver.hipsolverDnZgebrd
   rocm.bindings.hipsolver.hipsolverDnSSgels_bufferSize
   rocm.bindings.hipsolver.hipsolverDnDDgels_bufferSize
   rocm.bindings.hipsolver.hipsolverDnCCgels_bufferSize
   rocm.bindings.hipsolver.hipsolverDnZZgels_bufferSize
   rocm.bindings.hipsolver.hipsolverDnSSgels
   rocm.bindings.hipsolver.hipsolverDnDDgels
   rocm.bindings.hipsolver.hipsolverDnCCgels
   rocm.bindings.hipsolver.hipsolverDnZZgels
   rocm.bindings.hipsolver.hipsolverDnSgeqrf_bufferSize
   rocm.bindings.hipsolver.hipsolverDnDgeqrf_bufferSize
   rocm.bindings.hipsolver.hipsolverDnCgeqrf_bufferSize
   rocm.bindings.hipsolver.hipsolverDnZgeqrf_bufferSize
   rocm.bindings.hipsolver.hipsolverDnSgeqrf
   rocm.bindings.hipsolver.hipsolverDnDgeqrf
   rocm.bindings.hipsolver.hipsolverDnCgeqrf
   rocm.bindings.hipsolver.hipsolverDnZgeqrf
   rocm.bindings.hipsolver.hipsolverDnSSgesv_bufferSize
   rocm.bindings.hipsolver.hipsolverDnDDgesv_bufferSize
   rocm.bindings.hipsolver.hipsolverDnCCgesv_bufferSize
   rocm.bindings.hipsolver.hipsolverDnZZgesv_bufferSize
   rocm.bindings.hipsolver.hipsolverDnSSgesv
   rocm.bindings.hipsolver.hipsolverDnDDgesv
   rocm.bindings.hipsolver.hipsolverDnCCgesv
   rocm.bindings.hipsolver.hipsolverDnZZgesv
   rocm.bindings.hipsolver.hipsolverDnSgesvd_bufferSize
   rocm.bindings.hipsolver.hipsolverDnDgesvd_bufferSize
   rocm.bindings.hipsolver.hipsolverDnCgesvd_bufferSize
   rocm.bindings.hipsolver.hipsolverDnZgesvd_bufferSize
   rocm.bindings.hipsolver.hipsolverDnSgesvd
   rocm.bindings.hipsolver.hipsolverDnDgesvd
   rocm.bindings.hipsolver.hipsolverDnCgesvd
   rocm.bindings.hipsolver.hipsolverDnZgesvd
   rocm.bindings.hipsolver.hipsolverDnSgesvdj_bufferSize
   rocm.bindings.hipsolver.hipsolverDnDgesvdj_bufferSize
   rocm.bindings.hipsolver.hipsolverDnCgesvdj_bufferSize
   rocm.bindings.hipsolver.hipsolverDnZgesvdj_bufferSize
   rocm.bindings.hipsolver.hipsolverDnSgesvdj
   rocm.bindings.hipsolver.hipsolverDnDgesvdj
   rocm.bindings.hipsolver.hipsolverDnCgesvdj
   rocm.bindings.hipsolver.hipsolverDnZgesvdj
   rocm.bindings.hipsolver.hipsolverDnSgesvdjBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverDnDgesvdjBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverDnCgesvdjBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverDnZgesvdjBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverDnSgesvdjBatched
   rocm.bindings.hipsolver.hipsolverDnDgesvdjBatched
   rocm.bindings.hipsolver.hipsolverDnCgesvdjBatched
   rocm.bindings.hipsolver.hipsolverDnZgesvdjBatched
   rocm.bindings.hipsolver.hipsolverDnSgesvdaStridedBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverDnDgesvdaStridedBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverDnCgesvdaStridedBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverDnZgesvdaStridedBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverDnSgesvdaStridedBatched
   rocm.bindings.hipsolver.hipsolverDnDgesvdaStridedBatched
   rocm.bindings.hipsolver.hipsolverDnCgesvdaStridedBatched
   rocm.bindings.hipsolver.hipsolverDnZgesvdaStridedBatched
   rocm.bindings.hipsolver.hipsolverDnSgetrf_bufferSize
   rocm.bindings.hipsolver.hipsolverDnDgetrf_bufferSize
   rocm.bindings.hipsolver.hipsolverDnCgetrf_bufferSize
   rocm.bindings.hipsolver.hipsolverDnZgetrf_bufferSize
   rocm.bindings.hipsolver.hipsolverDnSgetrf
   rocm.bindings.hipsolver.hipsolverDnDgetrf
   rocm.bindings.hipsolver.hipsolverDnCgetrf
   rocm.bindings.hipsolver.hipsolverDnZgetrf
   rocm.bindings.hipsolver.hipsolverDnSgetrs
   rocm.bindings.hipsolver.hipsolverDnDgetrs
   rocm.bindings.hipsolver.hipsolverDnCgetrs
   rocm.bindings.hipsolver.hipsolverDnZgetrs
   rocm.bindings.hipsolver.hipsolverDnSpotrf_bufferSize
   rocm.bindings.hipsolver.hipsolverDnDpotrf_bufferSize
   rocm.bindings.hipsolver.hipsolverDnCpotrf_bufferSize
   rocm.bindings.hipsolver.hipsolverDnZpotrf_bufferSize
   rocm.bindings.hipsolver.hipsolverDnSpotrf
   rocm.bindings.hipsolver.hipsolverDnDpotrf
   rocm.bindings.hipsolver.hipsolverDnCpotrf
   rocm.bindings.hipsolver.hipsolverDnZpotrf
   rocm.bindings.hipsolver.hipsolverDnSpotrfBatched
   rocm.bindings.hipsolver.hipsolverDnDpotrfBatched
   rocm.bindings.hipsolver.hipsolverDnCpotrfBatched
   rocm.bindings.hipsolver.hipsolverDnZpotrfBatched
   rocm.bindings.hipsolver.hipsolverDnSpotri_bufferSize
   rocm.bindings.hipsolver.hipsolverDnDpotri_bufferSize
   rocm.bindings.hipsolver.hipsolverDnCpotri_bufferSize
   rocm.bindings.hipsolver.hipsolverDnZpotri_bufferSize
   rocm.bindings.hipsolver.hipsolverDnSpotri
   rocm.bindings.hipsolver.hipsolverDnDpotri
   rocm.bindings.hipsolver.hipsolverDnCpotri
   rocm.bindings.hipsolver.hipsolverDnZpotri
   rocm.bindings.hipsolver.hipsolverDnSpotrs
   rocm.bindings.hipsolver.hipsolverDnDpotrs
   rocm.bindings.hipsolver.hipsolverDnCpotrs
   rocm.bindings.hipsolver.hipsolverDnZpotrs
   rocm.bindings.hipsolver.hipsolverDnSpotrsBatched
   rocm.bindings.hipsolver.hipsolverDnDpotrsBatched
   rocm.bindings.hipsolver.hipsolverDnCpotrsBatched
   rocm.bindings.hipsolver.hipsolverDnZpotrsBatched
   rocm.bindings.hipsolver.hipsolverDnSsyevd_bufferSize
   rocm.bindings.hipsolver.hipsolverDnDsyevd_bufferSize
   rocm.bindings.hipsolver.hipsolverDnCheevd_bufferSize
   rocm.bindings.hipsolver.hipsolverDnZheevd_bufferSize
   rocm.bindings.hipsolver.hipsolverDnSsyevd
   rocm.bindings.hipsolver.hipsolverDnDsyevd
   rocm.bindings.hipsolver.hipsolverDnCheevd
   rocm.bindings.hipsolver.hipsolverDnZheevd
   rocm.bindings.hipsolver.hipsolverDnSsyevdx_bufferSize
   rocm.bindings.hipsolver.hipsolverDnDsyevdx_bufferSize
   rocm.bindings.hipsolver.hipsolverDnCheevdx_bufferSize
   rocm.bindings.hipsolver.hipsolverDnZheevdx_bufferSize
   rocm.bindings.hipsolver.hipsolverDnSsyevdx
   rocm.bindings.hipsolver.hipsolverDnDsyevdx
   rocm.bindings.hipsolver.hipsolverDnCheevdx
   rocm.bindings.hipsolver.hipsolverDnZheevdx
   rocm.bindings.hipsolver.hipsolverDnSsyevj_bufferSize
   rocm.bindings.hipsolver.hipsolverDnDsyevj_bufferSize
   rocm.bindings.hipsolver.hipsolverDnCheevj_bufferSize
   rocm.bindings.hipsolver.hipsolverDnZheevj_bufferSize
   rocm.bindings.hipsolver.hipsolverDnSsyevj
   rocm.bindings.hipsolver.hipsolverDnDsyevj
   rocm.bindings.hipsolver.hipsolverDnCheevj
   rocm.bindings.hipsolver.hipsolverDnZheevj
   rocm.bindings.hipsolver.hipsolverDnSsyevjBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverDnDsyevjBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverDnCheevjBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverDnZheevjBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverDnSsyevjBatched
   rocm.bindings.hipsolver.hipsolverDnDsyevjBatched
   rocm.bindings.hipsolver.hipsolverDnCheevjBatched
   rocm.bindings.hipsolver.hipsolverDnZheevjBatched
   rocm.bindings.hipsolver.hipsolverDnSsygvd_bufferSize
   rocm.bindings.hipsolver.hipsolverDnDsygvd_bufferSize
   rocm.bindings.hipsolver.hipsolverDnChegvd_bufferSize
   rocm.bindings.hipsolver.hipsolverDnZhegvd_bufferSize
   rocm.bindings.hipsolver.hipsolverDnSsygvd
   rocm.bindings.hipsolver.hipsolverDnDsygvd
   rocm.bindings.hipsolver.hipsolverDnChegvd
   rocm.bindings.hipsolver.hipsolverDnZhegvd
   rocm.bindings.hipsolver.hipsolverDnSsygvdx_bufferSize
   rocm.bindings.hipsolver.hipsolverDnDsygvdx_bufferSize
   rocm.bindings.hipsolver.hipsolverDnChegvdx_bufferSize
   rocm.bindings.hipsolver.hipsolverDnZhegvdx_bufferSize
   rocm.bindings.hipsolver.hipsolverDnSsygvdx
   rocm.bindings.hipsolver.hipsolverDnDsygvdx
   rocm.bindings.hipsolver.hipsolverDnChegvdx
   rocm.bindings.hipsolver.hipsolverDnZhegvdx
   rocm.bindings.hipsolver.hipsolverDnSsygvj_bufferSize
   rocm.bindings.hipsolver.hipsolverDnDsygvj_bufferSize
   rocm.bindings.hipsolver.hipsolverDnChegvj_bufferSize
   rocm.bindings.hipsolver.hipsolverDnZhegvj_bufferSize
   rocm.bindings.hipsolver.hipsolverDnSsygvj
   rocm.bindings.hipsolver.hipsolverDnDsygvj
   rocm.bindings.hipsolver.hipsolverDnChegvj
   rocm.bindings.hipsolver.hipsolverDnZhegvj
   rocm.bindings.hipsolver.hipsolverDnSsytrd_bufferSize
   rocm.bindings.hipsolver.hipsolverDnDsytrd_bufferSize
   rocm.bindings.hipsolver.hipsolverDnChetrd_bufferSize
   rocm.bindings.hipsolver.hipsolverDnZhetrd_bufferSize
   rocm.bindings.hipsolver.hipsolverDnSsytrd
   rocm.bindings.hipsolver.hipsolverDnDsytrd
   rocm.bindings.hipsolver.hipsolverDnChetrd
   rocm.bindings.hipsolver.hipsolverDnZhetrd
   rocm.bindings.hipsolver.hipsolverDnSsytrf_bufferSize
   rocm.bindings.hipsolver.hipsolverDnDsytrf_bufferSize
   rocm.bindings.hipsolver.hipsolverDnCsytrf_bufferSize
   rocm.bindings.hipsolver.hipsolverDnZsytrf_bufferSize
   rocm.bindings.hipsolver.hipsolverDnSsytrf
   rocm.bindings.hipsolver.hipsolverDnDsytrf
   rocm.bindings.hipsolver.hipsolverDnCsytrf
   rocm.bindings.hipsolver.hipsolverDnZsytrf
   rocm.bindings.hipsolver.hipsolverDnCreateParams
   rocm.bindings.hipsolver.hipsolverDnDestroyParams
   rocm.bindings.hipsolver.hipsolverDnSetAdvOptions
   rocm.bindings.hipsolver.hipsolverDnXgeev_bufferSize
   rocm.bindings.hipsolver.hipsolverDnXgeev
   rocm.bindings.hipsolver.hipsolverDnXgeqrf_bufferSize
   rocm.bindings.hipsolver.hipsolverDnXgeqrf
   rocm.bindings.hipsolver.hipsolverDnXgetrf_bufferSize
   rocm.bindings.hipsolver.hipsolverDnXgetrf
   rocm.bindings.hipsolver.hipsolverDnXgetrs
   rocm.bindings.hipsolver.hipsolverDnXpotrf_bufferSize
   rocm.bindings.hipsolver.hipsolverDnXpotrf
   rocm.bindings.hipsolver.hipsolverDnXpotrs
   rocm.bindings.hipsolver.hipsolverDnXsyevd_bufferSize
   rocm.bindings.hipsolver.hipsolverDnXsyevd
   rocm.bindings.hipsolver.hipsolverDnXsyevBatched_bufferSize
   rocm.bindings.hipsolver.hipsolverDnXsyevBatched
   rocm.bindings.hipsolver.hipsolverDnXsytrs_bufferSize
   rocm.bindings.hipsolver.hipsolverDnXsytrs
   rocm.bindings.hipsolver.hipsolverRfCreate
   rocm.bindings.hipsolver.hipsolverRfDestroy
   rocm.bindings.hipsolver.hipsolverRfSetupDevice
   rocm.bindings.hipsolver.hipsolverRfSetupHost
   rocm.bindings.hipsolver.hipsolverRfAccessBundledFactorsDevice
   rocm.bindings.hipsolver.hipsolverRfAnalyze
   rocm.bindings.hipsolver.hipsolverRfExtractBundledFactorsHost
   rocm.bindings.hipsolver.hipsolverRfExtractSplitFactorsHost
   rocm.bindings.hipsolver.hipsolverRfGet_Algs
   rocm.bindings.hipsolver.hipsolverRfGetMatrixFormat
   rocm.bindings.hipsolver.hipsolverRfGetNumericBoostReport
   rocm.bindings.hipsolver.hipsolverRfGetNumericProperties
   rocm.bindings.hipsolver.hipsolverRfGetResetValuesFastMode
   rocm.bindings.hipsolver.hipsolverRfRefactor
   rocm.bindings.hipsolver.hipsolverRfResetValues
   rocm.bindings.hipsolver.hipsolverRfSetAlgs
   rocm.bindings.hipsolver.hipsolverRfSetMatrixFormat
   rocm.bindings.hipsolver.hipsolverRfSetNumericProperties
   rocm.bindings.hipsolver.hipsolverRfSetResetValuesFastMode
   rocm.bindings.hipsolver.hipsolverRfSolve
   rocm.bindings.hipsolver.hipsolverRfBatchSetupHost
   rocm.bindings.hipsolver.hipsolverRfBatchAnalyze
   rocm.bindings.hipsolver.hipsolverRfBatchRefactor
   rocm.bindings.hipsolver.hipsolverRfBatchResetValues
   rocm.bindings.hipsolver.hipsolverRfBatchSolve
   rocm.bindings.hipsolver.hipsolverRfBatchZeroPivot
   rocm.bindings.hipsolver.hipsolverSpCreate
   rocm.bindings.hipsolver.hipsolverSpDestroy
   rocm.bindings.hipsolver.hipsolverSpSetStream
   rocm.bindings.hipsolver.hipsolverSpScsrlsvchol
   rocm.bindings.hipsolver.hipsolverSpDcsrlsvchol
   rocm.bindings.hipsolver.hipsolverSpScsrlsvcholHost
   rocm.bindings.hipsolver.hipsolverSpDcsrlsvcholHost
   rocm.bindings.hipsolver.hipsolverSpScsrlsvqr
   rocm.bindings.hipsolver.hipsolverSpDcsrlsvqr
   rocm.bindings.hipsolver.hipsolverSpCcsrlsvqr
   rocm.bindings.hipsolver.hipsolverSpZcsrlsvqr


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:data:: hipsolverVersionMajor
   :type:  Any

.. py:data:: hipsolverVersionMinor
   :type:  Any

.. py:data:: hipsolverVersionPatch
   :type:  Any

.. py:class:: hipsolverStatus_t

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: HIPSOLVER_STATUS_SUCCESS
      :type:  int


   .. py:attribute:: HIPSOLVER_STATUS_NOT_INITIALIZED
      :type:  int


   .. py:attribute:: HIPSOLVER_STATUS_ALLOC_FAILED
      :type:  int


   .. py:attribute:: HIPSOLVER_STATUS_INVALID_VALUE
      :type:  int


   .. py:attribute:: HIPSOLVER_STATUS_MAPPING_ERROR
      :type:  int


   .. py:attribute:: HIPSOLVER_STATUS_EXECUTION_FAILED
      :type:  int


   .. py:attribute:: HIPSOLVER_STATUS_INTERNAL_ERROR
      :type:  int


   .. py:attribute:: HIPSOLVER_STATUS_NOT_SUPPORTED
      :type:  int


   .. py:attribute:: HIPSOLVER_STATUS_ARCH_MISMATCH
      :type:  int


   .. py:attribute:: HIPSOLVER_STATUS_HANDLE_IS_NULLPTR
      :type:  int


   .. py:attribute:: HIPSOLVER_STATUS_INVALID_ENUM
      :type:  int


   .. py:attribute:: HIPSOLVER_STATUS_UNKNOWN
      :type:  int


   .. py:attribute:: HIPSOLVER_STATUS_ZERO_PIVOT
      :type:  int


   .. py:attribute:: HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED
      :type:  int


.. py:class:: hipsolverEigMode_t

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: HIPSOLVER_EIG_MODE_NOVECTOR
      :type:  int


   .. py:attribute:: HIPSOLVER_EIG_MODE_VECTOR
      :type:  int


.. py:class:: hipsolverEigType_t

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: HIPSOLVER_EIG_TYPE_1
      :type:  int


   .. py:attribute:: HIPSOLVER_EIG_TYPE_2
      :type:  int


   .. py:attribute:: HIPSOLVER_EIG_TYPE_3
      :type:  int


.. py:class:: hipsolverEigRange_t

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: HIPSOLVER_EIG_RANGE_ALL
      :type:  int


   .. py:attribute:: HIPSOLVER_EIG_RANGE_V
      :type:  int


   .. py:attribute:: HIPSOLVER_EIG_RANGE_I
      :type:  int


.. py:class:: hipsolverDeterministicMode_t

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: HIPSOLVER_DETERMINISTIC_RESULTS
      :type:  int


   .. py:attribute:: HIPSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS
      :type:  int


.. py:data:: hipsolverOperation_t

.. py:data:: hipsolverFillMode_t

.. py:data:: hipsolverSideMode_t

.. py:function:: hipsolverCreate()

   (No short description, might be part of a group.)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCreate(hipsolverHandle_t * handle)


.. py:function:: hipsolverDestroy(handle)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDestroy(hipsolverHandle_t handle)


.. py:function:: hipsolverSetStream(handle, streamId)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       streamId (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSetStream(hipsolverHandle_t handle, hipStream_t streamId)


.. py:function:: hipsolverGetStream(handle)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * streamId (:py:obj:`~.ihipStream_t`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverGetStream(hipsolverHandle_t handle, hipStream_t * streamId)


.. py:function:: hipsolverSetDeterministicMode(handle, mode)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       mode (:py:obj:`~.hipsolverDeterministicMode_t`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSetDeterministicMode(hipsolverHandle_t handle, hipsolverDeterministicMode_t mode)


.. py:function:: hipsolverGetDeterministicMode(handle, mode)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       mode (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverGetDeterministicMode(hipsolverHandle_t handle, hipsolverDeterministicMode_t * mode)


.. py:function:: hipsolverCreateGesvdjInfo()

   // gesvdj params

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCreateGesvdjInfo(hipsolverGesvdjInfo_t * info)


.. py:function:: hipsolverDestroyGesvdjInfo(info)

   (No short description, might be part of a group.)

   Args:
       info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDestroyGesvdjInfo(hipsolverGesvdjInfo_t info)


.. py:function:: hipsolverXgesvdjSetMaxSweeps(info, max_sweeps)

   (No short description, might be part of a group.)

   Args:
       info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       max_sweeps (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverXgesvdjSetMaxSweeps(hipsolverGesvdjInfo_t info, int max_sweeps)


.. py:function:: hipsolverXgesvdjSetSortEig(info, sort_eig)

   (No short description, might be part of a group.)

   Args:
       info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       sort_eig (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverXgesvdjSetSortEig(hipsolverGesvdjInfo_t info, int sort_eig)


.. py:function:: hipsolverXgesvdjSetTolerance(info, tolerance)

   (No short description, might be part of a group.)

   Args:
       info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tolerance (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverXgesvdjSetTolerance(hipsolverGesvdjInfo_t info, double tolerance)


.. py:function:: hipsolverXgesvdjGetResidual(handle, info, residual)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       residual (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverXgesvdjGetResidual(hipsolverHandle_t handle, hipsolverGesvdjInfo_t info, double * residual)


.. py:function:: hipsolverXgesvdjGetSweeps(handle, info, executed_sweeps)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       executed_sweeps (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverXgesvdjGetSweeps(hipsolverHandle_t handle, hipsolverGesvdjInfo_t info, int * executed_sweeps)


.. py:function:: hipsolverCreateSyevjInfo()

   // syevj params

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCreateSyevjInfo(hipsolverSyevjInfo_t * info)


.. py:function:: hipsolverDestroySyevjInfo(info)

   (No short description, might be part of a group.)

   Args:
       info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDestroySyevjInfo(hipsolverSyevjInfo_t info)


.. py:function:: hipsolverXsyevjSetMaxSweeps(info, max_sweeps)

   (No short description, might be part of a group.)

   Args:
       info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       max_sweeps (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverXsyevjSetMaxSweeps(hipsolverSyevjInfo_t info, int max_sweeps)


.. py:function:: hipsolverXsyevjSetSortEig(info, sort_eig)

   (No short description, might be part of a group.)

   Args:
       info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       sort_eig (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverXsyevjSetSortEig(hipsolverSyevjInfo_t info, int sort_eig)


.. py:function:: hipsolverXsyevjSetTolerance(info, tolerance)

   (No short description, might be part of a group.)

   Args:
       info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tolerance (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverXsyevjSetTolerance(hipsolverSyevjInfo_t info, double tolerance)


.. py:function:: hipsolverXsyevjGetResidual(handle, info, residual)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       residual (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverXsyevjGetResidual(hipsolverHandle_t handle, hipsolverSyevjInfo_t info, double * residual)


.. py:function:: hipsolverXsyevjGetSweeps(handle, info, executed_sweeps)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       executed_sweeps (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverXsyevjGetSweeps(hipsolverHandle_t handle, hipsolverSyevjInfo_t info, int * executed_sweeps)


.. py:function:: hipsolverSorgbr_bufferSize(handle, side, m, n, k, A, lda, tau)

   // orgbr/ungbr

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSorgbr_bufferSize(hipsolverHandle_t handle, hipsolverSideMode_t side, int m, int n, int k, float * A, int lda, float * tau, int * lwork)


.. py:function:: hipsolverDorgbr_bufferSize(handle, side, m, n, k, A, lda, tau)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDorgbr_bufferSize(hipsolverHandle_t handle, hipsolverSideMode_t side, int m, int n, int k, double * A, int lda, double * tau, int * lwork)


.. py:function:: hipsolverCungbr_bufferSize(handle, side, m, n, k, A, lda, tau)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCungbr_bufferSize(hipsolverHandle_t handle, hipsolverSideMode_t side, int m, int n, int k, hipFloatComplex * A, int lda, hipFloatComplex * tau, int * lwork)


.. py:function:: hipsolverZungbr_bufferSize(handle, side, m, n, k, A, lda, tau)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZungbr_bufferSize(hipsolverHandle_t handle, hipsolverSideMode_t side, int m, int n, int k, hipDoubleComplex * A, int lda, hipDoubleComplex * tau, int * lwork)


.. py:function:: hipsolverSorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSorgbr(hipsolverHandle_t handle, hipsolverSideMode_t side, int m, int n, int k, float * A, int lda, float * tau, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDorgbr(hipsolverHandle_t handle, hipsolverSideMode_t side, int m, int n, int k, double * A, int lda, double * tau, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverCungbr(handle, side, m, n, k, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCungbr(hipsolverHandle_t handle, hipsolverSideMode_t side, int m, int n, int k, hipFloatComplex * A, int lda, hipFloatComplex * tau, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverZungbr(handle, side, m, n, k, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZungbr(hipsolverHandle_t handle, hipsolverSideMode_t side, int m, int n, int k, hipDoubleComplex * A, int lda, hipDoubleComplex * tau, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverSorgqr_bufferSize(handle, m, n, k, A, lda, tau)

   // orgqr/ungqr

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSorgqr_bufferSize(hipsolverHandle_t handle, int m, int n, int k, float * A, int lda, float * tau, int * lwork)


.. py:function:: hipsolverDorgqr_bufferSize(handle, m, n, k, A, lda, tau)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDorgqr_bufferSize(hipsolverHandle_t handle, int m, int n, int k, double * A, int lda, double * tau, int * lwork)


.. py:function:: hipsolverCungqr_bufferSize(handle, m, n, k, A, lda, tau)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCungqr_bufferSize(hipsolverHandle_t handle, int m, int n, int k, hipFloatComplex * A, int lda, hipFloatComplex * tau, int * lwork)


.. py:function:: hipsolverZungqr_bufferSize(handle, m, n, k, A, lda, tau)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZungqr_bufferSize(hipsolverHandle_t handle, int m, int n, int k, hipDoubleComplex * A, int lda, hipDoubleComplex * tau, int * lwork)


.. py:function:: hipsolverSorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSorgqr(hipsolverHandle_t handle, int m, int n, int k, float * A, int lda, float * tau, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDorgqr(hipsolverHandle_t handle, int m, int n, int k, double * A, int lda, double * tau, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverCungqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCungqr(hipsolverHandle_t handle, int m, int n, int k, hipFloatComplex * A, int lda, hipFloatComplex * tau, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverZungqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZungqr(hipsolverHandle_t handle, int m, int n, int k, hipDoubleComplex * A, int lda, hipDoubleComplex * tau, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverSorgtr_bufferSize(handle, uplo, n, A, lda, tau)

   // orgtr/ungtr

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSorgtr_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, float * A, int lda, float * tau, int * lwork)


.. py:function:: hipsolverDorgtr_bufferSize(handle, uplo, n, A, lda, tau)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDorgtr_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double * A, int lda, double * tau, int * lwork)


.. py:function:: hipsolverCungtr_bufferSize(handle, uplo, n, A, lda, tau)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCungtr_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, hipFloatComplex * A, int lda, hipFloatComplex * tau, int * lwork)


.. py:function:: hipsolverZungtr_bufferSize(handle, uplo, n, A, lda, tau)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZungtr_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, hipDoubleComplex * A, int lda, hipDoubleComplex * tau, int * lwork)


.. py:function:: hipsolverSorgtr(handle, uplo, n, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSorgtr(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, float * A, int lda, float * tau, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDorgtr(handle, uplo, n, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDorgtr(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double * A, int lda, double * tau, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverCungtr(handle, uplo, n, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCungtr(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, hipFloatComplex * A, int lda, hipFloatComplex * tau, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverZungtr(handle, uplo, n, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZungtr(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, hipDoubleComplex * A, int lda, hipDoubleComplex * tau, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc)

   // ormqr/unmqr

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSormqr_bufferSize(hipsolverHandle_t handle, hipsolverSideMode_t side, hipsolverOperation_t trans, int m, int n, int k, float * A, int lda, float * tau, float * C, int ldc, int * lwork)


.. py:function:: hipsolverDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDormqr_bufferSize(hipsolverHandle_t handle, hipsolverSideMode_t side, hipsolverOperation_t trans, int m, int n, int k, double * A, int lda, double * tau, double * C, int ldc, int * lwork)


.. py:function:: hipsolverCunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCunmqr_bufferSize(hipsolverHandle_t handle, hipsolverSideMode_t side, hipsolverOperation_t trans, int m, int n, int k, hipFloatComplex * A, int lda, hipFloatComplex * tau, hipFloatComplex * C, int ldc, int * lwork)


.. py:function:: hipsolverZunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZunmqr_bufferSize(hipsolverHandle_t handle, hipsolverSideMode_t side, hipsolverOperation_t trans, int m, int n, int k, hipDoubleComplex * A, int lda, hipDoubleComplex * tau, hipDoubleComplex * C, int ldc, int * lwork)


.. py:function:: hipsolverSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSormqr(hipsolverHandle_t handle, hipsolverSideMode_t side, hipsolverOperation_t trans, int m, int n, int k, float * A, int lda, float * tau, float * C, int ldc, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDormqr(hipsolverHandle_t handle, hipsolverSideMode_t side, hipsolverOperation_t trans, int m, int n, int k, double * A, int lda, double * tau, double * C, int ldc, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverCunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCunmqr(hipsolverHandle_t handle, hipsolverSideMode_t side, hipsolverOperation_t trans, int m, int n, int k, hipFloatComplex * A, int lda, hipFloatComplex * tau, hipFloatComplex * C, int ldc, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverZunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZunmqr(hipsolverHandle_t handle, hipsolverSideMode_t side, hipsolverOperation_t trans, int m, int n, int k, hipDoubleComplex * A, int lda, hipDoubleComplex * tau, hipDoubleComplex * C, int ldc, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverSormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc)

   // ormtr/unmtr

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSormtr_bufferSize(hipsolverHandle_t handle, hipsolverSideMode_t side, hipsolverFillMode_t uplo, hipsolverOperation_t trans, int m, int n, float * A, int lda, float * tau, float * C, int ldc, int * lwork)


.. py:function:: hipsolverDormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDormtr_bufferSize(hipsolverHandle_t handle, hipsolverSideMode_t side, hipsolverFillMode_t uplo, hipsolverOperation_t trans, int m, int n, double * A, int lda, double * tau, double * C, int ldc, int * lwork)


.. py:function:: hipsolverCunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCunmtr_bufferSize(hipsolverHandle_t handle, hipsolverSideMode_t side, hipsolverFillMode_t uplo, hipsolverOperation_t trans, int m, int n, hipFloatComplex * A, int lda, hipFloatComplex * tau, hipFloatComplex * C, int ldc, int * lwork)


.. py:function:: hipsolverZunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZunmtr_bufferSize(hipsolverHandle_t handle, hipsolverSideMode_t side, hipsolverFillMode_t uplo, hipsolverOperation_t trans, int m, int n, hipDoubleComplex * A, int lda, hipDoubleComplex * tau, hipDoubleComplex * C, int ldc, int * lwork)


.. py:function:: hipsolverSormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSormtr(hipsolverHandle_t handle, hipsolverSideMode_t side, hipsolverFillMode_t uplo, hipsolverOperation_t trans, int m, int n, float * A, int lda, float * tau, float * C, int ldc, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDormtr(hipsolverHandle_t handle, hipsolverSideMode_t side, hipsolverFillMode_t uplo, hipsolverOperation_t trans, int m, int n, double * A, int lda, double * tau, double * C, int ldc, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverCunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCunmtr(hipsolverHandle_t handle, hipsolverSideMode_t side, hipsolverFillMode_t uplo, hipsolverOperation_t trans, int m, int n, hipFloatComplex * A, int lda, hipFloatComplex * tau, hipFloatComplex * C, int ldc, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverZunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZunmtr(hipsolverHandle_t handle, hipsolverSideMode_t side, hipsolverFillMode_t uplo, hipsolverOperation_t trans, int m, int n, hipDoubleComplex * A, int lda, hipDoubleComplex * tau, hipDoubleComplex * C, int ldc, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverSgebrd_bufferSize(handle, m, n)

   // gebrd

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSgebrd_bufferSize(hipsolverHandle_t handle, int m, int n, int * lwork)


.. py:function:: hipsolverDgebrd_bufferSize(handle, m, n)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDgebrd_bufferSize(hipsolverHandle_t handle, int m, int n, int * lwork)


.. py:function:: hipsolverCgebrd_bufferSize(handle, m, n)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCgebrd_bufferSize(hipsolverHandle_t handle, int m, int n, int * lwork)


.. py:function:: hipsolverZgebrd_bufferSize(handle, m, n)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZgebrd_bufferSize(hipsolverHandle_t handle, int m, int n, int * lwork)


.. py:function:: hipsolverSgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       E (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tauq (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       taup (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSgebrd(hipsolverHandle_t handle, int m, int n, float * A, int lda, float * D, float * E, float * tauq, float * taup, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       E (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tauq (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       taup (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDgebrd(hipsolverHandle_t handle, int m, int n, double * A, int lda, double * D, double * E, double * tauq, double * taup, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverCgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       E (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tauq (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       taup (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCgebrd(hipsolverHandle_t handle, int m, int n, hipFloatComplex * A, int lda, float * D, float * E, hipFloatComplex * tauq, hipFloatComplex * taup, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverZgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       E (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tauq (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       taup (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZgebrd(hipsolverHandle_t handle, int m, int n, hipDoubleComplex * A, int lda, double * D, double * E, hipDoubleComplex * tauq, hipDoubleComplex * taup, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverSSgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx)

   // gels

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSSgels_bufferSize(hipsolverHandle_t handle, int m, int n, int nrhs, float * A, int lda, float * B, int ldb, float * X, int ldx, size_t * lwork)


.. py:function:: hipsolverDDgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDDgels_bufferSize(hipsolverHandle_t handle, int m, int n, int nrhs, double * A, int lda, double * B, int ldb, double * X, int ldx, size_t * lwork)


.. py:function:: hipsolverCCgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCCgels_bufferSize(hipsolverHandle_t handle, int m, int n, int nrhs, hipFloatComplex * A, int lda, hipFloatComplex * B, int ldb, hipFloatComplex * X, int ldx, size_t * lwork)


.. py:function:: hipsolverZZgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZZgels_bufferSize(hipsolverHandle_t handle, int m, int n, int nrhs, hipDoubleComplex * A, int lda, hipDoubleComplex * B, int ldb, hipDoubleComplex * X, int ldx, size_t * lwork)


.. py:function:: hipsolverSSgels(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       niters (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSSgels(hipsolverHandle_t handle, int m, int n, int nrhs, float * A, int lda, float * B, int ldb, float * X, int ldx, void * work, size_t lwork, int * niters, int * devInfo)


.. py:function:: hipsolverDDgels(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       niters (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDDgels(hipsolverHandle_t handle, int m, int n, int nrhs, double * A, int lda, double * B, int ldb, double * X, int ldx, void * work, size_t lwork, int * niters, int * devInfo)


.. py:function:: hipsolverCCgels(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       niters (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCCgels(hipsolverHandle_t handle, int m, int n, int nrhs, hipFloatComplex * A, int lda, hipFloatComplex * B, int ldb, hipFloatComplex * X, int ldx, void * work, size_t lwork, int * niters, int * devInfo)


.. py:function:: hipsolverZZgels(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       niters (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZZgels(hipsolverHandle_t handle, int m, int n, int nrhs, hipDoubleComplex * A, int lda, hipDoubleComplex * B, int ldb, hipDoubleComplex * X, int ldx, void * work, size_t lwork, int * niters, int * devInfo)


.. py:function:: hipsolverSgeqrf_bufferSize(handle, m, n, A, lda)

   // geqrf

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSgeqrf_bufferSize(hipsolverHandle_t handle, int m, int n, float * A, int lda, int * lwork)


.. py:function:: hipsolverDgeqrf_bufferSize(handle, m, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDgeqrf_bufferSize(hipsolverHandle_t handle, int m, int n, double * A, int lda, int * lwork)


.. py:function:: hipsolverCgeqrf_bufferSize(handle, m, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCgeqrf_bufferSize(hipsolverHandle_t handle, int m, int n, hipFloatComplex * A, int lda, int * lwork)


.. py:function:: hipsolverZgeqrf_bufferSize(handle, m, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZgeqrf_bufferSize(hipsolverHandle_t handle, int m, int n, hipDoubleComplex * A, int lda, int * lwork)


.. py:function:: hipsolverSgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSgeqrf(hipsolverHandle_t handle, int m, int n, float * A, int lda, float * tau, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDgeqrf(hipsolverHandle_t handle, int m, int n, double * A, int lda, double * tau, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverCgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCgeqrf(hipsolverHandle_t handle, int m, int n, hipFloatComplex * A, int lda, hipFloatComplex * tau, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverZgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZgeqrf(hipsolverHandle_t handle, int m, int n, hipDoubleComplex * A, int lda, hipDoubleComplex * tau, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverSSgesv_bufferSize(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx)

   // gesv

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSSgesv_bufferSize(hipsolverHandle_t handle, int n, int nrhs, float * A, int lda, int * devIpiv, float * B, int ldb, float * X, int ldx, size_t * lwork)


.. py:function:: hipsolverDDgesv_bufferSize(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDDgesv_bufferSize(hipsolverHandle_t handle, int n, int nrhs, double * A, int lda, int * devIpiv, double * B, int ldb, double * X, int ldx, size_t * lwork)


.. py:function:: hipsolverCCgesv_bufferSize(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCCgesv_bufferSize(hipsolverHandle_t handle, int n, int nrhs, hipFloatComplex * A, int lda, int * devIpiv, hipFloatComplex * B, int ldb, hipFloatComplex * X, int ldx, size_t * lwork)


.. py:function:: hipsolverZZgesv_bufferSize(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZZgesv_bufferSize(hipsolverHandle_t handle, int n, int nrhs, hipDoubleComplex * A, int lda, int * devIpiv, hipDoubleComplex * B, int ldb, hipDoubleComplex * X, int ldx, size_t * lwork)


.. py:function:: hipsolverSSgesv(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work, lwork, niters, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       niters (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSSgesv(hipsolverHandle_t handle, int n, int nrhs, float * A, int lda, int * devIpiv, float * B, int ldb, float * X, int ldx, void * work, size_t lwork, int * niters, int * devInfo)


.. py:function:: hipsolverDDgesv(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work, lwork, niters, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       niters (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDDgesv(hipsolverHandle_t handle, int n, int nrhs, double * A, int lda, int * devIpiv, double * B, int ldb, double * X, int ldx, void * work, size_t lwork, int * niters, int * devInfo)


.. py:function:: hipsolverCCgesv(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work, lwork, niters, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       niters (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCCgesv(hipsolverHandle_t handle, int n, int nrhs, hipFloatComplex * A, int lda, int * devIpiv, hipFloatComplex * B, int ldb, hipFloatComplex * X, int ldx, void * work, size_t lwork, int * niters, int * devInfo)


.. py:function:: hipsolverZZgesv(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work, lwork, niters, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       niters (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZZgesv(hipsolverHandle_t handle, int n, int nrhs, hipDoubleComplex * A, int lda, int * devIpiv, hipDoubleComplex * B, int ldb, hipDoubleComplex * X, int ldx, void * work, size_t lwork, int * niters, int * devInfo)


.. py:function:: hipsolverSgesvd_bufferSize(handle, jobu, jobv, m, n)

   // gesvd

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobu (:py:obj:`~.b`/:py:obj:`~.y`/:py:obj:`~.t`/:py:obj:`~.e`/:py:obj:`~.s`):
           (undocumented)

       jobv (:py:obj:`~.b`/:py:obj:`~.y`/:py:obj:`~.t`/:py:obj:`~.e`/:py:obj:`~.s`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSgesvd_bufferSize(hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, int * lwork)


.. py:function:: hipsolverDgesvd_bufferSize(handle, jobu, jobv, m, n)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobu (:py:obj:`~.b`/:py:obj:`~.y`/:py:obj:`~.t`/:py:obj:`~.e`/:py:obj:`~.s`):
           (undocumented)

       jobv (:py:obj:`~.b`/:py:obj:`~.y`/:py:obj:`~.t`/:py:obj:`~.e`/:py:obj:`~.s`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDgesvd_bufferSize(hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, int * lwork)


.. py:function:: hipsolverCgesvd_bufferSize(handle, jobu, jobv, m, n)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobu (:py:obj:`~.b`/:py:obj:`~.y`/:py:obj:`~.t`/:py:obj:`~.e`/:py:obj:`~.s`):
           (undocumented)

       jobv (:py:obj:`~.b`/:py:obj:`~.y`/:py:obj:`~.t`/:py:obj:`~.e`/:py:obj:`~.s`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCgesvd_bufferSize(hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, int * lwork)


.. py:function:: hipsolverZgesvd_bufferSize(handle, jobu, jobv, m, n)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobu (:py:obj:`~.b`/:py:obj:`~.y`/:py:obj:`~.t`/:py:obj:`~.e`/:py:obj:`~.s`):
           (undocumented)

       jobv (:py:obj:`~.b`/:py:obj:`~.y`/:py:obj:`~.t`/:py:obj:`~.e`/:py:obj:`~.s`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZgesvd_bufferSize(hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, int * lwork)


.. py:function:: hipsolverSgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobu (:py:obj:`~.b`/:py:obj:`~.y`/:py:obj:`~.t`/:py:obj:`~.e`/:py:obj:`~.s`):
           (undocumented)

       jobv (:py:obj:`~.b`/:py:obj:`~.y`/:py:obj:`~.t`/:py:obj:`~.e`/:py:obj:`~.s`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       rwork (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSgesvd(hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, float * A, int lda, float * S, float * U, int ldu, float * V, int ldv, float * work, int lwork, float * rwork, int * devInfo)


.. py:function:: hipsolverDgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobu (:py:obj:`~.b`/:py:obj:`~.y`/:py:obj:`~.t`/:py:obj:`~.e`/:py:obj:`~.s`):
           (undocumented)

       jobv (:py:obj:`~.b`/:py:obj:`~.y`/:py:obj:`~.t`/:py:obj:`~.e`/:py:obj:`~.s`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       rwork (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDgesvd(hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, double * A, int lda, double * S, double * U, int ldu, double * V, int ldv, double * work, int lwork, double * rwork, int * devInfo)


.. py:function:: hipsolverCgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobu (:py:obj:`~.b`/:py:obj:`~.y`/:py:obj:`~.t`/:py:obj:`~.e`/:py:obj:`~.s`):
           (undocumented)

       jobv (:py:obj:`~.b`/:py:obj:`~.y`/:py:obj:`~.t`/:py:obj:`~.e`/:py:obj:`~.s`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       rwork (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCgesvd(hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, hipFloatComplex * A, int lda, float * S, hipFloatComplex * U, int ldu, hipFloatComplex * V, int ldv, hipFloatComplex * work, int lwork, float * rwork, int * devInfo)


.. py:function:: hipsolverZgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobu (:py:obj:`~.b`/:py:obj:`~.y`/:py:obj:`~.t`/:py:obj:`~.e`/:py:obj:`~.s`):
           (undocumented)

       jobv (:py:obj:`~.b`/:py:obj:`~.y`/:py:obj:`~.t`/:py:obj:`~.e`/:py:obj:`~.s`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       rwork (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZgesvd(hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, hipDoubleComplex * A, int lda, double * S, hipDoubleComplex * U, int ldu, hipDoubleComplex * V, int ldv, hipDoubleComplex * work, int lwork, double * rwork, int * devInfo)


.. py:function:: hipsolverSgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, params)

   // gesvdj

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       econ (:py:obj:`~.int`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSgesvdj_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int econ, int m, int n, const float * A, int lda, const float * S, const float * U, int ldu, const float * V, int ldv, int * lwork, hipsolverGesvdjInfo_t params)


.. py:function:: hipsolverDgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       econ (:py:obj:`~.int`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDgesvdj_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int econ, int m, int n, const double * A, int lda, const double * S, const double * U, int ldu, const double * V, int ldv, int * lwork, hipsolverGesvdjInfo_t params)


.. py:function:: hipsolverCgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       econ (:py:obj:`~.int`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCgesvdj_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int econ, int m, int n, const hipFloatComplex * A, int lda, const float * S, const hipFloatComplex * U, int ldu, const hipFloatComplex * V, int ldv, int * lwork, hipsolverGesvdjInfo_t params)


.. py:function:: hipsolverZgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       econ (:py:obj:`~.int`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZgesvdj_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int econ, int m, int n, const hipDoubleComplex * A, int lda, const double * S, const hipDoubleComplex * U, int ldu, const hipDoubleComplex * V, int ldv, int * lwork, hipsolverGesvdjInfo_t params)


.. py:function:: hipsolverSgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, devInfo, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       econ (:py:obj:`~.int`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSgesvdj(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int econ, int m, int n, float * A, int lda, float * S, float * U, int ldu, float * V, int ldv, float * work, int lwork, int * devInfo, hipsolverGesvdjInfo_t params)


.. py:function:: hipsolverDgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, devInfo, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       econ (:py:obj:`~.int`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDgesvdj(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int econ, int m, int n, double * A, int lda, double * S, double * U, int ldu, double * V, int ldv, double * work, int lwork, int * devInfo, hipsolverGesvdjInfo_t params)


.. py:function:: hipsolverCgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, devInfo, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       econ (:py:obj:`~.int`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCgesvdj(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int econ, int m, int n, hipFloatComplex * A, int lda, float * S, hipFloatComplex * U, int ldu, hipFloatComplex * V, int ldv, hipFloatComplex * work, int lwork, int * devInfo, hipsolverGesvdjInfo_t params)


.. py:function:: hipsolverZgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, devInfo, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       econ (:py:obj:`~.int`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZgesvdj(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int econ, int m, int n, hipDoubleComplex * A, int lda, double * S, hipDoubleComplex * U, int ldu, hipDoubleComplex * V, int ldv, hipDoubleComplex * work, int lwork, int * devInfo, hipsolverGesvdjInfo_t params)


.. py:function:: hipsolverSgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, params, batch_count)

   // gesvdj_batched

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSgesvdjBatched_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int m, int n, const float * A, int lda, const float * S, const float * U, int ldu, const float * V, int ldv, int * lwork, hipsolverGesvdjInfo_t params, int batch_count)


.. py:function:: hipsolverDgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDgesvdjBatched_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int m, int n, const double * A, int lda, const double * S, const double * U, int ldu, const double * V, int ldv, int * lwork, hipsolverGesvdjInfo_t params, int batch_count)


.. py:function:: hipsolverCgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCgesvdjBatched_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int m, int n, const hipFloatComplex * A, int lda, const float * S, const hipFloatComplex * U, int ldu, const hipFloatComplex * V, int ldv, int * lwork, hipsolverGesvdjInfo_t params, int batch_count)


.. py:function:: hipsolverZgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZgesvdjBatched_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int m, int n, const hipDoubleComplex * A, int lda, const double * S, const hipDoubleComplex * U, int ldu, const hipDoubleComplex * V, int ldv, int * lwork, hipsolverGesvdjInfo_t params, int batch_count)


.. py:function:: hipsolverSgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, devInfo, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSgesvdjBatched(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int m, int n, float * A, int lda, float * S, float * U, int ldu, float * V, int ldv, float * work, int lwork, int * devInfo, hipsolverGesvdjInfo_t params, int batch_count)


.. py:function:: hipsolverDgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, devInfo, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDgesvdjBatched(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int m, int n, double * A, int lda, double * S, double * U, int ldu, double * V, int ldv, double * work, int lwork, int * devInfo, hipsolverGesvdjInfo_t params, int batch_count)


.. py:function:: hipsolverCgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, devInfo, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCgesvdjBatched(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int m, int n, hipFloatComplex * A, int lda, float * S, hipFloatComplex * U, int ldu, hipFloatComplex * V, int ldv, hipFloatComplex * work, int lwork, int * devInfo, hipsolverGesvdjInfo_t params, int batch_count)


.. py:function:: hipsolverZgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, devInfo, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZgesvdjBatched(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int m, int n, hipDoubleComplex * A, int lda, double * S, hipDoubleComplex * U, int ldu, hipDoubleComplex * V, int ldv, hipDoubleComplex * work, int lwork, int * devInfo, hipsolverGesvdjInfo_t params, int batch_count)


.. py:function:: hipsolverSgetrf_bufferSize(handle, m, n, A, lda)

   // getrf

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSgetrf_bufferSize(hipsolverHandle_t handle, int m, int n, float * A, int lda, int * lwork)


.. py:function:: hipsolverDgetrf_bufferSize(handle, m, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDgetrf_bufferSize(hipsolverHandle_t handle, int m, int n, double * A, int lda, int * lwork)


.. py:function:: hipsolverCgetrf_bufferSize(handle, m, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCgetrf_bufferSize(hipsolverHandle_t handle, int m, int n, hipFloatComplex * A, int lda, int * lwork)


.. py:function:: hipsolverZgetrf_bufferSize(handle, m, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZgetrf_bufferSize(hipsolverHandle_t handle, int m, int n, hipDoubleComplex * A, int lda, int * lwork)


.. py:function:: hipsolverSgetrf(handle, m, n, A, lda, work, lwork, devIpiv, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSgetrf(hipsolverHandle_t handle, int m, int n, float * A, int lda, float * work, int lwork, int * devIpiv, int * devInfo)


.. py:function:: hipsolverDgetrf(handle, m, n, A, lda, work, lwork, devIpiv, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDgetrf(hipsolverHandle_t handle, int m, int n, double * A, int lda, double * work, int lwork, int * devIpiv, int * devInfo)


.. py:function:: hipsolverCgetrf(handle, m, n, A, lda, work, lwork, devIpiv, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCgetrf(hipsolverHandle_t handle, int m, int n, hipFloatComplex * A, int lda, hipFloatComplex * work, int lwork, int * devIpiv, int * devInfo)


.. py:function:: hipsolverZgetrf(handle, m, n, A, lda, work, lwork, devIpiv, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZgetrf(hipsolverHandle_t handle, int m, int n, hipDoubleComplex * A, int lda, hipDoubleComplex * work, int lwork, int * devIpiv, int * devInfo)


.. py:function:: hipsolverSgetrs_bufferSize(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb)

   // getrs

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSgetrs_bufferSize(hipsolverHandle_t handle, hipsolverOperation_t trans, int n, int nrhs, float * A, int lda, int * devIpiv, float * B, int ldb, int * lwork)


.. py:function:: hipsolverDgetrs_bufferSize(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDgetrs_bufferSize(hipsolverHandle_t handle, hipsolverOperation_t trans, int n, int nrhs, double * A, int lda, int * devIpiv, double * B, int ldb, int * lwork)


.. py:function:: hipsolverCgetrs_bufferSize(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCgetrs_bufferSize(hipsolverHandle_t handle, hipsolverOperation_t trans, int n, int nrhs, hipFloatComplex * A, int lda, int * devIpiv, hipFloatComplex * B, int ldb, int * lwork)


.. py:function:: hipsolverZgetrs_bufferSize(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZgetrs_bufferSize(hipsolverHandle_t handle, hipsolverOperation_t trans, int n, int nrhs, hipDoubleComplex * A, int lda, int * devIpiv, hipDoubleComplex * B, int ldb, int * lwork)


.. py:function:: hipsolverSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSgetrs(hipsolverHandle_t handle, hipsolverOperation_t trans, int n, int nrhs, float * A, int lda, int * devIpiv, float * B, int ldb, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDgetrs(hipsolverHandle_t handle, hipsolverOperation_t trans, int n, int nrhs, double * A, int lda, int * devIpiv, double * B, int ldb, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverCgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCgetrs(hipsolverHandle_t handle, hipsolverOperation_t trans, int n, int nrhs, hipFloatComplex * A, int lda, int * devIpiv, hipFloatComplex * B, int ldb, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverZgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZgetrs(hipsolverHandle_t handle, hipsolverOperation_t trans, int n, int nrhs, hipDoubleComplex * A, int lda, int * devIpiv, hipDoubleComplex * B, int ldb, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverSpotrf_bufferSize(handle, uplo, n, A, lda)

   // potrf

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSpotrf_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, float * A, int lda, int * lwork)


.. py:function:: hipsolverDpotrf_bufferSize(handle, uplo, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDpotrf_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double * A, int lda, int * lwork)


.. py:function:: hipsolverCpotrf_bufferSize(handle, uplo, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCpotrf_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, hipFloatComplex * A, int lda, int * lwork)


.. py:function:: hipsolverZpotrf_bufferSize(handle, uplo, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZpotrf_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, hipDoubleComplex * A, int lda, int * lwork)


.. py:function:: hipsolverSpotrf(handle, uplo, n, A, lda, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSpotrf(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, float * A, int lda, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDpotrf(handle, uplo, n, A, lda, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDpotrf(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double * A, int lda, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverCpotrf(handle, uplo, n, A, lda, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCpotrf(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, hipFloatComplex * A, int lda, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverZpotrf(handle, uplo, n, A, lda, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZpotrf(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, hipDoubleComplex * A, int lda, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverSpotrfBatched_bufferSize(handle, uplo, n, A, lda, batch_count)

   // potrf_batched

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSpotrfBatched_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, float *[] A, int lda, int * lwork, int batch_count)


.. py:function:: hipsolverDpotrfBatched_bufferSize(handle, uplo, n, A, lda, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDpotrfBatched_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double *[] A, int lda, int * lwork, int batch_count)


.. py:function:: hipsolverCpotrfBatched_bufferSize(handle, uplo, n, A, lda, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCpotrfBatched_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, hipFloatComplex *[] A, int lda, int * lwork, int batch_count)


.. py:function:: hipsolverZpotrfBatched_bufferSize(handle, uplo, n, A, lda, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZpotrfBatched_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, hipDoubleComplex *[] A, int lda, int * lwork, int batch_count)


.. py:function:: hipsolverSpotrfBatched(handle, uplo, n, A, lda, work, lwork, devInfo, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSpotrfBatched(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, float *[] A, int lda, float * work, int lwork, int * devInfo, int batch_count)


.. py:function:: hipsolverDpotrfBatched(handle, uplo, n, A, lda, work, lwork, devInfo, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDpotrfBatched(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double *[] A, int lda, double * work, int lwork, int * devInfo, int batch_count)


.. py:function:: hipsolverCpotrfBatched(handle, uplo, n, A, lda, work, lwork, devInfo, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCpotrfBatched(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, hipFloatComplex *[] A, int lda, hipFloatComplex * work, int lwork, int * devInfo, int batch_count)


.. py:function:: hipsolverZpotrfBatched(handle, uplo, n, A, lda, work, lwork, devInfo, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZpotrfBatched(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, hipDoubleComplex *[] A, int lda, hipDoubleComplex * work, int lwork, int * devInfo, int batch_count)


.. py:function:: hipsolverSpotri_bufferSize(handle, uplo, n, A, lda)

   // potri

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSpotri_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, float * A, int lda, int * lwork)


.. py:function:: hipsolverDpotri_bufferSize(handle, uplo, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDpotri_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double * A, int lda, int * lwork)


.. py:function:: hipsolverCpotri_bufferSize(handle, uplo, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCpotri_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, hipFloatComplex * A, int lda, int * lwork)


.. py:function:: hipsolverZpotri_bufferSize(handle, uplo, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZpotri_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, hipDoubleComplex * A, int lda, int * lwork)


.. py:function:: hipsolverSpotri(handle, uplo, n, A, lda, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSpotri(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, float * A, int lda, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDpotri(handle, uplo, n, A, lda, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDpotri(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double * A, int lda, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverCpotri(handle, uplo, n, A, lda, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCpotri(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, hipFloatComplex * A, int lda, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverZpotri(handle, uplo, n, A, lda, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZpotri(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, hipDoubleComplex * A, int lda, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverSpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb)

   // potrs

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSpotrs_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, int nrhs, float * A, int lda, float * B, int ldb, int * lwork)


.. py:function:: hipsolverDpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDpotrs_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, int nrhs, double * A, int lda, double * B, int ldb, int * lwork)


.. py:function:: hipsolverCpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCpotrs_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, int nrhs, hipFloatComplex * A, int lda, hipFloatComplex * B, int ldb, int * lwork)


.. py:function:: hipsolverZpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZpotrs_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, int nrhs, hipDoubleComplex * A, int lda, hipDoubleComplex * B, int ldb, int * lwork)


.. py:function:: hipsolverSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSpotrs(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, int nrhs, float * A, int lda, float * B, int ldb, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDpotrs(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, int nrhs, double * A, int lda, double * B, int ldb, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverCpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCpotrs(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, int nrhs, hipFloatComplex * A, int lda, hipFloatComplex * B, int ldb, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverZpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZpotrs(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, int nrhs, hipDoubleComplex * A, int lda, hipDoubleComplex * B, int ldb, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverSpotrsBatched_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, batch_count)

   // potrs_batched

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSpotrsBatched_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, int nrhs, float *[] A, int lda, float *[] B, int ldb, int * lwork, int batch_count)


.. py:function:: hipsolverDpotrsBatched_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDpotrsBatched_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, int nrhs, double *[] A, int lda, double *[] B, int ldb, int * lwork, int batch_count)


.. py:function:: hipsolverCpotrsBatched_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCpotrsBatched_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, int nrhs, hipFloatComplex *[] A, int lda, hipFloatComplex *[] B, int ldb, int * lwork, int batch_count)


.. py:function:: hipsolverZpotrsBatched_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZpotrsBatched_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, int nrhs, hipDoubleComplex *[] A, int lda, hipDoubleComplex *[] B, int ldb, int * lwork, int batch_count)


.. py:function:: hipsolverSpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, devInfo, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSpotrsBatched(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, int nrhs, float *[] A, int lda, float *[] B, int ldb, float * work, int lwork, int * devInfo, int batch_count)


.. py:function:: hipsolverDpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, devInfo, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDpotrsBatched(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, int nrhs, double *[] A, int lda, double *[] B, int ldb, double * work, int lwork, int * devInfo, int batch_count)


.. py:function:: hipsolverCpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, devInfo, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCpotrsBatched(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, int nrhs, hipFloatComplex *[] A, int lda, hipFloatComplex *[] B, int ldb, hipFloatComplex * work, int lwork, int * devInfo, int batch_count)


.. py:function:: hipsolverZpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, devInfo, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZpotrsBatched(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, int nrhs, hipDoubleComplex *[] A, int lda, hipDoubleComplex *[] B, int ldb, hipDoubleComplex * work, int lwork, int * devInfo, int batch_count)


.. py:function:: hipsolverSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, D)

   // syevd/heevd

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSsyevd_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, float * A, int lda, float * D, int * lwork)


.. py:function:: hipsolverDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, D)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDsyevd_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, double * A, int lda, double * D, int * lwork)


.. py:function:: hipsolverCheevd_bufferSize(handle, jobz, uplo, n, A, lda, D)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCheevd_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, hipFloatComplex * A, int lda, float * D, int * lwork)


.. py:function:: hipsolverZheevd_bufferSize(handle, jobz, uplo, n, A, lda, D)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZheevd_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, hipDoubleComplex * A, int lda, double * D, int * lwork)


.. py:function:: hipsolverSsyevd(handle, jobz, uplo, n, A, lda, D, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSsyevd(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, float * A, int lda, float * D, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDsyevd(handle, jobz, uplo, n, A, lda, D, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDsyevd(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, double * A, int lda, double * D, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverCheevd(handle, jobz, uplo, n, A, lda, D, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCheevd(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, hipFloatComplex * A, int lda, float * D, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverZheevd(handle, jobz, uplo, n, A, lda, D, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZheevd(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, hipDoubleComplex * A, int lda, double * D, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverSsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W)

   // syevdx/heevdx

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSsyevdx_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const float * A, int lda, float vl, float vu, int il, int iu, int * nev, const float * W, int * lwork)


.. py:function:: hipsolverDsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDsyevdx_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const double * A, int lda, double vl, double vu, int il, int iu, int * nev, const double * W, int * lwork)


.. py:function:: hipsolverCheevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCheevdx_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const hipFloatComplex * A, int lda, float vl, float vu, int il, int iu, int * nev, const float * W, int * lwork)


.. py:function:: hipsolverZheevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZheevdx_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const hipDoubleComplex * A, int lda, double vl, double vu, int il, int iu, int * nev, const double * W, int * lwork)


.. py:function:: hipsolverSsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSsyevdx(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, float * A, int lda, float vl, float vu, int il, int iu, int * nev, float * W, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDsyevdx(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, double * A, int lda, double vl, double vu, int il, int iu, int * nev, double * W, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverCheevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCheevdx(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, hipFloatComplex * A, int lda, float vl, float vu, int il, int iu, int * nev, float * W, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverZheevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZheevdx(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, hipDoubleComplex * A, int lda, double vl, double vu, int il, int iu, int * nev, double * W, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverSsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, params)

   // syevj/heevj

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSsyevj_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, float * A, int lda, float * W, int * lwork, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverDsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDsyevj_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, double * A, int lda, double * W, int * lwork, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverCheevj_bufferSize(handle, jobz, uplo, n, A, lda, W, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCheevj_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, hipFloatComplex * A, int lda, float * W, int * lwork, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverZheevj_bufferSize(handle, jobz, uplo, n, A, lda, W, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZheevj_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, hipDoubleComplex * A, int lda, double * W, int * lwork, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverSsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSsyevj(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, float * A, int lda, float * W, float * work, int lwork, int * devInfo, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverDsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDsyevj(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, double * A, int lda, double * W, double * work, int lwork, int * devInfo, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverCheevj(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCheevj(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, hipFloatComplex * A, int lda, float * W, hipFloatComplex * work, int lwork, int * devInfo, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverZheevj(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZheevj(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, hipDoubleComplex * A, int lda, double * W, hipDoubleComplex * work, int lwork, int * devInfo, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverSsyevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, params, batch_count)

   // syevj_batched/heevj_batched

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSsyevjBatched_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, float * A, int lda, float * W, int * lwork, hipsolverSyevjInfo_t params, int batch_count)


.. py:function:: hipsolverDsyevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDsyevjBatched_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, double * A, int lda, double * W, int * lwork, hipsolverSyevjInfo_t params, int batch_count)


.. py:function:: hipsolverCheevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCheevjBatched_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, hipFloatComplex * A, int lda, float * W, int * lwork, hipsolverSyevjInfo_t params, int batch_count)


.. py:function:: hipsolverZheevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZheevjBatched_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, hipDoubleComplex * A, int lda, double * W, int * lwork, hipsolverSyevjInfo_t params, int batch_count)


.. py:function:: hipsolverSsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSsyevjBatched(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, float * A, int lda, float * W, float * work, int lwork, int * devInfo, hipsolverSyevjInfo_t params, int batch_count)


.. py:function:: hipsolverDsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDsyevjBatched(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, double * A, int lda, double * W, double * work, int lwork, int * devInfo, hipsolverSyevjInfo_t params, int batch_count)


.. py:function:: hipsolverCheevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCheevjBatched(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, hipFloatComplex * A, int lda, float * W, hipFloatComplex * work, int lwork, int * devInfo, hipsolverSyevjInfo_t params, int batch_count)


.. py:function:: hipsolverZheevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZheevjBatched(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, hipDoubleComplex * A, int lda, double * W, hipDoubleComplex * work, int lwork, int * devInfo, hipsolverSyevjInfo_t params, int batch_count)


.. py:function:: hipsolverSsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W)

   // sygvd/hegvd

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSsygvd_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, float * A, int lda, float * B, int ldb, float * W, int * lwork)


.. py:function:: hipsolverDsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDsygvd_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, double * A, int lda, double * B, int ldb, double * W, int * lwork)


.. py:function:: hipsolverChegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverChegvd_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, hipFloatComplex * A, int lda, hipFloatComplex * B, int ldb, float * W, int * lwork)


.. py:function:: hipsolverZhegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZhegvd_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, hipDoubleComplex * A, int lda, hipDoubleComplex * B, int ldb, double * W, int * lwork)


.. py:function:: hipsolverSsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSsygvd(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, float * A, int lda, float * B, int ldb, float * W, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDsygvd(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, double * A, int lda, double * B, int ldb, double * W, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverChegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverChegvd(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, hipFloatComplex * A, int lda, hipFloatComplex * B, int ldb, float * W, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverZhegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZhegvd(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, hipDoubleComplex * A, int lda, hipDoubleComplex * B, int ldb, double * W, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverSsygvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, nev, W)

   // sygvdx/hegvdx

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSsygvdx_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const float * A, int lda, const float * B, int ldb, float vl, float vu, int il, int iu, int * nev, const float * W, int * lwork)


.. py:function:: hipsolverDsygvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, nev, W)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDsygvdx_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const double * A, int lda, const double * B, int ldb, double vl, double vu, int il, int iu, int * nev, const double * W, int * lwork)


.. py:function:: hipsolverChegvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, nev, W)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverChegvdx_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const hipFloatComplex * A, int lda, const hipFloatComplex * B, int ldb, float vl, float vu, int il, int iu, int * nev, const float * W, int * lwork)


.. py:function:: hipsolverZhegvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, nev, W)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZhegvdx_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const hipDoubleComplex * A, int lda, const hipDoubleComplex * B, int ldb, double vl, double vu, int il, int iu, int * nev, const double * W, int * lwork)


.. py:function:: hipsolverSsygvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, nev, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSsygvdx(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, float * A, int lda, float * B, int ldb, float vl, float vu, int il, int iu, int * nev, float * W, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDsygvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, nev, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDsygvdx(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, double * A, int lda, double * B, int ldb, double vl, double vu, int il, int iu, int * nev, double * W, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverChegvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, nev, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverChegvdx(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, hipFloatComplex * A, int lda, hipFloatComplex * B, int ldb, float vl, float vu, int il, int iu, int * nev, float * W, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverZhegvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, nev, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZhegvdx(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, hipDoubleComplex * A, int lda, hipDoubleComplex * B, int ldb, double vl, double vu, int il, int iu, int * nev, double * W, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverSsygvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, params)

   // sygvj/hegvj

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSsygvj_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, float * A, int lda, float * B, int ldb, float * W, int * lwork, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverDsygvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDsygvj_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, double * A, int lda, double * B, int ldb, double * W, int * lwork, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverChegvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverChegvj_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, hipFloatComplex * A, int lda, hipFloatComplex * B, int ldb, float * W, int * lwork, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverZhegvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZhegvj_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, hipDoubleComplex * A, int lda, hipDoubleComplex * B, int ldb, double * W, int * lwork, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverSsygvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, devInfo, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSsygvj(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, float * A, int lda, float * B, int ldb, float * W, float * work, int lwork, int * devInfo, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverDsygvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, devInfo, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDsygvj(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, double * A, int lda, double * B, int ldb, double * W, double * work, int lwork, int * devInfo, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverChegvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, devInfo, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverChegvj(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, hipFloatComplex * A, int lda, hipFloatComplex * B, int ldb, float * W, hipFloatComplex * work, int lwork, int * devInfo, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverZhegvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, devInfo, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZhegvj(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, hipDoubleComplex * A, int lda, hipDoubleComplex * B, int ldb, double * W, hipDoubleComplex * work, int lwork, int * devInfo, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverSsytrd_bufferSize(handle, uplo, n, A, lda, D, E, tau)

   // sytrd/hetrd

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       E (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSsytrd_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, float * A, int lda, float * D, float * E, float * tau, int * lwork)


.. py:function:: hipsolverDsytrd_bufferSize(handle, uplo, n, A, lda, D, E, tau)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       E (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDsytrd_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double * A, int lda, double * D, double * E, double * tau, int * lwork)


.. py:function:: hipsolverChetrd_bufferSize(handle, uplo, n, A, lda, D, E, tau)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       E (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverChetrd_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, hipFloatComplex * A, int lda, float * D, float * E, hipFloatComplex * tau, int * lwork)


.. py:function:: hipsolverZhetrd_bufferSize(handle, uplo, n, A, lda, D, E, tau)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       E (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZhetrd_bufferSize(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, hipDoubleComplex * A, int lda, double * D, double * E, hipDoubleComplex * tau, int * lwork)


.. py:function:: hipsolverSsytrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       E (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSsytrd(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, float * A, int lda, float * D, float * E, float * tau, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDsytrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       E (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDsytrd(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double * A, int lda, double * D, double * E, double * tau, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverChetrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       E (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverChetrd(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, hipFloatComplex * A, int lda, float * D, float * E, hipFloatComplex * tau, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverZhetrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       E (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZhetrd(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, hipDoubleComplex * A, int lda, double * D, double * E, hipDoubleComplex * tau, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverSsytrf_bufferSize(handle, n, A, lda)

   // sytrf

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSsytrf_bufferSize(hipsolverHandle_t handle, int n, float * A, int lda, int * lwork)


.. py:function:: hipsolverDsytrf_bufferSize(handle, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDsytrf_bufferSize(hipsolverHandle_t handle, int n, double * A, int lda, int * lwork)


.. py:function:: hipsolverCsytrf_bufferSize(handle, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCsytrf_bufferSize(hipsolverHandle_t handle, int n, hipFloatComplex * A, int lda, int * lwork)


.. py:function:: hipsolverZsytrf_bufferSize(handle, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZsytrf_bufferSize(hipsolverHandle_t handle, int n, hipDoubleComplex * A, int lda, int * lwork)


.. py:function:: hipsolverSsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       ipiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSsytrf(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, float * A, int lda, int * ipiv, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       ipiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDsytrf(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double * A, int lda, int * ipiv, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverCsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       ipiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverCsytrf(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, hipFloatComplex * A, int lda, int * ipiv, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverZsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       ipiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverZsytrf(hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, hipDoubleComplex * A, int lda, int * ipiv, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnCreate()

   An alias for :py:obj:`~.hipsolverCreate`.
   ******************************************************************************

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCreate(hipsolverHandle_t * handle)


.. py:function:: hipsolverDnDestroy(handle)

   An alias for :py:obj:`~.hipsolverDestroy`.
   ******************************************************************************

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDestroy(hipsolverHandle_t handle)


.. py:function:: hipsolverDnSetStream(handle, streamId)

   An alias for :py:obj:`~.hipsolverSetStream`.
   ******************************************************************************

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       streamId (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSetStream(hipsolverHandle_t handle, hipStream_t streamId)


.. py:function:: hipsolverDnGetStream(handle)

   An alias for :py:obj:`~.hipsolverGetStream`.
   ******************************************************************************

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * streamId (:py:obj:`~.ihipStream_t`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnGetStream(hipsolverHandle_t handle, hipStream_t * streamId)


.. py:function:: hipsolverDnSetDeterministicMode(handle, mode)

   An alias for :py:obj:`~.hipsolverSetDeterministicMode`.
   ******************************************************************************

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       mode (:py:obj:`~.hipsolverDeterministicMode_t`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSetDeterministicMode(hipsolverHandle_t handle, hipsolverDeterministicMode_t mode)


.. py:function:: hipsolverDnGetDeterministicMode(handle, mode)

   An alias for :py:obj:`~.hipsolverGetDeterministicMode`.
   ******************************************************************************

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       mode (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnGetDeterministicMode(hipsolverHandle_t handle, hipsolverDeterministicMode_t * mode)


.. py:function:: hipsolverDnCreateGesvdjInfo()

   // gesvdj params

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCreateGesvdjInfo(hipsolverGesvdjInfo_t * info)


.. py:function:: hipsolverDnDestroyGesvdjInfo(info)

   (No short description, might be part of a group.)

   Args:
       info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDestroyGesvdjInfo(hipsolverGesvdjInfo_t info)


.. py:function:: hipsolverDnXgesvdjSetMaxSweeps(info, max_sweeps)

   (No short description, might be part of a group.)

   Args:
       info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       max_sweeps (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnXgesvdjSetMaxSweeps(hipsolverGesvdjInfo_t info, int max_sweeps)


.. py:function:: hipsolverDnXgesvdjSetSortEig(info, sort_eig)

   (No short description, might be part of a group.)

   Args:
       info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       sort_eig (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnXgesvdjSetSortEig(hipsolverGesvdjInfo_t info, int sort_eig)


.. py:function:: hipsolverDnXgesvdjSetTolerance(info, tolerance)

   (No short description, might be part of a group.)

   Args:
       info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tolerance (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnXgesvdjSetTolerance(hipsolverGesvdjInfo_t info, double tolerance)


.. py:function:: hipsolverDnXgesvdjGetResidual(handle, info, residual)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       residual (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnXgesvdjGetResidual(hipsolverDnHandle_t handle, hipsolverGesvdjInfo_t info, double * residual)


.. py:function:: hipsolverDnXgesvdjGetSweeps(handle, info, executed_sweeps)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       executed_sweeps (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnXgesvdjGetSweeps(hipsolverDnHandle_t handle, hipsolverGesvdjInfo_t info, int * executed_sweeps)


.. py:function:: hipsolverDnCreateSyevjInfo()

   // syevj params

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCreateSyevjInfo(hipsolverSyevjInfo_t * info)


.. py:function:: hipsolverDnDestroySyevjInfo(info)

   (No short description, might be part of a group.)

   Args:
       info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDestroySyevjInfo(hipsolverSyevjInfo_t info)


.. py:function:: hipsolverDnXsyevjSetMaxSweeps(info, max_sweeps)

   (No short description, might be part of a group.)

   Args:
       info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       max_sweeps (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnXsyevjSetMaxSweeps(hipsolverSyevjInfo_t info, int max_sweeps)


.. py:function:: hipsolverDnXsyevjSetSortEig(info, sort_eig)

   (No short description, might be part of a group.)

   Args:
       info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       sort_eig (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnXsyevjSetSortEig(hipsolverSyevjInfo_t info, int sort_eig)


.. py:function:: hipsolverDnXsyevjSetTolerance(info, tolerance)

   (No short description, might be part of a group.)

   Args:
       info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tolerance (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnXsyevjSetTolerance(hipsolverSyevjInfo_t info, double tolerance)


.. py:function:: hipsolverDnXsyevjGetResidual(handle, info, residual)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       residual (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnXsyevjGetResidual(hipsolverDnHandle_t handle, hipsolverSyevjInfo_t info, double * residual)


.. py:function:: hipsolverDnXsyevjGetSweeps(handle, info, executed_sweeps)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       info (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       executed_sweeps (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnXsyevjGetSweeps(hipsolverDnHandle_t handle, hipsolverSyevjInfo_t info, int * executed_sweeps)


.. py:function:: hipsolverDnSorgbr_bufferSize(handle, side, m, n, k, A, lda, tau)

   // orgbr/ungbr

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSorgbr_bufferSize(hipsolverHandle_t handle, hipblasSideMode_t side, int m, int n, int k, const float * A, int lda, const float * tau, int * lwork)


.. py:function:: hipsolverDnDorgbr_bufferSize(handle, side, m, n, k, A, lda, tau)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDorgbr_bufferSize(hipsolverHandle_t handle, hipblasSideMode_t side, int m, int n, int k, const double * A, int lda, const double * tau, int * lwork)


.. py:function:: hipsolverDnCungbr_bufferSize(handle, side, m, n, k, A, lda, tau)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCungbr_bufferSize(hipsolverHandle_t handle, hipblasSideMode_t side, int m, int n, int k, const hipFloatComplex * A, int lda, const hipFloatComplex * tau, int * lwork)


.. py:function:: hipsolverDnZungbr_bufferSize(handle, side, m, n, k, A, lda, tau)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZungbr_bufferSize(hipsolverHandle_t handle, hipblasSideMode_t side, int m, int n, int k, const hipDoubleComplex * A, int lda, const hipDoubleComplex * tau, int * lwork)


.. py:function:: hipsolverDnSorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSorgbr(hipsolverHandle_t handle, hipblasSideMode_t side, int m, int n, int k, float * A, int lda, const float * tau, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnDorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDorgbr(hipsolverHandle_t handle, hipblasSideMode_t side, int m, int n, int k, double * A, int lda, const double * tau, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnCungbr(handle, side, m, n, k, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCungbr(hipsolverHandle_t handle, hipblasSideMode_t side, int m, int n, int k, hipFloatComplex * A, int lda, const hipFloatComplex * tau, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnZungbr(handle, side, m, n, k, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZungbr(hipsolverHandle_t handle, hipblasSideMode_t side, int m, int n, int k, hipDoubleComplex * A, int lda, const hipDoubleComplex * tau, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau)

   // orgqr/ungqr

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSorgqr_bufferSize(hipsolverHandle_t handle, int m, int n, int k, const float * A, int lda, const float * tau, int * lwork)


.. py:function:: hipsolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDorgqr_bufferSize(hipsolverHandle_t handle, int m, int n, int k, const double * A, int lda, const double * tau, int * lwork)


.. py:function:: hipsolverDnCungqr_bufferSize(handle, m, n, k, A, lda, tau)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCungqr_bufferSize(hipsolverHandle_t handle, int m, int n, int k, const hipFloatComplex * A, int lda, const hipFloatComplex * tau, int * lwork)


.. py:function:: hipsolverDnZungqr_bufferSize(handle, m, n, k, A, lda, tau)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZungqr_bufferSize(hipsolverHandle_t handle, int m, int n, int k, const hipDoubleComplex * A, int lda, const hipDoubleComplex * tau, int * lwork)


.. py:function:: hipsolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSorgqr(hipsolverHandle_t handle, int m, int n, int k, float * A, int lda, const float * tau, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDorgqr(hipsolverHandle_t handle, int m, int n, int k, double * A, int lda, const double * tau, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnCungqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCungqr(hipsolverHandle_t handle, int m, int n, int k, hipFloatComplex * A, int lda, const hipFloatComplex * tau, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnZungqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZungqr(hipsolverHandle_t handle, int m, int n, int k, hipDoubleComplex * A, int lda, const hipDoubleComplex * tau, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnSorgtr_bufferSize(handle, uplo, n, A, lda, tau)

   // orgtr/ungtr

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSorgtr_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, const float * A, int lda, const float * tau, int * lwork)


.. py:function:: hipsolverDnDorgtr_bufferSize(handle, uplo, n, A, lda, tau)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDorgtr_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, const double * A, int lda, const double * tau, int * lwork)


.. py:function:: hipsolverDnCungtr_bufferSize(handle, uplo, n, A, lda, tau)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCungtr_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, const hipFloatComplex * A, int lda, const hipFloatComplex * tau, int * lwork)


.. py:function:: hipsolverDnZungtr_bufferSize(handle, uplo, n, A, lda, tau)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZungtr_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, const hipDoubleComplex * A, int lda, const hipDoubleComplex * tau, int * lwork)


.. py:function:: hipsolverDnSorgtr(handle, uplo, n, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSorgtr(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, float * A, int lda, const float * tau, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnDorgtr(handle, uplo, n, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDorgtr(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, double * A, int lda, const double * tau, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnCungtr(handle, uplo, n, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCungtr(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipFloatComplex * A, int lda, const hipFloatComplex * tau, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnZungtr(handle, uplo, n, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZungtr(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipDoubleComplex * A, int lda, const hipDoubleComplex * tau, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc)

   // ormqr/unmqr

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSormqr_bufferSize(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasOperation_t trans, int m, int n, int k, const float * A, int lda, const float * tau, const float * C, int ldc, int * lwork)


.. py:function:: hipsolverDnDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDormqr_bufferSize(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasOperation_t trans, int m, int n, int k, const double * A, int lda, const double * tau, const double * C, int ldc, int * lwork)


.. py:function:: hipsolverDnCunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCunmqr_bufferSize(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasOperation_t trans, int m, int n, int k, const hipFloatComplex * A, int lda, const hipFloatComplex * tau, const hipFloatComplex * C, int ldc, int * lwork)


.. py:function:: hipsolverDnZunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZunmqr_bufferSize(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasOperation_t trans, int m, int n, int k, const hipDoubleComplex * A, int lda, const hipDoubleComplex * tau, const hipDoubleComplex * C, int ldc, int * lwork)


.. py:function:: hipsolverDnSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSormqr(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasOperation_t trans, int m, int n, int k, const float * A, int lda, const float * tau, float * C, int ldc, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDormqr(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasOperation_t trans, int m, int n, int k, const double * A, int lda, const double * tau, double * C, int ldc, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnCunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCunmqr(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasOperation_t trans, int m, int n, int k, const hipFloatComplex * A, int lda, const hipFloatComplex * tau, hipFloatComplex * C, int ldc, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnZunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       k (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZunmqr(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasOperation_t trans, int m, int n, int k, const hipDoubleComplex * A, int lda, const hipDoubleComplex * tau, hipDoubleComplex * C, int ldc, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnSormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc)

   // ormtr/unmtr

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSormtr_bufferSize(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t trans, int m, int n, const float * A, int lda, const float * tau, const float * C, int ldc, int * lwork)


.. py:function:: hipsolverDnDormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDormtr_bufferSize(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t trans, int m, int n, const double * A, int lda, const double * tau, const double * C, int ldc, int * lwork)


.. py:function:: hipsolverDnCunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCunmtr_bufferSize(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t trans, int m, int n, const hipFloatComplex * A, int lda, const hipFloatComplex * tau, const hipFloatComplex * C, int ldc, int * lwork)


.. py:function:: hipsolverDnZunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZunmtr_bufferSize(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t trans, int m, int n, const hipDoubleComplex * A, int lda, const hipDoubleComplex * tau, const hipDoubleComplex * C, int ldc, int * lwork)


.. py:function:: hipsolverDnSormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSormtr(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t trans, int m, int n, float * A, int lda, float * tau, float * C, int ldc, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnDormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDormtr(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t trans, int m, int n, double * A, int lda, double * tau, double * C, int ldc, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnCunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCunmtr(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t trans, int m, int n, hipFloatComplex * A, int lda, hipFloatComplex * tau, hipFloatComplex * C, int ldc, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnZunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       side (:py:obj:`~.hipblasSideMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldc (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZunmtr(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t trans, int m, int n, hipDoubleComplex * A, int lda, hipDoubleComplex * tau, hipDoubleComplex * C, int ldc, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnSgebrd_bufferSize(handle, m, n)

   // gebrd

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSgebrd_bufferSize(hipsolverHandle_t handle, int m, int n, int * lwork)


.. py:function:: hipsolverDnDgebrd_bufferSize(handle, m, n)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDgebrd_bufferSize(hipsolverHandle_t handle, int m, int n, int * lwork)


.. py:function:: hipsolverDnCgebrd_bufferSize(handle, m, n)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCgebrd_bufferSize(hipsolverHandle_t handle, int m, int n, int * lwork)


.. py:function:: hipsolverDnZgebrd_bufferSize(handle, m, n)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZgebrd_bufferSize(hipsolverHandle_t handle, int m, int n, int * lwork)


.. py:function:: hipsolverDnSgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       E (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tauq (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       taup (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSgebrd(hipsolverHandle_t handle, int m, int n, float * A, int lda, float * D, float * E, float * tauq, float * taup, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnDgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       E (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tauq (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       taup (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDgebrd(hipsolverHandle_t handle, int m, int n, double * A, int lda, double * D, double * E, double * tauq, double * taup, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnCgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       E (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tauq (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       taup (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCgebrd(hipsolverHandle_t handle, int m, int n, hipFloatComplex * A, int lda, float * D, float * E, hipFloatComplex * tauq, hipFloatComplex * taup, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnZgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       E (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tauq (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       taup (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZgebrd(hipsolverHandle_t handle, int m, int n, hipDoubleComplex * A, int lda, double * D, double * E, hipDoubleComplex * tauq, hipDoubleComplex * taup, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnSSgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work)

   // gels

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSSgels_bufferSize(hipsolverHandle_t handle, int m, int n, int nrhs, float * A, int lda, float * B, int ldb, float * X, int ldx, void * work, size_t * lwork)


.. py:function:: hipsolverDnDDgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDDgels_bufferSize(hipsolverHandle_t handle, int m, int n, int nrhs, double * A, int lda, double * B, int ldb, double * X, int ldx, void * work, size_t * lwork)


.. py:function:: hipsolverDnCCgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCCgels_bufferSize(hipsolverHandle_t handle, int m, int n, int nrhs, hipFloatComplex * A, int lda, hipFloatComplex * B, int ldb, hipFloatComplex * X, int ldx, void * work, size_t * lwork)


.. py:function:: hipsolverDnZZgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZZgels_bufferSize(hipsolverHandle_t handle, int m, int n, int nrhs, hipDoubleComplex * A, int lda, hipDoubleComplex * B, int ldb, hipDoubleComplex * X, int ldx, void * work, size_t * lwork)


.. py:function:: hipsolverDnSSgels(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       niters (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSSgels(hipsolverHandle_t handle, int m, int n, int nrhs, float * A, int lda, float * B, int ldb, float * X, int ldx, void * work, size_t lwork, int * niters, int * devInfo)


.. py:function:: hipsolverDnDDgels(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       niters (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDDgels(hipsolverHandle_t handle, int m, int n, int nrhs, double * A, int lda, double * B, int ldb, double * X, int ldx, void * work, size_t lwork, int * niters, int * devInfo)


.. py:function:: hipsolverDnCCgels(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       niters (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCCgels(hipsolverHandle_t handle, int m, int n, int nrhs, hipFloatComplex * A, int lda, hipFloatComplex * B, int ldb, hipFloatComplex * X, int ldx, void * work, size_t lwork, int * niters, int * devInfo)


.. py:function:: hipsolverDnZZgels(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       niters (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZZgels(hipsolverHandle_t handle, int m, int n, int nrhs, hipDoubleComplex * A, int lda, hipDoubleComplex * B, int ldb, hipDoubleComplex * X, int ldx, void * work, size_t lwork, int * niters, int * devInfo)


.. py:function:: hipsolverDnSgeqrf_bufferSize(handle, m, n, A, lda)

   // geqrf

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSgeqrf_bufferSize(hipsolverHandle_t handle, int m, int n, float * A, int lda, int * lwork)


.. py:function:: hipsolverDnDgeqrf_bufferSize(handle, m, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDgeqrf_bufferSize(hipsolverHandle_t handle, int m, int n, double * A, int lda, int * lwork)


.. py:function:: hipsolverDnCgeqrf_bufferSize(handle, m, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCgeqrf_bufferSize(hipsolverHandle_t handle, int m, int n, hipFloatComplex * A, int lda, int * lwork)


.. py:function:: hipsolverDnZgeqrf_bufferSize(handle, m, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZgeqrf_bufferSize(hipsolverHandle_t handle, int m, int n, hipDoubleComplex * A, int lda, int * lwork)


.. py:function:: hipsolverDnSgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSgeqrf(hipsolverHandle_t handle, int m, int n, float * A, int lda, float * tau, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnDgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDgeqrf(hipsolverHandle_t handle, int m, int n, double * A, int lda, double * tau, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnCgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCgeqrf(hipsolverHandle_t handle, int m, int n, hipFloatComplex * A, int lda, hipFloatComplex * tau, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnZgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZgeqrf(hipsolverHandle_t handle, int m, int n, hipDoubleComplex * A, int lda, hipDoubleComplex * tau, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnSSgesv_bufferSize(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work)

   // gesv

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSSgesv_bufferSize(hipsolverHandle_t handle, int n, int nrhs, float * A, int lda, int * devIpiv, float * B, int ldb, float * X, int ldx, void * work, size_t * lwork)


.. py:function:: hipsolverDnDDgesv_bufferSize(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDDgesv_bufferSize(hipsolverHandle_t handle, int n, int nrhs, double * A, int lda, int * devIpiv, double * B, int ldb, double * X, int ldx, void * work, size_t * lwork)


.. py:function:: hipsolverDnCCgesv_bufferSize(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCCgesv_bufferSize(hipsolverHandle_t handle, int n, int nrhs, hipFloatComplex * A, int lda, int * devIpiv, hipFloatComplex * B, int ldb, hipFloatComplex * X, int ldx, void * work, size_t * lwork)


.. py:function:: hipsolverDnZZgesv_bufferSize(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZZgesv_bufferSize(hipsolverHandle_t handle, int n, int nrhs, hipDoubleComplex * A, int lda, int * devIpiv, hipDoubleComplex * B, int ldb, hipDoubleComplex * X, int ldx, void * work, size_t * lwork)


.. py:function:: hipsolverDnSSgesv(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work, lwork, niters, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       niters (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSSgesv(hipsolverHandle_t handle, int n, int nrhs, float * A, int lda, int * devIpiv, float * B, int ldb, float * X, int ldx, void * work, size_t lwork, int * niters, int * devInfo)


.. py:function:: hipsolverDnDDgesv(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work, lwork, niters, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       niters (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDDgesv(hipsolverHandle_t handle, int n, int nrhs, double * A, int lda, int * devIpiv, double * B, int ldb, double * X, int ldx, void * work, size_t lwork, int * niters, int * devInfo)


.. py:function:: hipsolverDnCCgesv(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work, lwork, niters, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       niters (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCCgesv(hipsolverHandle_t handle, int n, int nrhs, hipFloatComplex * A, int lda, int * devIpiv, hipFloatComplex * B, int ldb, hipFloatComplex * X, int ldx, void * work, size_t lwork, int * niters, int * devInfo)


.. py:function:: hipsolverDnZZgesv(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work, lwork, niters, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       X (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldx (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       niters (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZZgesv(hipsolverHandle_t handle, int n, int nrhs, hipDoubleComplex * A, int lda, int * devIpiv, hipDoubleComplex * B, int ldb, hipDoubleComplex * X, int ldx, void * work, size_t lwork, int * niters, int * devInfo)


.. py:function:: hipsolverDnSgesvd_bufferSize(handle, m, n)

   // gesvd

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSgesvd_bufferSize(hipsolverHandle_t handle, int m, int n, int * lwork)


.. py:function:: hipsolverDnDgesvd_bufferSize(handle, m, n)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDgesvd_bufferSize(hipsolverHandle_t handle, int m, int n, int * lwork)


.. py:function:: hipsolverDnCgesvd_bufferSize(handle, m, n)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCgesvd_bufferSize(hipsolverHandle_t handle, int m, int n, int * lwork)


.. py:function:: hipsolverDnZgesvd_bufferSize(handle, m, n)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZgesvd_bufferSize(hipsolverHandle_t handle, int m, int n, int * lwork)


.. py:function:: hipsolverDnSgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobu (:py:obj:`~.b`/:py:obj:`~.y`/:py:obj:`~.t`/:py:obj:`~.e`/:py:obj:`~.s`):
           (undocumented)

       jobv (:py:obj:`~.b`/:py:obj:`~.y`/:py:obj:`~.t`/:py:obj:`~.e`/:py:obj:`~.s`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       rwork (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSgesvd(hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, float * A, int lda, float * S, float * U, int ldu, float * V, int ldv, float * work, int lwork, float * rwork, int * devInfo)


.. py:function:: hipsolverDnDgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobu (:py:obj:`~.b`/:py:obj:`~.y`/:py:obj:`~.t`/:py:obj:`~.e`/:py:obj:`~.s`):
           (undocumented)

       jobv (:py:obj:`~.b`/:py:obj:`~.y`/:py:obj:`~.t`/:py:obj:`~.e`/:py:obj:`~.s`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       rwork (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDgesvd(hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, double * A, int lda, double * S, double * U, int ldu, double * V, int ldv, double * work, int lwork, double * rwork, int * devInfo)


.. py:function:: hipsolverDnCgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobu (:py:obj:`~.b`/:py:obj:`~.y`/:py:obj:`~.t`/:py:obj:`~.e`/:py:obj:`~.s`):
           (undocumented)

       jobv (:py:obj:`~.b`/:py:obj:`~.y`/:py:obj:`~.t`/:py:obj:`~.e`/:py:obj:`~.s`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       rwork (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCgesvd(hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, hipFloatComplex * A, int lda, float * S, hipFloatComplex * U, int ldu, hipFloatComplex * V, int ldv, hipFloatComplex * work, int lwork, float * rwork, int * devInfo)


.. py:function:: hipsolverDnZgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobu (:py:obj:`~.b`/:py:obj:`~.y`/:py:obj:`~.t`/:py:obj:`~.e`/:py:obj:`~.s`):
           (undocumented)

       jobv (:py:obj:`~.b`/:py:obj:`~.y`/:py:obj:`~.t`/:py:obj:`~.e`/:py:obj:`~.s`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       rwork (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZgesvd(hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, hipDoubleComplex * A, int lda, double * S, hipDoubleComplex * U, int ldu, hipDoubleComplex * V, int ldv, hipDoubleComplex * work, int lwork, double * rwork, int * devInfo)


.. py:function:: hipsolverDnSgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, params)

   // gesvdj

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       econ (:py:obj:`~.int`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSgesvdj_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int econ, int m, int n, const float * A, int lda, const float * S, const float * U, int ldu, const float * V, int ldv, int * lwork, hipsolverGesvdjInfo_t params)


.. py:function:: hipsolverDnDgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       econ (:py:obj:`~.int`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDgesvdj_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int econ, int m, int n, const double * A, int lda, const double * S, const double * U, int ldu, const double * V, int ldv, int * lwork, hipsolverGesvdjInfo_t params)


.. py:function:: hipsolverDnCgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       econ (:py:obj:`~.int`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCgesvdj_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int econ, int m, int n, const hipFloatComplex * A, int lda, const float * S, const hipFloatComplex * U, int ldu, const hipFloatComplex * V, int ldv, int * lwork, hipsolverGesvdjInfo_t params)


.. py:function:: hipsolverDnZgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       econ (:py:obj:`~.int`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZgesvdj_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int econ, int m, int n, const hipDoubleComplex * A, int lda, const double * S, const hipDoubleComplex * U, int ldu, const hipDoubleComplex * V, int ldv, int * lwork, hipsolverGesvdjInfo_t params)


.. py:function:: hipsolverDnSgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, devInfo, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       econ (:py:obj:`~.int`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSgesvdj(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int econ, int m, int n, float * A, int lda, float * S, float * U, int ldu, float * V, int ldv, float * work, int lwork, int * devInfo, hipsolverGesvdjInfo_t params)


.. py:function:: hipsolverDnDgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, devInfo, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       econ (:py:obj:`~.int`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDgesvdj(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int econ, int m, int n, double * A, int lda, double * S, double * U, int ldu, double * V, int ldv, double * work, int lwork, int * devInfo, hipsolverGesvdjInfo_t params)


.. py:function:: hipsolverDnCgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, devInfo, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       econ (:py:obj:`~.int`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCgesvdj(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int econ, int m, int n, hipFloatComplex * A, int lda, float * S, hipFloatComplex * U, int ldu, hipFloatComplex * V, int ldv, hipFloatComplex * work, int lwork, int * devInfo, hipsolverGesvdjInfo_t params)


.. py:function:: hipsolverDnZgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, devInfo, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       econ (:py:obj:`~.int`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZgesvdj(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int econ, int m, int n, hipDoubleComplex * A, int lda, double * S, hipDoubleComplex * U, int ldu, hipDoubleComplex * V, int ldv, hipDoubleComplex * work, int lwork, int * devInfo, hipsolverGesvdjInfo_t params)


.. py:function:: hipsolverDnSgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, params, batch_count)

   // gesvdj_batched

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSgesvdjBatched_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int m, int n, const float * A, int lda, const float * S, const float * U, int ldu, const float * V, int ldv, int * lwork, hipsolverGesvdjInfo_t params, int batch_count)


.. py:function:: hipsolverDnDgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDgesvdjBatched_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int m, int n, const double * A, int lda, const double * S, const double * U, int ldu, const double * V, int ldv, int * lwork, hipsolverGesvdjInfo_t params, int batch_count)


.. py:function:: hipsolverDnCgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCgesvdjBatched_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int m, int n, const hipFloatComplex * A, int lda, const float * S, const hipFloatComplex * U, int ldu, const hipFloatComplex * V, int ldv, int * lwork, hipsolverGesvdjInfo_t params, int batch_count)


.. py:function:: hipsolverDnZgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZgesvdjBatched_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int m, int n, const hipDoubleComplex * A, int lda, const double * S, const hipDoubleComplex * U, int ldu, const hipDoubleComplex * V, int ldv, int * lwork, hipsolverGesvdjInfo_t params, int batch_count)


.. py:function:: hipsolverDnSgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, devInfo, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSgesvdjBatched(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int m, int n, float * A, int lda, float * S, float * U, int ldu, float * V, int ldv, float * work, int lwork, int * devInfo, hipsolverGesvdjInfo_t params, int batch_count)


.. py:function:: hipsolverDnDgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, devInfo, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDgesvdjBatched(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int m, int n, double * A, int lda, double * S, double * U, int ldu, double * V, int ldv, double * work, int lwork, int * devInfo, hipsolverGesvdjInfo_t params, int batch_count)


.. py:function:: hipsolverDnCgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, devInfo, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCgesvdjBatched(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int m, int n, hipFloatComplex * A, int lda, float * S, hipFloatComplex * U, int ldu, hipFloatComplex * V, int ldv, hipFloatComplex * work, int lwork, int * devInfo, hipsolverGesvdjInfo_t params, int batch_count)


.. py:function:: hipsolverDnZgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, devInfo, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZgesvdjBatched(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int m, int n, hipDoubleComplex * A, int lda, double * S, hipDoubleComplex * U, int ldu, hipDoubleComplex * V, int ldv, hipDoubleComplex * work, int lwork, int * devInfo, hipsolverGesvdjInfo_t params, int batch_count)


.. py:function:: hipsolverDnSgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, batch_count)

   // gesvda_strided_batched

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       rank (:py:obj:`~.int`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       strideA (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       strideS (:py:obj:`~.int`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       strideU (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       strideV (:py:obj:`~.int`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSgesvdaStridedBatched_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int rank, int m, int n, const float * A, int lda, long long strideA, const float * S, long long strideS, const float * U, int ldu, long long strideU, const float * V, int ldv, long long strideV, int * lwork, int batch_count)


.. py:function:: hipsolverDnDgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       rank (:py:obj:`~.int`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       strideA (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       strideS (:py:obj:`~.int`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       strideU (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       strideV (:py:obj:`~.int`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDgesvdaStridedBatched_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int rank, int m, int n, const double * A, int lda, long long strideA, const double * S, long long strideS, const double * U, int ldu, long long strideU, const double * V, int ldv, long long strideV, int * lwork, int batch_count)


.. py:function:: hipsolverDnCgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       rank (:py:obj:`~.int`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       strideA (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       strideS (:py:obj:`~.int`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       strideU (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       strideV (:py:obj:`~.int`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCgesvdaStridedBatched_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int rank, int m, int n, const hipFloatComplex * A, int lda, long long strideA, const float * S, long long strideS, const hipFloatComplex * U, int ldu, long long strideU, const hipFloatComplex * V, int ldv, long long strideV, int * lwork, int batch_count)


.. py:function:: hipsolverDnZgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       rank (:py:obj:`~.int`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       strideA (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       strideS (:py:obj:`~.int`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       strideU (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       strideV (:py:obj:`~.int`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZgesvdaStridedBatched_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int rank, int m, int n, const hipDoubleComplex * A, int lda, long long strideA, const double * S, long long strideS, const hipDoubleComplex * U, int ldu, long long strideU, const hipDoubleComplex * V, int ldv, long long strideV, int * lwork, int batch_count)


.. py:function:: hipsolverDnSgesvdaStridedBatched(handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, work, lwork, devInfo, hRnrmF, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       rank (:py:obj:`~.int`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       strideA (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       strideS (:py:obj:`~.int`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       strideU (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       strideV (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       hRnrmF (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSgesvdaStridedBatched(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int rank, int m, int n, const float * A, int lda, long long strideA, float * S, long long strideS, float * U, int ldu, long long strideU, float * V, int ldv, long long strideV, float * work, int lwork, int * devInfo, double * hRnrmF, int batch_count)


.. py:function:: hipsolverDnDgesvdaStridedBatched(handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, work, lwork, devInfo, hRnrmF, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       rank (:py:obj:`~.int`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       strideA (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       strideS (:py:obj:`~.int`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       strideU (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       strideV (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       hRnrmF (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDgesvdaStridedBatched(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int rank, int m, int n, const double * A, int lda, long long strideA, double * S, long long strideS, double * U, int ldu, long long strideU, double * V, int ldv, long long strideV, double * work, int lwork, int * devInfo, double * hRnrmF, int batch_count)


.. py:function:: hipsolverDnCgesvdaStridedBatched(handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, work, lwork, devInfo, hRnrmF, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       rank (:py:obj:`~.int`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       strideA (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       strideS (:py:obj:`~.int`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       strideU (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       strideV (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       hRnrmF (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCgesvdaStridedBatched(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int rank, int m, int n, const hipFloatComplex * A, int lda, long long strideA, float * S, long long strideS, hipFloatComplex * U, int ldu, long long strideU, hipFloatComplex * V, int ldv, long long strideV, hipFloatComplex * work, int lwork, int * devInfo, double * hRnrmF, int batch_count)


.. py:function:: hipsolverDnZgesvdaStridedBatched(handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, work, lwork, devInfo, hRnrmF, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       rank (:py:obj:`~.int`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       strideA (:py:obj:`~.int`):
           (undocumented)

       S (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       strideS (:py:obj:`~.int`):
           (undocumented)

       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldu (:py:obj:`~.int`):
           (undocumented)

       strideU (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldv (:py:obj:`~.int`):
           (undocumented)

       strideV (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       hRnrmF (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZgesvdaStridedBatched(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int rank, int m, int n, const hipDoubleComplex * A, int lda, long long strideA, double * S, long long strideS, hipDoubleComplex * U, int ldu, long long strideU, hipDoubleComplex * V, int ldv, long long strideV, hipDoubleComplex * work, int lwork, int * devInfo, double * hRnrmF, int batch_count)


.. py:function:: hipsolverDnSgetrf_bufferSize(handle, m, n, A, lda)

   // getrf

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSgetrf_bufferSize(hipsolverHandle_t handle, int m, int n, float * A, int lda, int * lwork)


.. py:function:: hipsolverDnDgetrf_bufferSize(handle, m, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDgetrf_bufferSize(hipsolverHandle_t handle, int m, int n, double * A, int lda, int * lwork)


.. py:function:: hipsolverDnCgetrf_bufferSize(handle, m, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCgetrf_bufferSize(hipsolverHandle_t handle, int m, int n, hipFloatComplex * A, int lda, int * lwork)


.. py:function:: hipsolverDnZgetrf_bufferSize(handle, m, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZgetrf_bufferSize(hipsolverHandle_t handle, int m, int n, hipDoubleComplex * A, int lda, int * lwork)


.. py:function:: hipsolverDnSgetrf(handle, m, n, A, lda, work, devIpiv, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSgetrf(hipsolverHandle_t handle, int m, int n, float * A, int lda, float * work, int * devIpiv, int * devInfo)


.. py:function:: hipsolverDnDgetrf(handle, m, n, A, lda, work, devIpiv, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDgetrf(hipsolverHandle_t handle, int m, int n, double * A, int lda, double * work, int * devIpiv, int * devInfo)


.. py:function:: hipsolverDnCgetrf(handle, m, n, A, lda, work, devIpiv, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCgetrf(hipsolverHandle_t handle, int m, int n, hipFloatComplex * A, int lda, hipFloatComplex * work, int * devIpiv, int * devInfo)


.. py:function:: hipsolverDnZgetrf(handle, m, n, A, lda, work, devIpiv, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZgetrf(hipsolverHandle_t handle, int m, int n, hipDoubleComplex * A, int lda, hipDoubleComplex * work, int * devIpiv, int * devInfo)


.. py:function:: hipsolverDnSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)

   // getrs

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSgetrs(hipsolverHandle_t handle, hipblasOperation_t trans, int n, int nrhs, const float * A, int lda, const int * devIpiv, float * B, int ldb, int * devInfo)


.. py:function:: hipsolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDgetrs(hipsolverHandle_t handle, hipblasOperation_t trans, int n, int nrhs, const double * A, int lda, const int * devIpiv, double * B, int ldb, int * devInfo)


.. py:function:: hipsolverDnCgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCgetrs(hipsolverHandle_t handle, hipblasOperation_t trans, int n, int nrhs, const hipFloatComplex * A, int lda, const int * devIpiv, hipFloatComplex * B, int ldb, int * devInfo)


.. py:function:: hipsolverDnZgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZgetrs(hipsolverHandle_t handle, hipblasOperation_t trans, int n, int nrhs, const hipDoubleComplex * A, int lda, const int * devIpiv, hipDoubleComplex * B, int ldb, int * devInfo)


.. py:function:: hipsolverDnSpotrf_bufferSize(handle, uplo, n, A, lda)

   // potrf

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSpotrf_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, float * A, int lda, int * lwork)


.. py:function:: hipsolverDnDpotrf_bufferSize(handle, uplo, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDpotrf_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, double * A, int lda, int * lwork)


.. py:function:: hipsolverDnCpotrf_bufferSize(handle, uplo, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCpotrf_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipFloatComplex * A, int lda, int * lwork)


.. py:function:: hipsolverDnZpotrf_bufferSize(handle, uplo, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZpotrf_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipDoubleComplex * A, int lda, int * lwork)


.. py:function:: hipsolverDnSpotrf(handle, uplo, n, A, lda, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSpotrf(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, float * A, int lda, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnDpotrf(handle, uplo, n, A, lda, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDpotrf(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, double * A, int lda, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnCpotrf(handle, uplo, n, A, lda, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCpotrf(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipFloatComplex * A, int lda, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnZpotrf(handle, uplo, n, A, lda, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZpotrf(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipDoubleComplex * A, int lda, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnSpotrfBatched(handle, uplo, n, A, lda, devInfo, batch_count)

   // potrf_batched

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSpotrfBatched(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, float *[] A, int lda, int * devInfo, int batch_count)


.. py:function:: hipsolverDnDpotrfBatched(handle, uplo, n, A, lda, devInfo, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDpotrfBatched(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, double *[] A, int lda, int * devInfo, int batch_count)


.. py:function:: hipsolverDnCpotrfBatched(handle, uplo, n, A, lda, devInfo, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCpotrfBatched(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipFloatComplex *[] A, int lda, int * devInfo, int batch_count)


.. py:function:: hipsolverDnZpotrfBatched(handle, uplo, n, A, lda, devInfo, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZpotrfBatched(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipDoubleComplex *[] A, int lda, int * devInfo, int batch_count)


.. py:function:: hipsolverDnSpotri_bufferSize(handle, uplo, n, A, lda)

   // potri

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSpotri_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, float * A, int lda, int * lwork)


.. py:function:: hipsolverDnDpotri_bufferSize(handle, uplo, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDpotri_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, double * A, int lda, int * lwork)


.. py:function:: hipsolverDnCpotri_bufferSize(handle, uplo, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCpotri_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipFloatComplex * A, int lda, int * lwork)


.. py:function:: hipsolverDnZpotri_bufferSize(handle, uplo, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZpotri_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipDoubleComplex * A, int lda, int * lwork)


.. py:function:: hipsolverDnSpotri(handle, uplo, n, A, lda, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSpotri(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, float * A, int lda, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnDpotri(handle, uplo, n, A, lda, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDpotri(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, double * A, int lda, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnCpotri(handle, uplo, n, A, lda, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCpotri(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipFloatComplex * A, int lda, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnZpotri(handle, uplo, n, A, lda, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZpotri(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipDoubleComplex * A, int lda, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)

   // potrs

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSpotrs(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, int nrhs, const float * A, int lda, float * B, int ldb, int * devInfo)


.. py:function:: hipsolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDpotrs(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, int nrhs, const double * A, int lda, double * B, int ldb, int * devInfo)


.. py:function:: hipsolverDnCpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCpotrs(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, int nrhs, const hipFloatComplex * A, int lda, hipFloatComplex * B, int ldb, int * devInfo)


.. py:function:: hipsolverDnZpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZpotrs(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, int nrhs, const hipDoubleComplex * A, int lda, hipDoubleComplex * B, int ldb, int * devInfo)


.. py:function:: hipsolverDnSpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo, batch_count)

   // potrs_batched

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSpotrsBatched(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, int nrhs, float *[] A, int lda, float *[] B, int ldb, int * devInfo, int batch_count)


.. py:function:: hipsolverDnDpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDpotrsBatched(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, int nrhs, double *[] A, int lda, double *[] B, int ldb, int * devInfo, int batch_count)


.. py:function:: hipsolverDnCpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCpotrsBatched(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, int nrhs, hipFloatComplex *[] A, int lda, hipFloatComplex *[] B, int ldb, int * devInfo, int batch_count)


.. py:function:: hipsolverDnZpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZpotrsBatched(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, int nrhs, hipDoubleComplex *[] A, int lda, hipDoubleComplex *[] B, int ldb, int * devInfo, int batch_count)


.. py:function:: hipsolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W)

   // syevd/heevd

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSsyevd_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const float * A, int lda, const float * W, int * lwork)


.. py:function:: hipsolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDsyevd_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const double * A, int lda, const double * W, int * lwork)


.. py:function:: hipsolverDnCheevd_bufferSize(handle, jobz, uplo, n, A, lda, W)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCheevd_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const hipFloatComplex * A, int lda, const float * W, int * lwork)


.. py:function:: hipsolverDnZheevd_bufferSize(handle, jobz, uplo, n, A, lda, W)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZheevd_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const hipDoubleComplex * A, int lda, const double * W, int * lwork)


.. py:function:: hipsolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSsyevd(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, float * A, int lda, float * W, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDsyevd(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, double * A, int lda, double * W, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnCheevd(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCheevd(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, hipFloatComplex * A, int lda, float * W, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnZheevd(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZheevd(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, hipDoubleComplex * A, int lda, double * W, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnSsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W)

   // syevdx/heevdx

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSsyevdx_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const float * A, int lda, float vl, float vu, int il, int iu, int * nev, const float * W, int * lwork)


.. py:function:: hipsolverDnDsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDsyevdx_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const double * A, int lda, double vl, double vu, int il, int iu, int * nev, const double * W, int * lwork)


.. py:function:: hipsolverDnCheevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCheevdx_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const hipFloatComplex * A, int lda, float vl, float vu, int il, int iu, int * nev, const float * W, int * lwork)


.. py:function:: hipsolverDnZheevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZheevdx_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const hipDoubleComplex * A, int lda, double vl, double vu, int il, int iu, int * nev, const double * W, int * lwork)


.. py:function:: hipsolverDnSsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSsyevdx(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, float * A, int lda, float vl, float vu, int il, int iu, int * nev, float * W, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnDsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDsyevdx(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, double * A, int lda, double vl, double vu, int il, int iu, int * nev, double * W, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnCheevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCheevdx(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, hipFloatComplex * A, int lda, float vl, float vu, int il, int iu, int * nev, float * W, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnZheevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZheevdx(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, hipDoubleComplex * A, int lda, double vl, double vu, int il, int iu, int * nev, double * W, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnSsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, params)

   // syevj/heevj

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSsyevj_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const float * A, int lda, const float * W, int * lwork, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverDnDsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDsyevj_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const double * A, int lda, const double * W, int * lwork, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverDnCheevj_bufferSize(handle, jobz, uplo, n, A, lda, W, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCheevj_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const hipFloatComplex * A, int lda, const float * W, int * lwork, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverDnZheevj_bufferSize(handle, jobz, uplo, n, A, lda, W, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZheevj_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const hipDoubleComplex * A, int lda, const double * W, int * lwork, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverDnSsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSsyevj(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, float * A, int lda, float * W, float * work, int lwork, int * devInfo, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverDnDsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDsyevj(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, double * A, int lda, double * W, double * work, int lwork, int * devInfo, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverDnCheevj(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCheevj(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, hipFloatComplex * A, int lda, float * W, hipFloatComplex * work, int lwork, int * devInfo, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverDnZheevj(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZheevj(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, hipDoubleComplex * A, int lda, double * W, hipDoubleComplex * work, int lwork, int * devInfo, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverDnSsyevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, params, batch_count)

   // syevj_batched/heevj_batched

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSsyevjBatched_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const float * A, int lda, const float * W, int * lwork, hipsolverSyevjInfo_t params, int batch_count)


.. py:function:: hipsolverDnDsyevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDsyevjBatched_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const double * A, int lda, const double * W, int * lwork, hipsolverSyevjInfo_t params, int batch_count)


.. py:function:: hipsolverDnCheevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCheevjBatched_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const hipFloatComplex * A, int lda, const float * W, int * lwork, hipsolverSyevjInfo_t params, int batch_count)


.. py:function:: hipsolverDnZheevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZheevjBatched_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const hipDoubleComplex * A, int lda, const double * W, int * lwork, hipsolverSyevjInfo_t params, int batch_count)


.. py:function:: hipsolverDnSsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSsyevjBatched(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, float * A, int lda, float * W, float * work, int lwork, int * devInfo, hipsolverSyevjInfo_t params, int batch_count)


.. py:function:: hipsolverDnDsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDsyevjBatched(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, double * A, int lda, double * W, double * work, int lwork, int * devInfo, hipsolverSyevjInfo_t params, int batch_count)


.. py:function:: hipsolverDnCheevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCheevjBatched(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, hipFloatComplex * A, int lda, float * W, hipFloatComplex * work, int lwork, int * devInfo, hipsolverSyevjInfo_t params, int batch_count)


.. py:function:: hipsolverDnZheevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, params, batch_count)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       batch_count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZheevjBatched(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, hipDoubleComplex * A, int lda, double * W, hipDoubleComplex * work, int lwork, int * devInfo, hipsolverSyevjInfo_t params, int batch_count)


.. py:function:: hipsolverDnSsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W)

   // sygvd/hegvd

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSsygvd_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const float * A, int lda, const float * B, int ldb, const float * W, int * lwork)


.. py:function:: hipsolverDnDsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDsygvd_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const double * A, int lda, const double * B, int ldb, const double * W, int * lwork)


.. py:function:: hipsolverDnChegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnChegvd_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const hipFloatComplex * A, int lda, const hipFloatComplex * B, int ldb, const float * W, int * lwork)


.. py:function:: hipsolverDnZhegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZhegvd_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const hipDoubleComplex * A, int lda, const hipDoubleComplex * B, int ldb, const double * W, int * lwork)


.. py:function:: hipsolverDnSsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSsygvd(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, float * A, int lda, float * B, int ldb, float * W, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnDsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDsygvd(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, double * A, int lda, double * B, int ldb, double * W, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnChegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnChegvd(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, hipFloatComplex * A, int lda, hipFloatComplex * B, int ldb, float * W, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnZhegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZhegvd(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, hipDoubleComplex * A, int lda, hipDoubleComplex * B, int ldb, double * W, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnSsygvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, nev, W)

   // sygvdx/hegvdx

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSsygvdx_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const float * A, int lda, const float * B, int ldb, float vl, float vu, int il, int iu, int * nev, const float * W, int * lwork)


.. py:function:: hipsolverDnDsygvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, nev, W)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDsygvdx_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const double * A, int lda, const double * B, int ldb, double vl, double vu, int il, int iu, int * nev, const double * W, int * lwork)


.. py:function:: hipsolverDnChegvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, nev, W)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnChegvdx_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const hipFloatComplex * A, int lda, const hipFloatComplex * B, int ldb, float vl, float vu, int il, int iu, int * nev, const float * W, int * lwork)


.. py:function:: hipsolverDnZhegvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, nev, W)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZhegvdx_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const hipDoubleComplex * A, int lda, const hipDoubleComplex * B, int ldb, double vl, double vu, int il, int iu, int * nev, const double * W, int * lwork)


.. py:function:: hipsolverDnSsygvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, nev, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSsygvdx(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, float * A, int lda, float * B, int ldb, float vl, float vu, int il, int iu, int * nev, float * W, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnDsygvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, nev, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDsygvdx(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, double * A, int lda, double * B, int ldb, double vl, double vu, int il, int iu, int * nev, double * W, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnChegvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, nev, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnChegvdx(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, hipFloatComplex * A, int lda, hipFloatComplex * B, int ldb, float vl, float vu, int il, int iu, int * nev, float * W, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnZhegvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, nev, W, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       range (:py:obj:`~.hipsolverEigRange_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       vl (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       vu (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       il (:py:obj:`~.int`):
           (undocumented)

       iu (:py:obj:`~.int`):
           (undocumented)

       nev (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZhegvdx(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, hipDoubleComplex * A, int lda, hipDoubleComplex * B, int ldb, double vl, double vu, int il, int iu, int * nev, double * W, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnSsygvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, params)

   // sygvj/hegvj

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSsygvj_bufferSize(hipsolverDnHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const float * A, int lda, const float * B, int ldb, const float * W, int * lwork, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverDnDsygvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDsygvj_bufferSize(hipsolverDnHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const double * A, int lda, const double * B, int ldb, const double * W, int * lwork, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverDnChegvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnChegvj_bufferSize(hipsolverDnHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const hipFloatComplex * A, int lda, const hipFloatComplex * B, int ldb, const float * W, int * lwork, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverDnZhegvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZhegvj_bufferSize(hipsolverDnHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const hipDoubleComplex * A, int lda, const hipDoubleComplex * B, int ldb, const double * W, int * lwork, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverDnSsygvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, devInfo, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSsygvj(hipsolverDnHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, float * A, int lda, float * B, int ldb, float * W, float * work, int lwork, int * devInfo, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverDnDsygvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, devInfo, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDsygvj(hipsolverDnHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, double * A, int lda, double * B, int ldb, double * W, double * work, int lwork, int * devInfo, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverDnChegvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, devInfo, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnChegvj(hipsolverDnHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, hipFloatComplex * A, int lda, hipFloatComplex * B, int ldb, float * W, hipFloatComplex * work, int lwork, int * devInfo, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverDnZhegvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, devInfo, params)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       itype (:py:obj:`~.hipsolverEigType_t`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZhegvj(hipsolverDnHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, hipDoubleComplex * A, int lda, hipDoubleComplex * B, int ldb, double * W, hipDoubleComplex * work, int lwork, int * devInfo, hipsolverSyevjInfo_t params)


.. py:function:: hipsolverDnSsytrd_bufferSize(handle, uplo, n, A, lda, D, E, tau)

   // sytrd/hetrd

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       E (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSsytrd_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, const float * A, int lda, const float * D, const float * E, const float * tau, int * lwork)


.. py:function:: hipsolverDnDsytrd_bufferSize(handle, uplo, n, A, lda, D, E, tau)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       E (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDsytrd_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, const double * A, int lda, const double * D, const double * E, const double * tau, int * lwork)


.. py:function:: hipsolverDnChetrd_bufferSize(handle, uplo, n, A, lda, D, E, tau)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       E (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnChetrd_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, const hipFloatComplex * A, int lda, const float * D, const float * E, const hipFloatComplex * tau, int * lwork)


.. py:function:: hipsolverDnZhetrd_bufferSize(handle, uplo, n, A, lda, D, E, tau)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       E (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZhetrd_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, const hipDoubleComplex * A, int lda, const double * D, const double * E, const hipDoubleComplex * tau, int * lwork)


.. py:function:: hipsolverDnSsytrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       E (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSsytrd(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, float * A, int lda, float * D, float * E, float * tau, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnDsytrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       E (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDsytrd(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, double * A, int lda, double * D, double * E, double * tau, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnChetrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       E (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnChetrd(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipFloatComplex * A, int lda, float * D, float * E, hipFloatComplex * tau, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnZhetrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       E (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZhetrd(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipDoubleComplex * A, int lda, double * D, double * E, hipDoubleComplex * tau, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnSsytrf_bufferSize(handle, n, A, lda)

   // sytrf

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSsytrf_bufferSize(hipsolverHandle_t handle, int n, float * A, int lda, int * lwork)


.. py:function:: hipsolverDnDsytrf_bufferSize(handle, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDsytrf_bufferSize(hipsolverHandle_t handle, int n, double * A, int lda, int * lwork)


.. py:function:: hipsolverDnCsytrf_bufferSize(handle, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCsytrf_bufferSize(hipsolverHandle_t handle, int n, hipFloatComplex * A, int lda, int * lwork)


.. py:function:: hipsolverDnZsytrf_bufferSize(handle, n, A, lda)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lwork (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZsytrf_bufferSize(hipsolverHandle_t handle, int n, hipDoubleComplex * A, int lda, int * lwork)


.. py:function:: hipsolverDnSsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       ipiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSsytrf(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, float * A, int lda, int * ipiv, float * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnDsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       ipiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDsytrf(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, double * A, int lda, int * ipiv, double * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnCsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       ipiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCsytrf(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipFloatComplex * A, int lda, int * ipiv, hipFloatComplex * work, int lwork, int * devInfo)


.. py:function:: hipsolverDnZsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       ipiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       work (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lwork (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnZsytrf(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipDoubleComplex * A, int lda, int * ipiv, hipDoubleComplex * work, int lwork, int * devInfo)


.. py:class:: hipsolverAlgMode_t

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: HIPSOLVER_ALG_0
      :type:  int


   .. py:attribute:: HIPSOLVER_ALG_1
      :type:  int


.. py:class:: hipsolverDnFunction_t

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: HIPSOLVERDN_GETRF
      :type:  int


.. py:function:: hipsolverDnCreateParams()

   (No short description, might be part of a group.)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnCreateParams(hipsolverDnParams_t * params)


.. py:function:: hipsolverDnDestroyParams(params)

   (No short description, might be part of a group.)

   Args:
       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnDestroyParams(hipsolverDnParams_t params)


.. py:function:: hipsolverDnSetAdvOptions(params, func, alg)

   (No short description, might be part of a group.)

   Args:
       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       func (:py:obj:`~.hipsolverDnFunction_t`):
           (undocumented)

       alg (:py:obj:`~.hipsolverAlgMode_t`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnSetAdvOptions(hipsolverDnParams_t params, hipsolverDnFunction_t func, hipsolverAlgMode_t alg)


.. py:function:: hipsolverDnXgeev_bufferSize(handle, params, jobvl, jobvr, n, dataTypeA, A, lda, dataTypeW, W, dataTypeVL, VL, ldvl, dataTypeVR, VR, ldvr, computeType)

   // geev

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobvl (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       jobvr (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       dataTypeA (:py:obj:`~.hipDataType`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       dataTypeW (:py:obj:`~.hipDataType`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       dataTypeVL (:py:obj:`~.hipDataType`):
           (undocumented)

       VL (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldvl (:py:obj:`~.int`):
           (undocumented)

       dataTypeVR (:py:obj:`~.hipDataType`):
           (undocumented)

       VR (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldvr (:py:obj:`~.int`):
           (undocumented)

       computeType (:py:obj:`~.hipDataType`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lworkOnDevice (:py:obj:`~.int`):
           (undocumented)
       * lworkOnHost (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnXgeev_bufferSize(hipsolverDnHandle_t handle, hipsolverDnParams_t params, hipsolverEigMode_t jobvl, hipsolverEigMode_t jobvr, int64_t n, hipDataType dataTypeA, const void * A, int64_t lda, hipDataType dataTypeW, const void * W, hipDataType dataTypeVL, const void * VL, int64_t ldvl, hipDataType dataTypeVR, const void * VR, int64_t ldvr, hipDataType computeType, size_t * lworkOnDevice, size_t * lworkOnHost)


.. py:function:: hipsolverDnXgeev(handle, params, jobvl, jobvr, n, dataTypeA, A, lda, dataTypeW, W, dataTypeVL, VL, ldvl, dataTypeVR, VR, ldvr, computeType, workOnDevice, lworkOnDevice, workOnHost, lworkOnHost, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobvl (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       jobvr (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       dataTypeA (:py:obj:`~.hipDataType`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       dataTypeW (:py:obj:`~.hipDataType`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       dataTypeVL (:py:obj:`~.hipDataType`):
           (undocumented)

       VL (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldvl (:py:obj:`~.int`):
           (undocumented)

       dataTypeVR (:py:obj:`~.hipDataType`):
           (undocumented)

       VR (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldvr (:py:obj:`~.int`):
           (undocumented)

       computeType (:py:obj:`~.hipDataType`):
           (undocumented)

       workOnDevice (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lworkOnDevice (:py:obj:`~.int`):
           (undocumented)

       workOnHost (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lworkOnHost (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnXgeev(hipsolverDnHandle_t handle, hipsolverDnParams_t params, hipsolverEigMode_t jobvl, hipsolverEigMode_t jobvr, int64_t n, hipDataType dataTypeA, void * A, int64_t lda, hipDataType dataTypeW, void * W, hipDataType dataTypeVL, void * VL, int64_t ldvl, hipDataType dataTypeVR, void * VR, int64_t ldvr, hipDataType computeType, void * workOnDevice, size_t lworkOnDevice, void * workOnHost, size_t lworkOnHost, int * devInfo)


.. py:function:: hipsolverDnXgeqrf_bufferSize(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType)

   // geqrf

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       dataTypeA (:py:obj:`~.hipDataType`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       dataTypeTau (:py:obj:`~.hipDataType`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       computeType (:py:obj:`~.hipDataType`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lworkOnDevice (:py:obj:`~.int`):
           (undocumented)
       * lworkOnHost (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnXgeqrf_bufferSize(hipsolverDnHandle_t handle, hipsolverDnParams_t params, int64_t m, int64_t n, hipDataType dataTypeA, const void * A, int64_t lda, hipDataType dataTypeTau, const void * tau, hipDataType computeType, size_t * lworkOnDevice, size_t * lworkOnHost)


.. py:function:: hipsolverDnXgeqrf(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, workOnDevice, lworkOnDevice, workOnHost, lworkOnHost, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       dataTypeA (:py:obj:`~.hipDataType`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       dataTypeTau (:py:obj:`~.hipDataType`):
           (undocumented)

       tau (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       computeType (:py:obj:`~.hipDataType`):
           (undocumented)

       workOnDevice (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lworkOnDevice (:py:obj:`~.int`):
           (undocumented)

       workOnHost (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lworkOnHost (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnXgeqrf(hipsolverDnHandle_t handle, hipsolverDnParams_t params, int64_t m, int64_t n, hipDataType dataTypeA, void * A, int64_t lda, hipDataType dataTypeTau, void * tau, hipDataType computeType, void * workOnDevice, size_t lworkOnDevice, void * workOnHost, size_t lworkOnHost, int * devInfo)


.. py:function:: hipsolverDnXgetrf_bufferSize(handle, params, m, n, dataTypeA, A, lda, computeType)

   // getrf

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       dataTypeA (:py:obj:`~.hipDataType`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       computeType (:py:obj:`~.hipDataType`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lworkOnDevice (:py:obj:`~.int`):
           (undocumented)
       * lworkOnHost (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnXgetrf_bufferSize(hipsolverDnHandle_t handle, hipsolverDnParams_t params, int64_t m, int64_t n, hipDataType dataTypeA, const void * A, int64_t lda, hipDataType computeType, size_t * lworkOnDevice, size_t * lworkOnHost)


.. py:function:: hipsolverDnXgetrf(handle, params, m, n, dataTypeA, A, lda, devIpiv, computeType, workOnDevice, lworkOnDevice, workOnHost, lworkOnHost, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       m (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       dataTypeA (:py:obj:`~.hipDataType`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt64`/:py:obj:`~.object`):
           (undocumented)

       computeType (:py:obj:`~.hipDataType`):
           (undocumented)

       workOnDevice (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lworkOnDevice (:py:obj:`~.int`):
           (undocumented)

       workOnHost (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lworkOnHost (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnXgetrf(hipsolverDnHandle_t handle, hipsolverDnParams_t params, int64_t m, int64_t n, hipDataType dataTypeA, void * A, int64_t lda, int64_t * devIpiv, hipDataType computeType, void * workOnDevice, size_t lworkOnDevice, void * workOnHost, size_t lworkOnHost, int * devInfo)


.. py:function:: hipsolverDnXgetrs(handle, params, trans, n, nrhs, dataTypeA, A, lda, devIpiv, dataTypeB, B, ldb, devInfo)

   // getrs

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       trans (:py:obj:`~.hipblasOperation_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       dataTypeA (:py:obj:`~.hipDataType`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt64`/:py:obj:`~.object`):
           (undocumented)

       dataTypeB (:py:obj:`~.hipDataType`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnXgetrs(hipsolverDnHandle_t handle, hipsolverDnParams_t params, hipsolverOperation_t trans, int64_t n, int64_t nrhs, hipDataType dataTypeA, const void * A, int64_t lda, const int64_t * devIpiv, hipDataType dataTypeB, void * B, int64_t ldb, int * devInfo)


.. py:function:: hipsolverDnXpotrf_bufferSize(handle, params, uplo, n, dataTypeA, A, lda, computeType)

   // potrf

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       dataTypeA (:py:obj:`~.hipDataType`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       computeType (:py:obj:`~.hipDataType`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lworkOnDevice (:py:obj:`~.int`):
           (undocumented)
       * lworkOnHost (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnXpotrf_bufferSize(hipsolverDnHandle_t handle, hipsolverDnParams_t params, hipsolverFillMode_t uplo, int64_t n, hipDataType dataTypeA, const void * A, int64_t lda, hipDataType computeType, size_t * lworkOnDevice, size_t * lworkOnHost)


.. py:function:: hipsolverDnXpotrf(handle, params, uplo, n, dataTypeA, A, lda, computeType, workOnDevice, lworkOnDevice, workOnHost, lworkOnHost, info)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       dataTypeA (:py:obj:`~.hipDataType`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       computeType (:py:obj:`~.hipDataType`):
           (undocumented)

       workOnDevice (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lworkOnDevice (:py:obj:`~.int`):
           (undocumented)

       workOnHost (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lworkOnHost (:py:obj:`~.int`):
           (undocumented)

       info (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnXpotrf(hipsolverDnHandle_t handle, hipsolverDnParams_t params, hipsolverFillMode_t uplo, int64_t n, hipDataType dataTypeA, void * A, int64_t lda, hipDataType computeType, void * workOnDevice, size_t lworkOnDevice, void * workOnHost, size_t lworkOnHost, int * info)


.. py:function:: hipsolverDnXpotrs(handle, params, uplo, n, nrhs, dataTypeA, A, lda, dataTypeB, B, ldb, info)

   // potrs

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       dataTypeA (:py:obj:`~.hipDataType`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       dataTypeB (:py:obj:`~.hipDataType`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       info (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnXpotrs(hipsolverDnHandle_t handle, hipsolverDnParams_t params, hipsolverFillMode_t uplo, int64_t n, int64_t nrhs, hipDataType dataTypeA, const void * A, int64_t lda, hipDataType dataTypeB, void * B, int64_t ldb, int * info)


.. py:function:: hipsolverDnXsyevd_bufferSize(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType)

   // syevd

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       dataTypeA (:py:obj:`~.hipDataType`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       dataTypeW (:py:obj:`~.hipDataType`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       computeType (:py:obj:`~.hipDataType`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lworkOnDevice (:py:obj:`~.int`):
           (undocumented)
       * lworkOnHost (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnXsyevd_bufferSize(hipsolverDnHandle_t handle, hipsolverDnParams_t params, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int64_t n, hipDataType dataTypeA, const void * A, int64_t lda, hipDataType dataTypeW, const void * W, hipDataType computeType, size_t * lworkOnDevice, size_t * lworkOnHost)


.. py:function:: hipsolverDnXsyevd(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, workOnDevice, lworkOnDevice, workOnHost, lworkOnHost, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       dataTypeA (:py:obj:`~.hipDataType`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       dataTypeW (:py:obj:`~.hipDataType`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       computeType (:py:obj:`~.hipDataType`):
           (undocumented)

       workOnDevice (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lworkOnDevice (:py:obj:`~.int`):
           (undocumented)

       workOnHost (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lworkOnHost (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnXsyevd(hipsolverDnHandle_t handle, hipsolverDnParams_t params, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int64_t n, hipDataType dataTypeA, void * A, int64_t lda, hipDataType dataTypeW, void * W, hipDataType computeType, void * workOnDevice, size_t lworkOnDevice, void * workOnHost, size_t lworkOnHost, int * devInfo)


.. py:function:: hipsolverDnXsyevBatched_bufferSize(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, batchSize)

   // syevBatched

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       dataTypeA (:py:obj:`~.hipDataType`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       dataTypeW (:py:obj:`~.hipDataType`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       computeType (:py:obj:`~.hipDataType`):
           (undocumented)

       batchSize (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lworkOnDevice (:py:obj:`~.int`):
           (undocumented)
       * lworkOnHost (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnXsyevBatched_bufferSize(hipsolverDnHandle_t handle, hipsolverDnParams_t params, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int64_t n, hipDataType dataTypeA, const void * A, int64_t lda, hipDataType dataTypeW, const void * W, hipDataType computeType, size_t * lworkOnDevice, size_t * lworkOnHost, int64_t batchSize)


.. py:function:: hipsolverDnXsyevBatched(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, workOnDevice, lworkOnDevice, workOnHost, lworkOnHost, devInfo, batchSize)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       params (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       jobz (:py:obj:`~.hipsolverEigMode_t`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       dataTypeA (:py:obj:`~.hipDataType`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       dataTypeW (:py:obj:`~.hipDataType`):
           (undocumented)

       W (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       computeType (:py:obj:`~.hipDataType`):
           (undocumented)

       workOnDevice (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lworkOnDevice (:py:obj:`~.int`):
           (undocumented)

       workOnHost (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lworkOnHost (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       batchSize (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnXsyevBatched(hipsolverDnHandle_t handle, hipsolverDnParams_t params, hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int64_t n, hipDataType dataTypeA, void * A, int64_t lda, hipDataType dataTypeW, void * W, hipDataType computeType, void * workOnDevice, size_t lworkOnDevice, void * workOnHost, size_t lworkOnHost, int * devInfo, int64_t batchSize)


.. py:function:: hipsolverDnXsytrs_bufferSize(handle, uplo, n, nrhs, dataTypeA, A, lda, devIpiv, dataTypeB, B, ldb)

   // sytrs

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       dataTypeA (:py:obj:`~.hipDataType`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt64`/:py:obj:`~.object`):
           (undocumented)

       dataTypeB (:py:obj:`~.hipDataType`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * lworkOnDevice (:py:obj:`~.int`):
           (undocumented)
       * lworkOnHost (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnXsytrs_bufferSize(hipsolverDnHandle_t handle, hipsolverFillMode_t uplo, int64_t n, int64_t nrhs, hipDataType dataTypeA, const void * A, int64_t lda, const int64_t * devIpiv, hipDataType dataTypeB, void * B, int64_t ldb, size_t * lworkOnDevice, size_t * lworkOnHost)


.. py:function:: hipsolverDnXsytrs(handle, uplo, n, nrhs, dataTypeA, A, lda, devIpiv, dataTypeB, B, ldb, workOnDevice, lworkOnDevice, workOnHost, lworkOnHost, devInfo)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       uplo (:py:obj:`~.hipblasFillMode_t`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       dataTypeA (:py:obj:`~.hipDataType`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lda (:py:obj:`~.int`):
           (undocumented)

       devIpiv (:py:obj:`~.rocm.bindings.util.types.ListOfInt64`/:py:obj:`~.object`):
           (undocumented)

       dataTypeB (:py:obj:`~.hipDataType`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldb (:py:obj:`~.int`):
           (undocumented)

       workOnDevice (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lworkOnDevice (:py:obj:`~.int`):
           (undocumented)

       workOnHost (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       lworkOnHost (:py:obj:`~.int`):
           (undocumented)

       devInfo (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverDnXsytrs(hipsolverDnHandle_t handle, hipsolverFillMode_t uplo, int64_t n, int64_t nrhs, hipDataType dataTypeA, const void * A, int64_t lda, const int64_t * devIpiv, hipDataType dataTypeB, void * B, int64_t ldb, void * workOnDevice, size_t lworkOnDevice, void * workOnHost, size_t lworkOnHost, int * devInfo)


.. py:class:: hipsolverRfFactorization_t

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: HIPSOLVERRF_FACTORIZATION_ALG0
      :type:  int


   .. py:attribute:: HIPSOLVERRF_FACTORIZATION_ALG1
      :type:  int


   .. py:attribute:: HIPSOLVERRF_FACTORIZATION_ALG2
      :type:  int


.. py:class:: hipsolverRfMatrixFormat_t

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: HIPSOLVERRF_MATRIX_FORMAT_CSR
      :type:  int


   .. py:attribute:: HIPSOLVERRF_MATRIX_FORMAT_CSC
      :type:  int


.. py:class:: hipsolverRfNumericBoostReport_t

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: HIPSOLVERRF_NUMERIC_BOOST_NOT_USED
      :type:  int


   .. py:attribute:: HIPSOLVERRF_NUMERIC_BOOST_USED
      :type:  int


.. py:class:: hipsolverRfResetValuesFastMode_t

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: HIPSOLVERRF_RESET_VALUES_FAST_MODE_OFF
      :type:  int


   .. py:attribute:: HIPSOLVERRF_RESET_VALUES_FAST_MODE_ON
      :type:  int


.. py:class:: hipsolverRfTriangularSolve_t

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: HIPSOLVERRF_TRIANGULAR_SOLVE_ALG1
      :type:  int


   .. py:attribute:: HIPSOLVERRF_TRIANGULAR_SOLVE_ALG2
      :type:  int


   .. py:attribute:: HIPSOLVERRF_TRIANGULAR_SOLVE_ALG3
      :type:  int


.. py:class:: hipsolverRfUnitDiagonal_t

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: HIPSOLVERRF_UNIT_DIAGONAL_STORED_L
      :type:  int


   .. py:attribute:: HIPSOLVERRF_UNIT_DIAGONAL_STORED_U
      :type:  int


   .. py:attribute:: HIPSOLVERRF_UNIT_DIAGONAL_ASSUMED_L
      :type:  int


   .. py:attribute:: HIPSOLVERRF_UNIT_DIAGONAL_ASSUMED_U
      :type:  int


.. py:function:: hipsolverRfCreate()

   (No short description, might be part of a group.)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverRfCreate(hipsolverRfHandle_t * handle)


.. py:function:: hipsolverRfDestroy(handle)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverRfDestroy(hipsolverRfHandle_t handle)


.. py:function:: hipsolverRfSetupDevice(n, nnzA, csrRowPtrA, csrColIndA, csrValA, nnzL, csrRowPtrL, csrColIndL, csrValL, nnzU, csrRowPtrU, csrColIndU, csrValU, P, Q, handle)

   // non-batched routines

   Args:
       n (:py:obj:`~.int`):
           (undocumented)

       nnzA (:py:obj:`~.int`):
           (undocumented)

       csrRowPtrA (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       csrColIndA (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       csrValA (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       nnzL (:py:obj:`~.int`):
           (undocumented)

       csrRowPtrL (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       csrColIndL (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       csrValL (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       nnzU (:py:obj:`~.int`):
           (undocumented)

       csrRowPtrU (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       csrColIndU (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       csrValU (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       P (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       Q (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverRfSetupDevice(int n, int nnzA, int * csrRowPtrA, int * csrColIndA, double * csrValA, int nnzL, int * csrRowPtrL, int * csrColIndL, double * csrValL, int nnzU, int * csrRowPtrU, int * csrColIndU, double * csrValU, int * P, int * Q, hipsolverRfHandle_t handle)


.. py:function:: hipsolverRfSetupHost(n, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA, nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL, nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU, h_P, h_Q, handle)

   (No short description, might be part of a group.)

   Args:
       n (:py:obj:`~.int`):
           (undocumented)

       nnzA (:py:obj:`~.int`):
           (undocumented)

       h_csrRowPtrA (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       h_csrColIndA (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       h_csrValA (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       nnzL (:py:obj:`~.int`):
           (undocumented)

       h_csrRowPtrL (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       h_csrColIndL (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       h_csrValL (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       nnzU (:py:obj:`~.int`):
           (undocumented)

       h_csrRowPtrU (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       h_csrColIndU (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       h_csrValU (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       h_P (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       h_Q (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverRfSetupHost(int n, int nnzA, int * h_csrRowPtrA, int * h_csrColIndA, double * h_csrValA, int nnzL, int * h_csrRowPtrL, int * h_csrColIndL, double * h_csrValL, int nnzU, int * h_csrRowPtrU, int * h_csrColIndU, double * h_csrValU, int * h_P, int * h_Q, hipsolverRfHandle_t handle)


.. py:function:: hipsolverRfAccessBundledFactorsDevice(handle, nnzM)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       nnzM (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 4 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * Mp (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)
       * Mi (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)
       * Mx (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverRfAccessBundledFactorsDevice(hipsolverRfHandle_t handle, int * nnzM, int ** Mp, int ** Mi, double ** Mx)


.. py:function:: hipsolverRfAnalyze(handle)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverRfAnalyze(hipsolverRfHandle_t handle)


.. py:function:: hipsolverRfExtractBundledFactorsHost(handle, h_nnzM)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       h_nnzM (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 4 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * h_Mp (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)
       * h_Mi (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)
       * h_Mx (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverRfExtractBundledFactorsHost(hipsolverRfHandle_t handle, int * h_nnzM, int ** h_Mp, int ** h_Mi, double ** h_Mx)


.. py:function:: hipsolverRfExtractSplitFactorsHost(handle, h_nnzL, h_nnzU)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       h_nnzL (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       h_nnzU (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 7 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * h_Lp (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)
       * h_Li (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)
       * h_Lx (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)
       * h_Up (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)
       * h_Ui (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)
       * h_Ux (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverRfExtractSplitFactorsHost(hipsolverRfHandle_t handle, int * h_nnzL, int ** h_Lp, int ** h_Li, double ** h_Lx, int * h_nnzU, int ** h_Up, int ** h_Ui, double ** h_Ux)


.. py:function:: hipsolverRfGet_Algs(handle, fact_alg, solve_alg)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       fact_alg (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       solve_alg (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverRfGet_Algs(hipsolverRfHandle_t handle, hipsolverRfFactorization_t * fact_alg, hipsolverRfTriangularSolve_t * solve_alg)


.. py:function:: hipsolverRfGetMatrixFormat(handle, format, diag)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       format (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       diag (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverRfGetMatrixFormat(hipsolverRfHandle_t handle, hipsolverRfMatrixFormat_t * format, hipsolverRfUnitDiagonal_t * diag)


.. py:function:: hipsolverRfGetNumericBoostReport(handle, report)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       report (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverRfGetNumericBoostReport(hipsolverRfHandle_t handle, hipsolverRfNumericBoostReport_t * report)


.. py:function:: hipsolverRfGetNumericProperties(handle, zero, boost)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       zero (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       boost (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverRfGetNumericProperties(hipsolverRfHandle_t handle, double * zero, double * boost)


.. py:function:: hipsolverRfGetResetValuesFastMode(handle, fastMode)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       fastMode (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverRfGetResetValuesFastMode(hipsolverRfHandle_t handle, hipsolverRfResetValuesFastMode_t * fastMode)


.. py:function:: hipsolverRfRefactor(handle)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverRfRefactor(hipsolverRfHandle_t handle)


.. py:function:: hipsolverRfResetValues(n, nnzA, csrRowPtrA, csrColIndA, csrValA, P, Q, handle)

   (No short description, might be part of a group.)

   Args:
       n (:py:obj:`~.int`):
           (undocumented)

       nnzA (:py:obj:`~.int`):
           (undocumented)

       csrRowPtrA (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       csrColIndA (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       csrValA (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       P (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       Q (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverRfResetValues(int n, int nnzA, int * csrRowPtrA, int * csrColIndA, double * csrValA, int * P, int * Q, hipsolverRfHandle_t handle)


.. py:function:: hipsolverRfSetAlgs(handle, fact_alg, solve_alg)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       fact_alg (:py:obj:`~.hipsolverRfFactorization_t`):
           (undocumented)

       solve_alg (:py:obj:`~.hipsolverRfTriangularSolve_t`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverRfSetAlgs(hipsolverRfHandle_t handle, hipsolverRfFactorization_t fact_alg, hipsolverRfTriangularSolve_t solve_alg)


.. py:function:: hipsolverRfSetMatrixFormat(handle, format, diag)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       format (:py:obj:`~.hipsolverRfMatrixFormat_t`):
           (undocumented)

       diag (:py:obj:`~.hipsolverRfUnitDiagonal_t`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverRfSetMatrixFormat(hipsolverRfHandle_t handle, hipsolverRfMatrixFormat_t format, hipsolverRfUnitDiagonal_t diag)


.. py:function:: hipsolverRfSetNumericProperties(handle, effective_zero, boost_val)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       effective_zero (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       boost_val (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverRfSetNumericProperties(hipsolverRfHandle_t handle, double effective_zero, double boost_val)


.. py:function:: hipsolverRfSetResetValuesFastMode(handle, fastMode)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       fastMode (:py:obj:`~.hipsolverRfResetValuesFastMode_t`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverRfSetResetValuesFastMode(hipsolverRfHandle_t handle, hipsolverRfResetValuesFastMode_t fastMode)


.. py:function:: hipsolverRfSolve(handle, P, Q, nrhs, Temp, ldt, XF, ldxf)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       P (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       Q (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       Temp (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldt (:py:obj:`~.int`):
           (undocumented)

       XF (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldxf (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverRfSolve(hipsolverRfHandle_t handle, int * P, int * Q, int nrhs, double * Temp, int ldt, double * XF, int ldxf)


.. py:function:: hipsolverRfBatchSetupHost(batchSize, n, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA_array, nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL, nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU, h_P, h_Q, handle)

   // batched routines

   Args:
       batchSize (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nnzA (:py:obj:`~.int`):
           (undocumented)

       h_csrRowPtrA (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       h_csrColIndA (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       h_csrValA_array (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       nnzL (:py:obj:`~.int`):
           (undocumented)

       h_csrRowPtrL (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       h_csrColIndL (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       h_csrValL (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       nnzU (:py:obj:`~.int`):
           (undocumented)

       h_csrRowPtrU (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       h_csrColIndU (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       h_csrValU (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       h_P (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       h_Q (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverRfBatchSetupHost(int batchSize, int n, int nnzA, int * h_csrRowPtrA, int * h_csrColIndA, double *[] h_csrValA_array, int nnzL, int * h_csrRowPtrL, int * h_csrColIndL, double * h_csrValL, int nnzU, int * h_csrRowPtrU, int * h_csrColIndU, double * h_csrValU, int * h_P, int * h_Q, hipsolverRfHandle_t handle)


.. py:function:: hipsolverRfBatchAnalyze(handle)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverRfBatchAnalyze(hipsolverRfHandle_t handle)


.. py:function:: hipsolverRfBatchRefactor(handle)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverRfBatchRefactor(hipsolverRfHandle_t handle)


.. py:function:: hipsolverRfBatchResetValues(batchSize, n, nnzA, csrRowPtrA, csrColIndA, csrValA_array, P, Q, handle)

   (No short description, might be part of a group.)

   Args:
       batchSize (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nnzA (:py:obj:`~.int`):
           (undocumented)

       csrRowPtrA (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       csrColIndA (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       csrValA_array (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       P (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       Q (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverRfBatchResetValues(int batchSize, int n, int nnzA, int * csrRowPtrA, int * csrColIndA, double *[] csrValA_array, int * P, int * Q, hipsolverRfHandle_t handle)


.. py:function:: hipsolverRfBatchSolve(handle, P, Q, nrhs, Temp, ldt, XF_array, ldxf)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       P (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       Q (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       nrhs (:py:obj:`~.int`):
           (undocumented)

       Temp (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldt (:py:obj:`~.int`):
           (undocumented)

       XF_array (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ldxf (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverRfBatchSolve(hipsolverRfHandle_t handle, int * P, int * Q, int nrhs, double * Temp, int ldt, double *[] XF_array, int ldxf)


.. py:function:: hipsolverRfBatchZeroPivot(handle, position)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       position (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverRfBatchZeroPivot(hipsolverRfHandle_t handle, int * position)


.. py:function:: hipsolverSpCreate()

   (No short description, might be part of a group.)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)
       * handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSpCreate(hipsolverSpHandle_t * handle)


.. py:function:: hipsolverSpDestroy(handle)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSpDestroy(hipsolverSpHandle_t handle)


.. py:function:: hipsolverSpSetStream(handle, streamId)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       streamId (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSpSetStream(hipsolverSpHandle_t handle, hipStream_t streamId)


.. py:function:: hipsolverSpScsrlsvchol(handle, n, nnzA, descrA, csrVal, csrRowPtr, csrColInd, b, tolerance, reorder, x, singularity)

   // linear solver based on Cholesky

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nnzA (:py:obj:`~.int`):
           (undocumented)

       descrA (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       csrVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       csrRowPtr (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       csrColInd (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       b (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tolerance (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       reorder (:py:obj:`~.int`):
           (undocumented)

       x (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       singularity (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSpScsrlsvchol(hipsolverSpHandle_t handle, int n, int nnzA, const hipsparseMatDescr_t descrA, const float * csrVal, const int * csrRowPtr, const int * csrColInd, const float * b, float tolerance, int reorder, float * x, int * singularity)


.. py:function:: hipsolverSpDcsrlsvchol(handle, n, nnzA, descrA, csrVal, csrRowPtr, csrColInd, b, tolerance, reorder, x, singularity)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nnzA (:py:obj:`~.int`):
           (undocumented)

       descrA (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       csrVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       csrRowPtr (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       csrColInd (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       b (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tolerance (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       reorder (:py:obj:`~.int`):
           (undocumented)

       x (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       singularity (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSpDcsrlsvchol(hipsolverSpHandle_t handle, int n, int nnzA, const hipsparseMatDescr_t descrA, const double * csrVal, const int * csrRowPtr, const int * csrColInd, const double * b, double tolerance, int reorder, double * x, int * singularity)


.. py:function:: hipsolverSpScsrlsvcholHost(handle, n, nnzA, descrA, csrVal, csrRowPtr, csrColInd, b, tolerance, reorder, x, singularity)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nnzA (:py:obj:`~.int`):
           (undocumented)

       descrA (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       csrVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       csrRowPtr (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       csrColInd (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       b (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tolerance (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       reorder (:py:obj:`~.int`):
           (undocumented)

       x (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       singularity (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSpScsrlsvcholHost(hipsolverSpHandle_t handle, int n, int nnzA, const hipsparseMatDescr_t descrA, const float * csrVal, const int * csrRowPtr, const int * csrColInd, const float * b, float tolerance, int reorder, float * x, int * singularity)


.. py:function:: hipsolverSpDcsrlsvcholHost(handle, n, nnzA, descrA, csrVal, csrRowPtr, csrColInd, b, tolerance, reorder, x, singularity)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nnzA (:py:obj:`~.int`):
           (undocumented)

       descrA (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       csrVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       csrRowPtr (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       csrColInd (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       b (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tolerance (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       reorder (:py:obj:`~.int`):
           (undocumented)

       x (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       singularity (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSpDcsrlsvcholHost(hipsolverSpHandle_t handle, int n, int nnzA, const hipsparseMatDescr_t descrA, const double * csrVal, const int * csrRowPtr, const int * csrColInd, const double * b, double tolerance, int reorder, double * x, int * singularity)


.. py:function:: hipsolverSpScsrlsvqr(handle, n, nnz, descrA, csrVal, csrRowPts, csrColInd, b, tolerance, reorder, x, singularity)

   // linear solver based on QR

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nnz (:py:obj:`~.int`):
           (undocumented)

       descrA (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       csrVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       csrRowPts (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       csrColInd (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       b (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tolerance (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       reorder (:py:obj:`~.int`):
           (undocumented)

       x (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       singularity (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSpScsrlsvqr(hipsolverSpHandle_t handle, int n, int nnz, const hipsparseMatDescr_t descrA, const float * csrVal, const int * csrRowPts, const int * csrColInd, const float * b, double tolerance, int reorder, float * x, int * singularity)


.. py:function:: hipsolverSpDcsrlsvqr(handle, n, nnz, descrA, csrVal, csrRowPts, csrColInd, b, tolerance, reorder, x, singularity)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nnz (:py:obj:`~.int`):
           (undocumented)

       descrA (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       csrVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       csrRowPts (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       csrColInd (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       b (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tolerance (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       reorder (:py:obj:`~.int`):
           (undocumented)

       x (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       singularity (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSpDcsrlsvqr(hipsolverSpHandle_t handle, int n, int nnz, const hipsparseMatDescr_t descrA, const double * csrVal, const int * csrRowPts, const int * csrColInd, const double * b, double tolerance, int reorder, double * x, int * singularity)


.. py:function:: hipsolverSpCcsrlsvqr(handle, n, nnz, descrA, csrVal, csrRowPts, csrColInd, b, tolerance, reorder, x, singularity)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nnz (:py:obj:`~.int`):
           (undocumented)

       descrA (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       csrVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       csrRowPts (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       csrColInd (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       b (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tolerance (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       reorder (:py:obj:`~.int`):
           (undocumented)

       x (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       singularity (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSpCcsrlsvqr(hipsolverSpHandle_t handle, int n, int nnz, const hipsparseMatDescr_t descrA, const hipFloatComplex * csrVal, const int * csrRowPts, const int * csrColInd, const hipFloatComplex * b, double tolerance, int reorder, hipFloatComplex * x, int * singularity)


.. py:function:: hipsolverSpZcsrlsvqr(handle, n, nnz, descrA, csrVal, csrRowPts, csrColInd, b, tolerance, reorder, x, singularity)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       n (:py:obj:`~.int`):
           (undocumented)

       nnz (:py:obj:`~.int`):
           (undocumented)

       descrA (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       csrVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       csrRowPts (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       csrColInd (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

       b (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       tolerance (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

       reorder (:py:obj:`~.int`):
           (undocumented)

       x (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       singularity (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsolverStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsolverStatus_t hipsolverSpZcsrlsvqr(hipsolverSpHandle_t handle, int n, int nnz, const hipsparseMatDescr_t descrA, const hipDoubleComplex * csrVal, const int * csrRowPts, const int * csrColInd, const hipDoubleComplex * b, double tolerance, int reorder, hipDoubleComplex * x, int * singularity)


