import enum
import os

cuda2hip={
    "cuGetErrorName": "hipDrvGetErrorName",
    "cuGetErrorString": "hipDrvGetErrorString",
    "cublasAxpyEx": "hipblasAxpyEx",
    "cublasCaxpy": "hipblasCaxpy",
    "cublasCaxpy_v2": "hipblasCaxpy",
    "cublasCcopy": "hipblasCcopy",
    "cublasCcopy_v2": "hipblasCcopy",
    "cublasCdgmm": "hipblasCdgmm",
    "cublasCdotc": "hipblasCdotc",
    "cublasCdotc_v2": "hipblasCdotc",
    "cublasCdotu": "hipblasCdotu",
    "cublasCdotu_v2": "hipblasCdotu",
    "cublasCgbmv": "hipblasCgbmv",
    "cublasCgbmv_v2": "hipblasCgbmv",
    "cublasCgeam": "hipblasCgeam",
    "cublasCgemm": "hipblasCgemm",
    "cublasCgemmBatched": "hipblasCgemmBatched",
    "cublasCgemmStridedBatched": "hipblasCgemmStridedBatched",
    "cublasCgemm_v2": "hipblasCgemm",
    "cublasCgemv": "hipblasCgemv",
    "cublasCgemv_v2": "hipblasCgemv",
    "cublasCgerc": "hipblasCgerc",
    "cublasCgerc_v2": "hipblasCgerc",
    "cublasCgeru": "hipblasCgeru",
    "cublasCgeru_v2": "hipblasCgeru",
    "cublasChbmv": "hipblasChbmv",
    "cublasChbmv_v2": "hipblasChbmv",
    "cublasChemm": "hipblasChemm",
    "cublasChemm_v2": "hipblasChemm",
    "cublasChemv": "hipblasChemv",
    "cublasChemv_v2": "hipblasChemv",
    "cublasCher": "hipblasCher",
    "cublasCher2": "hipblasCher2",
    "cublasCher2_v2": "hipblasCher2",
    "cublasCher2k": "hipblasCher2k",
    "cublasCher2k_v2": "hipblasCher2k",
    "cublasCher_v2": "hipblasCher",
    "cublasCherk": "hipblasCherk",
    "cublasCherk_v2": "hipblasCherk",
    "cublasCherkx": "hipblasCherkx",
    "cublasChpmv": "hipblasChpmv",
    "cublasChpmv_v2": "hipblasChpmv",
    "cublasChpr": "hipblasChpr",
    "cublasChpr2": "hipblasChpr2",
    "cublasChpr2_v2": "hipblasChpr2",
    "cublasChpr_v2": "hipblasChpr",
    "cublasCreate": "hipblasCreate",
    "cublasCreate_v2": "hipblasCreate",
    "cublasCrot": "hipblasCrot",
    "cublasCrot_v2": "hipblasCrot",
    "cublasCrotg": "hipblasCrotg",
    "cublasCrotg_v2": "hipblasCrotg",
    "cublasCscal": "hipblasCscal",
    "cublasCscal_v2": "hipblasCscal",
    "cublasCsrot": "hipblasCsrot",
    "cublasCsrot_v2": "hipblasCsrot",
    "cublasCsscal": "hipblasCsscal",
    "cublasCsscal_v2": "hipblasCsscal",
    "cublasCswap": "hipblasCswap",
    "cublasCswap_v2": "hipblasCswap",
    "cublasCsymm": "hipblasCsymm",
    "cublasCsymm_v2": "hipblasCsymm",
    "cublasCsymv": "hipblasCsymv",
    "cublasCsymv_v2": "hipblasCsymv",
    "cublasCsyr": "hipblasCsyr",
    "cublasCsyr2": "hipblasCsyr2",
    "cublasCsyr2_v2": "hipblasCsyr2",
    "cublasCsyr2k": "hipblasCsyr2k",
    "cublasCsyr2k_v2": "hipblasCsyr2k",
    "cublasCsyr_v2": "hipblasCsyr",
    "cublasCsyrk": "hipblasCsyrk",
    "cublasCsyrk_v2": "hipblasCsyrk",
    "cublasCsyrkx": "hipblasCsyrkx",
    "cublasCtbmv": "hipblasCtbmv",
    "cublasCtbmv_v2": "hipblasCtbmv",
    "cublasCtbsv": "hipblasCtbsv",
    "cublasCtbsv_v2": "hipblasCtbsv",
    "cublasCtpmv": "hipblasCtpmv",
    "cublasCtpmv_v2": "hipblasCtpmv",
    "cublasCtpsv": "hipblasCtpsv",
    "cublasCtpsv_v2": "hipblasCtpsv",
    "cublasCtrmm": "rocblas_ctrmm_outofplace",
    "cublasCtrmm_v2": "rocblas_ctrmm_outofplace",
    "cublasCtrmv": "hipblasCtrmv",
    "cublasCtrmv_v2": "hipblasCtrmv",
    "cublasCtrsm": "hipblasCtrsm",
    "cublasCtrsmBatched": "hipblasCtrsmBatched",
    "cublasCtrsm_v2": "hipblasCtrsm",
    "cublasCtrsv": "hipblasCtrsv",
    "cublasCtrsv_v2": "hipblasCtrsv",
    "cublasDasum": "hipblasDasum",
    "cublasDasum_v2": "hipblasDasum",
    "cublasDaxpy": "hipblasDaxpy",
    "cublasDaxpy_v2": "hipblasDaxpy",
    "cublasDcopy": "hipblasDcopy",
    "cublasDcopy_v2": "hipblasDcopy",
    "cublasDdgmm": "hipblasDdgmm",
    "cublasDdot": "hipblasDdot",
    "cublasDdot_v2": "hipblasDdot",
    "cublasDestroy": "hipblasDestroy",
    "cublasDestroy_v2": "hipblasDestroy",
    "cublasDgbmv": "hipblasDgbmv",
    "cublasDgbmv_v2": "hipblasDgbmv",
    "cublasDgeam": "hipblasDgeam",
    "cublasDgemm": "hipblasDgemm",
    "cublasDgemmBatched": "hipblasDgemmBatched",
    "cublasDgemmStridedBatched": "hipblasDgemmStridedBatched",
    "cublasDgemm_v2": "hipblasDgemm",
    "cublasDgemv": "hipblasDgemv",
    "cublasDgemv_v2": "hipblasDgemv",
    "cublasDger": "hipblasDger",
    "cublasDger_v2": "hipblasDger",
    "cublasDnrm2": "hipblasDnrm2",
    "cublasDnrm2_v2": "hipblasDnrm2",
    "cublasDotEx": "hipblasDotEx",
    "cublasDotcEx": "hipblasDotcEx",
    "cublasDrot": "hipblasDrot",
    "cublasDrot_v2": "hipblasDrot",
    "cublasDrotg": "hipblasDrotg",
    "cublasDrotg_v2": "hipblasDrotg",
    "cublasDrotm": "hipblasDrotm",
    "cublasDrotm_v2": "hipblasDrotm",
    "cublasDrotmg": "hipblasDrotmg",
    "cublasDrotmg_v2": "hipblasDrotmg",
    "cublasDsbmv": "hipblasDsbmv",
    "cublasDsbmv_v2": "hipblasDsbmv",
    "cublasDscal": "hipblasDscal",
    "cublasDscal_v2": "hipblasDscal",
    "cublasDspmv": "hipblasDspmv",
    "cublasDspmv_v2": "hipblasDspmv",
    "cublasDspr": "hipblasDspr",
    "cublasDspr2": "hipblasDspr2",
    "cublasDspr2_v2": "hipblasDspr2",
    "cublasDspr_v2": "hipblasDspr",
    "cublasDswap": "hipblasDswap",
    "cublasDswap_v2": "hipblasDswap",
    "cublasDsymm": "hipblasDsymm",
    "cublasDsymm_v2": "hipblasDsymm",
    "cublasDsymv": "hipblasDsymv",
    "cublasDsymv_v2": "hipblasDsymv",
    "cublasDsyr": "hipblasDsyr",
    "cublasDsyr2": "hipblasDsyr2",
    "cublasDsyr2_v2": "hipblasDsyr2",
    "cublasDsyr2k": "hipblasDsyr2k",
    "cublasDsyr2k_v2": "hipblasDsyr2k",
    "cublasDsyr_v2": "hipblasDsyr",
    "cublasDsyrk": "hipblasDsyrk",
    "cublasDsyrk_v2": "hipblasDsyrk",
    "cublasDsyrkx": "hipblasDsyrkx",
    "cublasDtbmv": "hipblasDtbmv",
    "cublasDtbmv_v2": "hipblasDtbmv",
    "cublasDtbsv": "hipblasDtbsv",
    "cublasDtbsv_v2": "hipblasDtbsv",
    "cublasDtpmv": "hipblasDtpmv",
    "cublasDtpmv_v2": "hipblasDtpmv",
    "cublasDtpsv": "hipblasDtpsv",
    "cublasDtpsv_v2": "hipblasDtpsv",
    "cublasDtrmm": "rocblas_dtrmm_outofplace",
    "cublasDtrmm_v2": "rocblas_dtrmm_outofplace",
    "cublasDtrmv": "hipblasDtrmv",
    "cublasDtrmv_v2": "hipblasDtrmv",
    "cublasDtrsm": "hipblasDtrsm",
    "cublasDtrsmBatched": "hipblasDtrsmBatched",
    "cublasDtrsm_v2": "hipblasDtrsm",
    "cublasDtrsv": "hipblasDtrsv",
    "cublasDtrsv_v2": "hipblasDtrsv",
    "cublasDzasum": "hipblasDzasum",
    "cublasDzasum_v2": "hipblasDzasum",
    "cublasDznrm2": "hipblasDznrm2",
    "cublasDznrm2_v2": "hipblasDznrm2",
    "cublasGemmBatchedEx": "hipblasGemmBatchedEx",
    "cublasGemmEx": "hipblasGemmEx",
    "cublasGemmStridedBatchedEx": "hipblasGemmStridedBatchedEx",
    "cublasGetAtomicsMode": "hipblasGetAtomicsMode",
    "cublasGetMatrix": "hipblasGetMatrix",
    "cublasGetMatrixAsync": "hipblasGetMatrixAsync",
    "cublasGetPointerMode": "hipblasGetPointerMode",
    "cublasGetPointerMode_v2": "hipblasGetPointerMode",
    "cublasGetStatusString": "rocblas_status_to_string",
    "cublasGetStream": "hipblasGetStream",
    "cublasGetStream_v2": "hipblasGetStream",
    "cublasGetVector": "hipblasGetVector",
    "cublasGetVectorAsync": "hipblasGetVectorAsync",
    "cublasHgemm": "hipblasHgemm",
    "cublasHgemmBatched": "hipblasHgemmBatched",
    "cublasHgemmStridedBatched": "hipblasHgemmStridedBatched",
    "cublasIcamax": "hipblasIcamax",
    "cublasIcamax_v2": "hipblasIcamax",
    "cublasIcamin": "hipblasIcamin",
    "cublasIcamin_v2": "hipblasIcamin",
    "cublasIdamax": "hipblasIdamax",
    "cublasIdamax_v2": "hipblasIdamax",
    "cublasIdamin": "hipblasIdamin",
    "cublasIdamin_v2": "hipblasIdamin",
    "cublasInit": "rocblas_initialize",
    "cublasIsamax": "hipblasIsamax",
    "cublasIsamax_v2": "hipblasIsamax",
    "cublasIsamin": "hipblasIsamin",
    "cublasIsamin_v2": "hipblasIsamin",
    "cublasIzamax": "hipblasIzamax",
    "cublasIzamax_v2": "hipblasIzamax",
    "cublasIzamin": "hipblasIzamin",
    "cublasIzamin_v2": "hipblasIzamin",
    "cublasNrm2Ex": "hipblasNrm2Ex",
    "cublasRotEx": "hipblasRotEx",
    "cublasSasum": "hipblasSasum",
    "cublasSasum_v2": "hipblasSasum",
    "cublasSaxpy": "hipblasSaxpy",
    "cublasSaxpy_v2": "hipblasSaxpy",
    "cublasScalEx": "hipblasScalEx",
    "cublasScasum": "hipblasScasum",
    "cublasScasum_v2": "hipblasScasum",
    "cublasScnrm2": "hipblasScnrm2",
    "cublasScnrm2_v2": "hipblasScnrm2",
    "cublasScopy": "hipblasScopy",
    "cublasScopy_v2": "hipblasScopy",
    "cublasSdgmm": "hipblasSdgmm",
    "cublasSdot": "hipblasSdot",
    "cublasSdot_v2": "hipblasSdot",
    "cublasSetAtomicsMode": "hipblasSetAtomicsMode",
    "cublasSetMatrix": "hipblasSetMatrix",
    "cublasSetMatrixAsync": "hipblasSetMatrixAsync",
    "cublasSetPointerMode": "hipblasSetPointerMode",
    "cublasSetPointerMode_v2": "hipblasSetPointerMode",
    "cublasSetStream": "hipblasSetStream",
    "cublasSetStream_v2": "hipblasSetStream",
    "cublasSetVector": "hipblasSetVector",
    "cublasSetVectorAsync": "hipblasSetVectorAsync",
    "cublasSgbmv": "hipblasSgbmv",
    "cublasSgbmv_v2": "hipblasSgbmv",
    "cublasSgeam": "hipblasSgeam",
    "cublasSgemm": "hipblasSgemm",
    "cublasSgemmBatched": "hipblasSgemmBatched",
    "cublasSgemmStridedBatched": "hipblasSgemmStridedBatched",
    "cublasSgemm_v2": "hipblasSgemm",
    "cublasSgemv": "hipblasSgemv",
    "cublasSgemv_v2": "hipblasSgemv",
    "cublasSger": "hipblasSger",
    "cublasSger_v2": "hipblasSger",
    "cublasSnrm2": "hipblasSnrm2",
    "cublasSnrm2_v2": "hipblasSnrm2",
    "cublasSrot": "hipblasSrot",
    "cublasSrot_v2": "hipblasSrot",
    "cublasSrotg": "hipblasSrotg",
    "cublasSrotg_v2": "hipblasSrotg",
    "cublasSrotm": "hipblasSrotm",
    "cublasSrotm_v2": "hipblasSrotm",
    "cublasSrotmg": "hipblasSrotmg",
    "cublasSrotmg_v2": "hipblasSrotmg",
    "cublasSsbmv": "hipblasSsbmv",
    "cublasSsbmv_v2": "hipblasSsbmv",
    "cublasSscal": "hipblasSscal",
    "cublasSscal_v2": "hipblasSscal",
    "cublasSspmv": "hipblasSspmv",
    "cublasSspmv_v2": "hipblasSspmv",
    "cublasSspr": "hipblasSspr",
    "cublasSspr2": "hipblasSspr2",
    "cublasSspr2_v2": "hipblasSspr2",
    "cublasSspr_v2": "hipblasSspr",
    "cublasSswap": "hipblasSswap",
    "cublasSswap_v2": "hipblasSswap",
    "cublasSsymm": "hipblasSsymm",
    "cublasSsymm_v2": "hipblasSsymm",
    "cublasSsymv": "hipblasSsymv",
    "cublasSsymv_v2": "hipblasSsymv",
    "cublasSsyr": "hipblasSsyr",
    "cublasSsyr2": "hipblasSsyr2",
    "cublasSsyr2_v2": "hipblasSsyr2",
    "cublasSsyr2k": "hipblasSsyr2k",
    "cublasSsyr2k_v2": "hipblasSsyr2k",
    "cublasSsyr_v2": "hipblasSsyr",
    "cublasSsyrk": "hipblasSsyrk",
    "cublasSsyrk_v2": "hipblasSsyrk",
    "cublasSsyrkx": "hipblasSsyrkx",
    "cublasStbmv": "hipblasStbmv",
    "cublasStbmv_v2": "hipblasStbmv",
    "cublasStbsv": "hipblasStbsv",
    "cublasStbsv_v2": "hipblasStbsv",
    "cublasStpmv": "hipblasStpmv",
    "cublasStpmv_v2": "hipblasStpmv",
    "cublasStpsv": "hipblasStpsv",
    "cublasStpsv_v2": "hipblasStpsv",
    "cublasStrmm": "rocblas_strmm_outofplace",
    "cublasStrmm_v2": "rocblas_strmm_outofplace",
    "cublasStrmv": "hipblasStrmv",
    "cublasStrmv_v2": "hipblasStrmv",
    "cublasStrsm": "hipblasStrsm",
    "cublasStrsmBatched": "hipblasStrsmBatched",
    "cublasStrsm_v2": "hipblasStrsm",
    "cublasStrsv": "hipblasStrsv",
    "cublasStrsv_v2": "hipblasStrsv",
    "cublasZaxpy": "hipblasZaxpy",
    "cublasZaxpy_v2": "hipblasZaxpy",
    "cublasZcopy": "hipblasZcopy",
    "cublasZcopy_v2": "hipblasZcopy",
    "cublasZdgmm": "hipblasZdgmm",
    "cublasZdotc": "hipblasZdotc",
    "cublasZdotc_v2": "hipblasZdotc",
    "cublasZdotu": "hipblasZdotu",
    "cublasZdotu_v2": "hipblasZdotu",
    "cublasZdrot": "hipblasZdrot",
    "cublasZdrot_v2": "hipblasZdrot",
    "cublasZdscal": "hipblasZdscal",
    "cublasZdscal_v2": "hipblasZdscal",
    "cublasZgbmv": "hipblasZgbmv",
    "cublasZgbmv_v2": "hipblasZgbmv",
    "cublasZgeam": "hipblasZgeam",
    "cublasZgemm": "hipblasZgemm",
    "cublasZgemmBatched": "hipblasZgemmBatched",
    "cublasZgemmStridedBatched": "hipblasZgemmStridedBatched",
    "cublasZgemm_v2": "hipblasZgemm",
    "cublasZgemv": "hipblasZgemv",
    "cublasZgemv_v2": "hipblasZgemv",
    "cublasZgerc": "hipblasZgerc",
    "cublasZgerc_v2": "hipblasZgerc",
    "cublasZgeru": "hipblasZgeru",
    "cublasZgeru_v2": "hipblasZgeru",
    "cublasZhbmv": "hipblasZhbmv",
    "cublasZhbmv_v2": "hipblasZhbmv",
    "cublasZhemm": "hipblasZhemm",
    "cublasZhemm_v2": "hipblasZhemm",
    "cublasZhemv": "hipblasZhemv",
    "cublasZhemv_v2": "hipblasZhemv",
    "cublasZher": "hipblasZher",
    "cublasZher2": "hipblasZher2",
    "cublasZher2_v2": "hipblasZher2",
    "cublasZher2k": "hipblasZher2k",
    "cublasZher2k_v2": "hipblasZher2k",
    "cublasZher_v2": "hipblasZher",
    "cublasZherk": "hipblasZherk",
    "cublasZherk_v2": "hipblasZherk",
    "cublasZherkx": "hipblasZherkx",
    "cublasZhpmv": "hipblasZhpmv",
    "cublasZhpmv_v2": "hipblasZhpmv",
    "cublasZhpr": "hipblasZhpr",
    "cublasZhpr2": "hipblasZhpr2",
    "cublasZhpr2_v2": "hipblasZhpr2",
    "cublasZhpr_v2": "hipblasZhpr",
    "cublasZrot": "hipblasZrot",
    "cublasZrot_v2": "hipblasZrot",
    "cublasZrotg": "hipblasZrotg",
    "cublasZrotg_v2": "hipblasZrotg",
    "cublasZscal": "hipblasZscal",
    "cublasZscal_v2": "hipblasZscal",
    "cublasZswap": "hipblasZswap",
    "cublasZswap_v2": "hipblasZswap",
    "cublasZsymm": "hipblasZsymm",
    "cublasZsymm_v2": "hipblasZsymm",
    "cublasZsymv": "hipblasZsymv",
    "cublasZsymv_v2": "hipblasZsymv",
    "cublasZsyr": "hipblasZsyr",
    "cublasZsyr2": "hipblasZsyr2",
    "cublasZsyr2_v2": "hipblasZsyr2",
    "cublasZsyr2k": "hipblasZsyr2k",
    "cublasZsyr2k_v2": "hipblasZsyr2k",
    "cublasZsyr_v2": "hipblasZsyr",
    "cublasZsyrk": "hipblasZsyrk",
    "cublasZsyrk_v2": "hipblasZsyrk",
    "cublasZsyrkx": "hipblasZsyrkx",
    "cublasZtbmv": "hipblasZtbmv",
    "cublasZtbmv_v2": "hipblasZtbmv",
    "cublasZtbsv": "hipblasZtbsv",
    "cublasZtbsv_v2": "hipblasZtbsv",
    "cublasZtpmv": "hipblasZtpmv",
    "cublasZtpmv_v2": "hipblasZtpmv",
    "cublasZtpsv": "hipblasZtpsv",
    "cublasZtpsv_v2": "hipblasZtpsv",
    "cublasZtrmm": "rocblas_ztrmm_outofplace",
    "cublasZtrmm_v2": "rocblas_ztrmm_outofplace",
    "cublasZtrmv": "hipblasZtrmv",
    "cublasZtrmv_v2": "hipblasZtrmv",
    "cublasZtrsm": "hipblasZtrsm",
    "cublasZtrsmBatched": "hipblasZtrsmBatched",
    "cublasZtrsm_v2": "hipblasZtrsm",
    "cublasZtrsv": "hipblasZtrsv",
    "cublasZtrsv_v2": "hipblasZtrsv",
    "cublasAtomicsMode_t": "hipblasAtomicsMode_t",
    "cublasContext": "_rocblas_handle",
    "cublasDataType_t": "hipblasDatatype_t",
    "cublasDiagType_t": "hipblasDiagType_t",
    "cublasFillMode_t": "hipblasFillMode_t",
    "cublasGemmAlgo_t": "hipblasGemmAlgo_t",
    "cublasHandle_t": "hipblasHandle_t",
    "cublasOperation_t": "hipblasOperation_t",
    "cublasPointerMode_t": "hipblasPointerMode_t",
    "cublasSideMode_t": "hipblasSideMode_t",
    "cublasStatus": "hipblasStatus_t",
    "cublasStatus_t": "hipblasStatus_t",
    "cudaDataType": "hipblasDatatype_t",
    "cudaDataType_t": "hipblasDatatype_t",
    "CUBLAS_ATOMICS_ALLOWED": "HIPBLAS_ATOMICS_ALLOWED",
    "CUBLAS_ATOMICS_NOT_ALLOWED": "HIPBLAS_ATOMICS_NOT_ALLOWED",
    "CUBLAS_DIAG_NON_UNIT": "HIPBLAS_DIAG_NON_UNIT",
    "CUBLAS_DIAG_UNIT": "HIPBLAS_DIAG_UNIT",
    "CUBLAS_FILL_MODE_FULL": "HIPBLAS_FILL_MODE_FULL",
    "CUBLAS_FILL_MODE_LOWER": "HIPBLAS_FILL_MODE_LOWER",
    "CUBLAS_FILL_MODE_UPPER": "HIPBLAS_FILL_MODE_UPPER",
    "CUBLAS_GEMM_DEFAULT": "HIPBLAS_GEMM_DEFAULT",
    "CUBLAS_GEMM_DFALT": "HIPBLAS_GEMM_DEFAULT",
    "CUBLAS_OP_C": "HIPBLAS_OP_C",
    "CUBLAS_OP_HERMITAN": "HIPBLAS_OP_C",
    "CUBLAS_OP_N": "HIPBLAS_OP_N",
    "CUBLAS_OP_T": "HIPBLAS_OP_T",
    "CUBLAS_POINTER_MODE_DEVICE": "HIPBLAS_POINTER_MODE_DEVICE",
    "CUBLAS_POINTER_MODE_HOST": "HIPBLAS_POINTER_MODE_HOST",
    "CUBLAS_SIDE_LEFT": "HIPBLAS_SIDE_LEFT",
    "CUBLAS_SIDE_RIGHT": "HIPBLAS_SIDE_RIGHT",
    "CUBLAS_STATUS_ALLOC_FAILED": "HIPBLAS_STATUS_ALLOC_FAILED",
    "CUBLAS_STATUS_ARCH_MISMATCH": "HIPBLAS_STATUS_ARCH_MISMATCH",
    "CUBLAS_STATUS_EXECUTION_FAILED": "HIPBLAS_STATUS_EXECUTION_FAILED",
    "CUBLAS_STATUS_INTERNAL_ERROR": "HIPBLAS_STATUS_INTERNAL_ERROR",
    "CUBLAS_STATUS_INVALID_VALUE": "HIPBLAS_STATUS_INVALID_VALUE",
    "CUBLAS_STATUS_MAPPING_ERROR": "HIPBLAS_STATUS_MAPPING_ERROR",
    "CUBLAS_STATUS_NOT_INITIALIZED": "HIPBLAS_STATUS_NOT_INITIALIZED",
    "CUBLAS_STATUS_NOT_SUPPORTED": "HIPBLAS_STATUS_NOT_SUPPORTED",
    "CUBLAS_STATUS_SUCCESS": "HIPBLAS_STATUS_SUCCESS",
    "CUDA_C_16BF": "HIPBLAS_C_16B",
    "CUDA_C_16F": "HIPBLAS_C_16F",
    "CUDA_C_32F": "HIPBLAS_C_32F",
    "CUDA_C_32I": "HIPBLAS_C_32I",
    "CUDA_C_32U": "HIPBLAS_C_32U",
    "CUDA_C_64F": "HIPBLAS_C_64F",
    "CUDA_C_8I": "HIPBLAS_C_8I",
    "CUDA_C_8U": "HIPBLAS_C_8U",
    "CUDA_R_16BF": "HIPBLAS_R_16B",
    "CUDA_R_16F": "HIPBLAS_R_16F",
    "CUDA_R_32F": "HIPBLAS_R_32F",
    "CUDA_R_32I": "HIPBLAS_R_32I",
    "CUDA_R_32U": "HIPBLAS_R_32U",
    "CUDA_R_64F": "HIPBLAS_R_64F",
    "CUDA_R_8I": "HIPBLAS_R_8I",
    "CUDA_R_8U": "HIPBLAS_R_8U",
    "cudaGetErrorName": "hipGetErrorName",
    "cudaGetErrorString": "hipGetErrorString",
    "cudaGetLastError": "hipGetLastError",
    "cudaPeekAtLastError": "hipPeekAtLastError",
    "cuInit": "hipInit",
    "cuDriverGetVersion": "hipDriverGetVersion",
    "cudaDriverGetVersion": "hipDriverGetVersion",
    "cudaRuntimeGetVersion": "hipRuntimeGetVersion",
    "cuDeviceComputeCapability": "hipDeviceComputeCapability",
    "cuDeviceGet": "hipDeviceGet",
    "cuDeviceGetAttribute": "hipDeviceGetAttribute",
    "cuDeviceGetCount": "hipGetDeviceCount",
    "cuDeviceGetDefaultMemPool": "hipDeviceGetDefaultMemPool",
    "cuDeviceGetMemPool": "hipDeviceGetMemPool",
    "cuDeviceGetName": "hipDeviceGetName",
    "cuDeviceGetUuid": "hipDeviceGetUuid",
    "cuDeviceGetUuid_v2": "hipDeviceGetUuid",
    "cuDeviceSetMemPool": "hipDeviceSetMemPool",
    "cuDeviceTotalMem": "hipDeviceTotalMem",
    "cuDeviceTotalMem_v2": "hipDeviceTotalMem",
    "cudaChooseDevice": "hipChooseDevice",
    "cudaDeviceGetAttribute": "hipDeviceGetAttribute",
    "cudaDeviceGetByPCIBusId": "hipDeviceGetByPCIBusId",
    "cudaDeviceGetCacheConfig": "hipDeviceGetCacheConfig",
    "cudaDeviceGetDefaultMemPool": "hipDeviceGetDefaultMemPool",
    "cudaDeviceGetLimit": "hipDeviceGetLimit",
    "cudaDeviceGetMemPool": "hipDeviceGetMemPool",
    "cudaDeviceGetP2PAttribute": "hipDeviceGetP2PAttribute",
    "cudaDeviceGetPCIBusId": "hipDeviceGetPCIBusId",
    "cudaDeviceGetSharedMemConfig": "hipDeviceGetSharedMemConfig",
    "cudaDeviceGetStreamPriorityRange": "hipDeviceGetStreamPriorityRange",
    "cudaDeviceReset": "hipDeviceReset",
    "cudaDeviceSetCacheConfig": "hipDeviceSetCacheConfig",
    "cudaDeviceSetLimit": "hipDeviceSetLimit",
    "cudaDeviceSetMemPool": "hipDeviceSetMemPool",
    "cudaDeviceSetSharedMemConfig": "hipDeviceSetSharedMemConfig",
    "cudaDeviceSynchronize": "hipDeviceSynchronize",
    "cudaFuncSetCacheConfig": "hipFuncSetCacheConfig",
    "cudaGetDevice": "hipGetDevice",
    "cudaGetDeviceCount": "hipGetDeviceCount",
    "cudaGetDeviceFlags": "hipGetDeviceFlags",
    "cudaGetDeviceProperties": "hipGetDeviceProperties",
    "cudaIpcCloseMemHandle": "hipIpcCloseMemHandle",
    "cudaIpcGetEventHandle": "hipIpcGetEventHandle",
    "cudaIpcGetMemHandle": "hipIpcGetMemHandle",
    "cudaIpcOpenEventHandle": "hipIpcOpenEventHandle",
    "cudaIpcOpenMemHandle": "hipIpcOpenMemHandle",
    "cudaSetDevice": "hipSetDevice",
    "cudaSetDeviceFlags": "hipSetDeviceFlags",
    "cuCtxCreate": "hipCtxCreate",
    "cuCtxCreate_v2": "hipCtxCreate",
    "cuCtxDestroy": "hipCtxDestroy",
    "cuCtxDestroy_v2": "hipCtxDestroy",
    "cuCtxGetApiVersion": "hipCtxGetApiVersion",
    "cuCtxGetCacheConfig": "hipCtxGetCacheConfig",
    "cuCtxGetCurrent": "hipCtxGetCurrent",
    "cuCtxGetDevice": "hipCtxGetDevice",
    "cuCtxGetFlags": "hipCtxGetFlags",
    "cuCtxGetLimit": "hipDeviceGetLimit",
    "cuCtxGetSharedMemConfig": "hipCtxGetSharedMemConfig",
    "cuCtxGetStreamPriorityRange": "hipDeviceGetStreamPriorityRange",
    "cuCtxPopCurrent": "hipCtxPopCurrent",
    "cuCtxPopCurrent_v2": "hipCtxPopCurrent",
    "cuCtxPushCurrent": "hipCtxPushCurrent",
    "cuCtxPushCurrent_v2": "hipCtxPushCurrent",
    "cuCtxSetCacheConfig": "hipCtxSetCacheConfig",
    "cuCtxSetCurrent": "hipCtxSetCurrent",
    "cuCtxSetLimit": "hipDeviceSetLimit",
    "cuCtxSetSharedMemConfig": "hipCtxSetSharedMemConfig",
    "cuCtxSynchronize": "hipCtxSynchronize",
    "cuDevicePrimaryCtxGetState": "hipDevicePrimaryCtxGetState",
    "cuDevicePrimaryCtxRelease": "hipDevicePrimaryCtxRelease",
    "cuDevicePrimaryCtxRelease_v2": "hipDevicePrimaryCtxRelease",
    "cuDevicePrimaryCtxReset": "hipDevicePrimaryCtxReset",
    "cuDevicePrimaryCtxReset_v2": "hipDevicePrimaryCtxReset",
    "cuDevicePrimaryCtxRetain": "hipDevicePrimaryCtxRetain",
    "cuDevicePrimaryCtxSetFlags": "hipDevicePrimaryCtxSetFlags",
    "cuDevicePrimaryCtxSetFlags_v2": "hipDevicePrimaryCtxSetFlags",
    "cuLinkAddData": "hiprtcLinkAddData",
    "cuLinkAddData_v2": "hiprtcLinkAddData",
    "cuLinkAddFile": "hiprtcLinkAddFile",
    "cuLinkAddFile_v2": "hiprtcLinkAddFile",
    "cuLinkComplete": "hiprtcLinkComplete",
    "cuLinkCreate": "hiprtcLinkCreate",
    "cuLinkCreate_v2": "hiprtcLinkCreate",
    "cuLinkDestroy": "hiprtcLinkDestroy",
    "cuModuleGetFunction": "hipModuleGetFunction",
    "cuModuleGetGlobal": "hipModuleGetGlobal",
    "cuModuleGetGlobal_v2": "hipModuleGetGlobal",
    "cuModuleGetTexRef": "hipModuleGetTexRef",
    "cuModuleLoad": "hipModuleLoad",
    "cuModuleLoadData": "hipModuleLoadData",
    "cuModuleLoadDataEx": "hipModuleLoadDataEx",
    "cuModuleUnload": "hipModuleUnload",
    "cuArray3DCreate": "hipArray3DCreate",
    "cuArray3DCreate_v2": "hipArray3DCreate",
    "cuArrayCreate": "hipArrayCreate",
    "cuArrayCreate_v2": "hipArrayCreate",
    "cuArrayDestroy": "hipArrayDestroy",
    "cuDeviceGetByPCIBusId": "hipDeviceGetByPCIBusId",
    "cuDeviceGetPCIBusId": "hipDeviceGetPCIBusId",
    "cuIpcCloseMemHandle": "hipIpcCloseMemHandle",
    "cuIpcGetEventHandle": "hipIpcGetEventHandle",
    "cuIpcGetMemHandle": "hipIpcGetMemHandle",
    "cuIpcOpenEventHandle": "hipIpcOpenEventHandle",
    "cuIpcOpenMemHandle": "hipIpcOpenMemHandle",
    "cuMemAlloc": "hipMalloc",
    "cuMemAllocHost": "hipMemAllocHost",
    "cuMemAllocHost_v2": "hipMemAllocHost",
    "cuMemAllocManaged": "hipMallocManaged",
    "cuMemAllocPitch": "hipMemAllocPitch",
    "cuMemAllocPitch_v2": "hipMemAllocPitch",
    "cuMemAlloc_v2": "hipMalloc",
    "cuMemFree": "hipFree",
    "cuMemFreeHost": "hipHostFree",
    "cuMemFree_v2": "hipFree",
    "cuMemGetAddressRange": "hipMemGetAddressRange",
    "cuMemGetAddressRange_v2": "hipMemGetAddressRange",
    "cuMemGetInfo": "hipMemGetInfo",
    "cuMemGetInfo_v2": "hipMemGetInfo",
    "cuMemHostAlloc": "hipHostAlloc",
    "cuMemHostGetDevicePointer": "hipHostGetDevicePointer",
    "cuMemHostGetDevicePointer_v2": "hipHostGetDevicePointer",
    "cuMemHostGetFlags": "hipHostGetFlags",
    "cuMemHostRegister": "hipHostRegister",
    "cuMemHostRegister_v2": "hipHostRegister",
    "cuMemHostUnregister": "hipHostUnregister",
    "cuMemcpy2D": "hipMemcpyParam2D",
    "cuMemcpy2DAsync": "hipMemcpyParam2DAsync",
    "cuMemcpy2DAsync_v2": "hipMemcpyParam2DAsync",
    "cuMemcpy2DUnaligned": "hipDrvMemcpy2DUnaligned",
    "cuMemcpy2DUnaligned_v2": "hipDrvMemcpy2DUnaligned",
    "cuMemcpy2D_v2": "hipMemcpyParam2D",
    "cuMemcpy3D": "hipDrvMemcpy3D",
    "cuMemcpy3DAsync": "hipDrvMemcpy3DAsync",
    "cuMemcpy3DAsync_v2": "hipDrvMemcpy3DAsync",
    "cuMemcpy3D_v2": "hipDrvMemcpy3D",
    "cuMemcpyAtoH": "hipMemcpyAtoH",
    "cuMemcpyAtoH_v2": "hipMemcpyAtoH",
    "cuMemcpyDtoD": "hipMemcpyDtoD",
    "cuMemcpyDtoDAsync": "hipMemcpyDtoDAsync",
    "cuMemcpyDtoDAsync_v2": "hipMemcpyDtoDAsync",
    "cuMemcpyDtoD_v2": "hipMemcpyDtoD",
    "cuMemcpyDtoH": "hipMemcpyDtoH",
    "cuMemcpyDtoHAsync": "hipMemcpyDtoHAsync",
    "cuMemcpyDtoHAsync_v2": "hipMemcpyDtoHAsync",
    "cuMemcpyDtoH_v2": "hipMemcpyDtoH",
    "cuMemcpyHtoA": "hipMemcpyHtoA",
    "cuMemcpyHtoA_v2": "hipMemcpyHtoA",
    "cuMemcpyHtoD": "hipMemcpyHtoD",
    "cuMemcpyHtoDAsync": "hipMemcpyHtoDAsync",
    "cuMemcpyHtoDAsync_v2": "hipMemcpyHtoDAsync",
    "cuMemcpyHtoD_v2": "hipMemcpyHtoD",
    "cuMemsetD16": "hipMemsetD16",
    "cuMemsetD16Async": "hipMemsetD16Async",
    "cuMemsetD16_v2": "hipMemsetD16",
    "cuMemsetD32": "hipMemsetD32",
    "cuMemsetD32Async": "hipMemsetD32Async",
    "cuMemsetD32_v2": "hipMemsetD32",
    "cuMemsetD8": "hipMemsetD8",
    "cuMemsetD8Async": "hipMemsetD8Async",
    "cuMemsetD8_v2": "hipMemsetD8",
    "cuMipmappedArrayCreate": "hipMipmappedArrayCreate",
    "cuMipmappedArrayDestroy": "hipMipmappedArrayDestroy",
    "cuMipmappedArrayGetLevel": "hipMipmappedArrayGetLevel",
    "cudaFree": "hipFree",
    "cudaFreeArray": "hipFreeArray",
    "cudaFreeAsync": "hipFreeAsync",
    "cudaFreeHost": "hipHostFree",
    "cudaFreeMipmappedArray": "hipFreeMipmappedArray",
    "cudaGetMipmappedArrayLevel": "hipGetMipmappedArrayLevel",
    "cudaGetSymbolAddress": "hipGetSymbolAddress",
    "cudaGetSymbolSize": "hipGetSymbolSize",
    "cudaHostAlloc": "hipHostAlloc",
    "cudaHostGetDevicePointer": "hipHostGetDevicePointer",
    "cudaHostGetFlags": "hipHostGetFlags",
    "cudaHostRegister": "hipHostRegister",
    "cudaHostUnregister": "hipHostUnregister",
    "cudaMalloc": "hipMalloc",
    "cudaMalloc3D": "hipMalloc3D",
    "cudaMalloc3DArray": "hipMalloc3DArray",
    "cudaMallocArray": "hipMallocArray",
    "cudaMallocAsync": "hipMallocAsync",
    "cudaMallocFromPoolAsync": "hipMallocFromPoolAsync",
    "cudaMallocHost": "hipHostMalloc",
    "cudaMallocManaged": "hipMallocManaged",
    "cudaMallocMipmappedArray": "hipMallocMipmappedArray",
    "cudaMallocPitch": "hipMallocPitch",
    "cudaMemAdvise": "hipMemAdvise",
    "cudaMemGetInfo": "hipMemGetInfo",
    "cudaMemPoolCreate": "hipMemPoolCreate",
    "cudaMemPoolDestroy": "hipMemPoolDestroy",
    "cudaMemPoolExportPointer": "hipMemPoolExportPointer",
    "cudaMemPoolExportToShareableHandle": "hipMemPoolExportToShareableHandle",
    "cudaMemPoolGetAccess": "hipMemPoolGetAccess",
    "cudaMemPoolGetAttribute": "hipMemPoolGetAttribute",
    "cudaMemPoolImportFromShareableHandle": "hipMemPoolImportFromShareableHandle",
    "cudaMemPoolImportPointer": "hipMemPoolImportPointer",
    "cudaMemPoolSetAccess": "hipMemPoolSetAccess",
    "cudaMemPoolSetAttribute": "hipMemPoolSetAttribute",
    "cudaMemPoolTrimTo": "hipMemPoolTrimTo",
    "cudaMemPrefetchAsync": "hipMemPrefetchAsync",
    "cudaMemRangeGetAttribute": "hipMemRangeGetAttribute",
    "cudaMemRangeGetAttributes": "hipMemRangeGetAttributes",
    "cudaMemcpy": "hipMemcpy",
    "cudaMemcpy2D": "hipMemcpy2D",
    "cudaMemcpy2DAsync": "hipMemcpy2DAsync",
    "cudaMemcpy2DFromArray": "hipMemcpy2DFromArray",
    "cudaMemcpy2DFromArrayAsync": "hipMemcpy2DFromArrayAsync",
    "cudaMemcpy2DToArray": "hipMemcpy2DToArray",
    "cudaMemcpy2DToArrayAsync": "hipMemcpy2DToArrayAsync",
    "cudaMemcpy3D": "hipMemcpy3D",
    "cudaMemcpy3DAsync": "hipMemcpy3DAsync",
    "cudaMemcpyAsync": "hipMemcpyAsync",
    "cudaMemcpyFromArray": "hipMemcpyFromArray",
    "cudaMemcpyFromSymbol": "hipMemcpyFromSymbol",
    "cudaMemcpyFromSymbolAsync": "hipMemcpyFromSymbolAsync",
    "cudaMemcpyPeer": "hipMemcpyPeer",
    "cudaMemcpyPeerAsync": "hipMemcpyPeerAsync",
    "cudaMemcpyToArray": "hipMemcpyToArray",
    "cudaMemcpyToSymbol": "hipMemcpyToSymbol",
    "cudaMemcpyToSymbolAsync": "hipMemcpyToSymbolAsync",
    "cudaMemset": "hipMemset",
    "cudaMemset2D": "hipMemset2D",
    "cudaMemset2DAsync": "hipMemset2DAsync",
    "cudaMemset3D": "hipMemset3D",
    "cudaMemset3DAsync": "hipMemset3DAsync",
    "cudaMemsetAsync": "hipMemsetAsync",
    "make_cudaExtent": "make_hipExtent",
    "make_cudaPitchedPtr": "make_hipPitchedPtr",
    "make_cudaPos": "make_hipPos",
    "cuMemAddressFree": "hipMemAddressFree",
    "cuMemAddressReserve": "hipMemAddressReserve",
    "cuMemCreate": "hipMemCreate",
    "cuMemExportToShareableHandle": "hipMemExportToShareableHandle",
    "cuMemGetAccess": "hipMemGetAccess",
    "cuMemGetAllocationGranularity": "hipMemGetAllocationGranularity",
    "cuMemGetAllocationPropertiesFromHandle": "hipMemGetAllocationPropertiesFromHandle",
    "cuMemImportFromShareableHandle": "hipMemImportFromShareableHandle",
    "cuMemMap": "hipMemMap",
    "cuMemMapArrayAsync": "hipMemMapArrayAsync",
    "cuMemRelease": "hipMemRelease",
    "cuMemRetainAllocationHandle": "hipMemRetainAllocationHandle",
    "cuMemSetAccess": "hipMemSetAccess",
    "cuMemUnmap": "hipMemUnmap",
    "cuMemAllocAsync": "hipMallocAsync",
    "cuMemAllocFromPoolAsync": "hipMallocFromPoolAsync",
    "cuMemFreeAsync": "hipFreeAsync",
    "cuMemPoolCreate": "hipMemPoolCreate",
    "cuMemPoolDestroy": "hipMemPoolDestroy",
    "cuMemPoolExportPointer": "hipMemPoolExportPointer",
    "cuMemPoolExportToShareableHandle": "hipMemPoolExportToShareableHandle",
    "cuMemPoolGetAccess": "hipMemPoolGetAccess",
    "cuMemPoolGetAttribute": "hipMemPoolGetAttribute",
    "cuMemPoolImportFromShareableHandle": "hipMemPoolImportFromShareableHandle",
    "cuMemPoolImportPointer": "hipMemPoolImportPointer",
    "cuMemPoolSetAccess": "hipMemPoolSetAccess",
    "cuMemPoolSetAttribute": "hipMemPoolSetAttribute",
    "cuMemPoolTrimTo": "hipMemPoolTrimTo",
    "cuMemAdvise": "hipMemAdvise",
    "cuMemPrefetchAsync": "hipMemPrefetchAsync",
    "cuMemRangeGetAttribute": "hipMemRangeGetAttribute",
    "cuMemRangeGetAttributes": "hipMemRangeGetAttributes",
    "cuPointerGetAttribute": "hipPointerGetAttribute",
    "cuPointerGetAttributes": "hipDrvPointerGetAttributes",
    "cudaPointerGetAttributes": "hipPointerGetAttributes",
    "cuStreamAddCallback": "hipStreamAddCallback",
    "cuStreamAttachMemAsync": "hipStreamAttachMemAsync",
    "cuStreamBeginCapture": "hipStreamBeginCapture",
    "cuStreamBeginCapture_v2": "hipStreamBeginCapture",
    "cuStreamCreate": "hipStreamCreateWithFlags",
    "cuStreamCreateWithPriority": "hipStreamCreateWithPriority",
    "cuStreamDestroy": "hipStreamDestroy",
    "cuStreamDestroy_v2": "hipStreamDestroy",
    "cuStreamEndCapture": "hipStreamEndCapture",
    "cuStreamGetCaptureInfo": "hipStreamGetCaptureInfo",
    "cuStreamGetCaptureInfo_v2": "hipStreamGetCaptureInfo_v2",
    "cuStreamGetFlags": "hipStreamGetFlags",
    "cuStreamGetPriority": "hipStreamGetPriority",
    "cuStreamIsCapturing": "hipStreamIsCapturing",
    "cuStreamQuery": "hipStreamQuery",
    "cuStreamSynchronize": "hipStreamSynchronize",
    "cuStreamUpdateCaptureDependencies": "hipStreamUpdateCaptureDependencies",
    "cuStreamWaitEvent": "hipStreamWaitEvent",
    "cuThreadExchangeStreamCaptureMode": "hipThreadExchangeStreamCaptureMode",
    "cudaStreamAddCallback": "hipStreamAddCallback",
    "cudaStreamAttachMemAsync": "hipStreamAttachMemAsync",
    "cudaStreamBeginCapture": "hipStreamBeginCapture",
    "cudaStreamCreate": "hipStreamCreate",
    "cudaStreamCreateWithFlags": "hipStreamCreateWithFlags",
    "cudaStreamCreateWithPriority": "hipStreamCreateWithPriority",
    "cudaStreamDestroy": "hipStreamDestroy",
    "cudaStreamEndCapture": "hipStreamEndCapture",
    "cudaStreamGetCaptureInfo": "hipStreamGetCaptureInfo",
    "cudaStreamGetFlags": "hipStreamGetFlags",
    "cudaStreamGetPriority": "hipStreamGetPriority",
    "cudaStreamIsCapturing": "hipStreamIsCapturing",
    "cudaStreamQuery": "hipStreamQuery",
    "cudaStreamSynchronize": "hipStreamSynchronize",
    "cudaStreamWaitEvent": "hipStreamWaitEvent",
    "cudaThreadExchangeStreamCaptureMode": "hipThreadExchangeStreamCaptureMode",
    "cuEventCreate": "hipEventCreateWithFlags",
    "cuEventDestroy": "hipEventDestroy",
    "cuEventDestroy_v2": "hipEventDestroy",
    "cuEventElapsedTime": "hipEventElapsedTime",
    "cuEventQuery": "hipEventQuery",
    "cuEventRecord": "hipEventRecord",
    "cuEventSynchronize": "hipEventSynchronize",
    "cudaEventCreate": "hipEventCreate",
    "cudaEventCreateWithFlags": "hipEventCreateWithFlags",
    "cudaEventDestroy": "hipEventDestroy",
    "cudaEventElapsedTime": "hipEventElapsedTime",
    "cudaEventQuery": "hipEventQuery",
    "cudaEventRecord": "hipEventRecord",
    "cudaEventSynchronize": "hipEventSynchronize",
    "cuDestroyExternalMemory": "hipDestroyExternalMemory",
    "cuDestroyExternalSemaphore": "hipDestroyExternalSemaphore",
    "cuExternalMemoryGetMappedBuffer": "hipExternalMemoryGetMappedBuffer",
    "cuImportExternalMemory": "hipImportExternalMemory",
    "cuImportExternalSemaphore": "hipImportExternalSemaphore",
    "cuSignalExternalSemaphoresAsync": "hipSignalExternalSemaphoresAsync",
    "cuWaitExternalSemaphoresAsync": "hipWaitExternalSemaphoresAsync",
    "cudaDestroyExternalMemory": "hipDestroyExternalMemory",
    "cudaDestroyExternalSemaphore": "hipDestroyExternalSemaphore",
    "cudaExternalMemoryGetMappedBuffer": "hipExternalMemoryGetMappedBuffer",
    "cudaImportExternalMemory": "hipImportExternalMemory",
    "cudaImportExternalSemaphore": "hipImportExternalSemaphore",
    "cudaSignalExternalSemaphoresAsync": "hipSignalExternalSemaphoresAsync",
    "cudaWaitExternalSemaphoresAsync": "hipWaitExternalSemaphoresAsync",
    "cuStreamWaitValue32": "hipStreamWaitValue32",
    "cuStreamWaitValue32_v2": "hipStreamWaitValue32",
    "cuStreamWaitValue64": "hipStreamWaitValue64",
    "cuStreamWaitValue64_v2": "hipStreamWaitValue64",
    "cuStreamWriteValue32": "hipStreamWriteValue32",
    "cuStreamWriteValue32_v2": "hipStreamWriteValue32",
    "cuStreamWriteValue64": "hipStreamWriteValue64",
    "cuStreamWriteValue64_v2": "hipStreamWriteValue64",
    "cuFuncGetAttribute": "hipFuncGetAttribute",
    "cuLaunchHostFunc": "hipLaunchHostFunc",
    "cuLaunchKernel": "hipModuleLaunchKernel",
    "cudaConfigureCall": "hipConfigureCall",
    "cudaFuncGetAttributes": "hipFuncGetAttributes",
    "cudaFuncSetAttribute": "hipFuncSetAttribute",
    "cudaFuncSetSharedMemConfig": "hipFuncSetSharedMemConfig",
    "cudaLaunch": "hipLaunchByPtr",
    "cudaLaunchCooperativeKernel": "hipLaunchCooperativeKernel",
    "cudaLaunchCooperativeKernelMultiDevice": "hipLaunchCooperativeKernelMultiDevice",
    "cudaLaunchHostFunc": "hipLaunchHostFunc",
    "cudaLaunchKernel": "hipLaunchKernel",
    "cudaSetupArgument": "hipSetupArgument",
    "cuDeviceGetGraphMemAttribute": "hipDeviceGetGraphMemAttribute",
    "cuDeviceGraphMemTrim": "hipDeviceGraphMemTrim",
    "cuDeviceSetGraphMemAttribute": "hipDeviceSetGraphMemAttribute",
    "cuGraphAddChildGraphNode": "hipGraphAddChildGraphNode",
    "cuGraphAddDependencies": "hipGraphAddDependencies",
    "cuGraphAddEmptyNode": "hipGraphAddEmptyNode",
    "cuGraphAddEventRecordNode": "hipGraphAddEventRecordNode",
    "cuGraphAddEventWaitNode": "hipGraphAddEventWaitNode",
    "cuGraphAddHostNode": "hipGraphAddHostNode",
    "cuGraphAddKernelNode": "hipGraphAddKernelNode",
    "cuGraphChildGraphNodeGetGraph": "hipGraphChildGraphNodeGetGraph",
    "cuGraphClone": "hipGraphClone",
    "cuGraphCreate": "hipGraphCreate",
    "cuGraphDestroy": "hipGraphDestroy",
    "cuGraphDestroyNode": "hipGraphDestroyNode",
    "cuGraphEventRecordNodeGetEvent": "hipGraphEventRecordNodeGetEvent",
    "cuGraphEventRecordNodeSetEvent": "hipGraphEventRecordNodeSetEvent",
    "cuGraphEventWaitNodeGetEvent": "hipGraphEventWaitNodeGetEvent",
    "cuGraphEventWaitNodeSetEvent": "hipGraphEventWaitNodeSetEvent",
    "cuGraphExecChildGraphNodeSetParams": "hipGraphExecChildGraphNodeSetParams",
    "cuGraphExecDestroy": "hipGraphExecDestroy",
    "cuGraphExecEventRecordNodeSetEvent": "hipGraphExecEventRecordNodeSetEvent",
    "cuGraphExecEventWaitNodeSetEvent": "hipGraphExecEventWaitNodeSetEvent",
    "cuGraphExecHostNodeSetParams": "hipGraphExecHostNodeSetParams",
    "cuGraphExecKernelNodeSetParams": "hipGraphExecKernelNodeSetParams",
    "cuGraphExecUpdate": "hipGraphExecUpdate",
    "cuGraphGetEdges": "hipGraphGetEdges",
    "cuGraphGetNodes": "hipGraphGetNodes",
    "cuGraphGetRootNodes": "hipGraphGetRootNodes",
    "cuGraphHostNodeGetParams": "hipGraphHostNodeGetParams",
    "cuGraphHostNodeSetParams": "hipGraphHostNodeSetParams",
    "cuGraphInstantiate": "hipGraphInstantiate",
    "cuGraphInstantiateWithFlags": "hipGraphInstantiateWithFlags",
    "cuGraphInstantiate_v2": "hipGraphInstantiate",
    "cuGraphKernelNodeGetAttribute": "hipGraphKernelNodeGetAttribute",
    "cuGraphKernelNodeGetParams": "hipGraphKernelNodeGetParams",
    "cuGraphKernelNodeSetAttribute": "hipGraphKernelNodeSetAttribute",
    "cuGraphKernelNodeSetParams": "hipGraphKernelNodeSetParams",
    "cuGraphLaunch": "hipGraphLaunch",
    "cuGraphMemcpyNodeGetParams": "hipGraphMemcpyNodeGetParams",
    "cuGraphMemcpyNodeSetParams": "hipGraphMemcpyNodeSetParams",
    "cuGraphMemsetNodeGetParams": "hipGraphMemsetNodeGetParams",
    "cuGraphMemsetNodeSetParams": "hipGraphMemsetNodeSetParams",
    "cuGraphNodeFindInClone": "hipGraphNodeFindInClone",
    "cuGraphNodeGetDependencies": "hipGraphNodeGetDependencies",
    "cuGraphNodeGetDependentNodes": "hipGraphNodeGetDependentNodes",
    "cuGraphNodeGetType": "hipGraphNodeGetType",
    "cuGraphReleaseUserObject": "hipGraphReleaseUserObject",
    "cuGraphRemoveDependencies": "hipGraphRemoveDependencies",
    "cuGraphRetainUserObject": "hipGraphRetainUserObject",
    "cuGraphUpload": "hipGraphUpload",
    "cuUserObjectCreate": "hipUserObjectCreate",
    "cuUserObjectRelease": "hipUserObjectRelease",
    "cuUserObjectRetain": "hipUserObjectRetain",
    "cudaDeviceGetGraphMemAttribute": "hipDeviceGetGraphMemAttribute",
    "cudaDeviceGraphMemTrim": "hipDeviceGraphMemTrim",
    "cudaDeviceSetGraphMemAttribute": "hipDeviceSetGraphMemAttribute",
    "cudaGraphAddChildGraphNode": "hipGraphAddChildGraphNode",
    "cudaGraphAddDependencies": "hipGraphAddDependencies",
    "cudaGraphAddEmptyNode": "hipGraphAddEmptyNode",
    "cudaGraphAddEventRecordNode": "hipGraphAddEventRecordNode",
    "cudaGraphAddEventWaitNode": "hipGraphAddEventWaitNode",
    "cudaGraphAddHostNode": "hipGraphAddHostNode",
    "cudaGraphAddKernelNode": "hipGraphAddKernelNode",
    "cudaGraphAddMemcpyNode": "hipGraphAddMemcpyNode",
    "cudaGraphAddMemcpyNode1D": "hipGraphAddMemcpyNode1D",
    "cudaGraphAddMemcpyNodeFromSymbol": "hipGraphAddMemcpyNodeFromSymbol",
    "cudaGraphAddMemcpyNodeToSymbol": "hipGraphAddMemcpyNodeToSymbol",
    "cudaGraphAddMemsetNode": "hipGraphAddMemsetNode",
    "cudaGraphChildGraphNodeGetGraph": "hipGraphChildGraphNodeGetGraph",
    "cudaGraphClone": "hipGraphClone",
    "cudaGraphCreate": "hipGraphCreate",
    "cudaGraphDestroy": "hipGraphDestroy",
    "cudaGraphDestroyNode": "hipGraphDestroyNode",
    "cudaGraphEventRecordNodeGetEvent": "hipGraphEventRecordNodeGetEvent",
    "cudaGraphEventRecordNodeSetEvent": "hipGraphEventRecordNodeSetEvent",
    "cudaGraphEventWaitNodeGetEvent": "hipGraphEventWaitNodeGetEvent",
    "cudaGraphEventWaitNodeSetEvent": "hipGraphEventWaitNodeSetEvent",
    "cudaGraphExecChildGraphNodeSetParams": "hipGraphExecChildGraphNodeSetParams",
    "cudaGraphExecDestroy": "hipGraphExecDestroy",
    "cudaGraphExecEventRecordNodeSetEvent": "hipGraphExecEventRecordNodeSetEvent",
    "cudaGraphExecEventWaitNodeSetEvent": "hipGraphExecEventWaitNodeSetEvent",
    "cudaGraphExecHostNodeSetParams": "hipGraphExecHostNodeSetParams",
    "cudaGraphExecKernelNodeSetParams": "hipGraphExecKernelNodeSetParams",
    "cudaGraphExecMemcpyNodeSetParams": "hipGraphExecMemcpyNodeSetParams",
    "cudaGraphExecMemcpyNodeSetParams1D": "hipGraphExecMemcpyNodeSetParams1D",
    "cudaGraphExecMemcpyNodeSetParamsFromSymbol": "hipGraphExecMemcpyNodeSetParamsFromSymbol",
    "cudaGraphExecMemcpyNodeSetParamsToSymbol": "hipGraphExecMemcpyNodeSetParamsToSymbol",
    "cudaGraphExecMemsetNodeSetParams": "hipGraphExecMemsetNodeSetParams",
    "cudaGraphExecUpdate": "hipGraphExecUpdate",
    "cudaGraphGetEdges": "hipGraphGetEdges",
    "cudaGraphGetNodes": "hipGraphGetNodes",
    "cudaGraphGetRootNodes": "hipGraphGetRootNodes",
    "cudaGraphHostNodeGetParams": "hipGraphHostNodeGetParams",
    "cudaGraphHostNodeSetParams": "hipGraphHostNodeSetParams",
    "cudaGraphInstantiate": "hipGraphInstantiate",
    "cudaGraphInstantiateWithFlags": "hipGraphInstantiateWithFlags",
    "cudaGraphKernelNodeGetAttribute": "hipGraphKernelNodeGetAttribute",
    "cudaGraphKernelNodeGetParams": "hipGraphKernelNodeGetParams",
    "cudaGraphKernelNodeSetAttribute": "hipGraphKernelNodeSetAttribute",
    "cudaGraphKernelNodeSetParams": "hipGraphKernelNodeSetParams",
    "cudaGraphLaunch": "hipGraphLaunch",
    "cudaGraphMemcpyNodeGetParams": "hipGraphMemcpyNodeGetParams",
    "cudaGraphMemcpyNodeSetParams": "hipGraphMemcpyNodeSetParams",
    "cudaGraphMemcpyNodeSetParams1D": "hipGraphMemcpyNodeSetParams1D",
    "cudaGraphMemcpyNodeSetParamsFromSymbol": "hipGraphMemcpyNodeSetParamsFromSymbol",
    "cudaGraphMemcpyNodeSetParamsToSymbol": "hipGraphMemcpyNodeSetParamsToSymbol",
    "cudaGraphMemsetNodeGetParams": "hipGraphMemsetNodeGetParams",
    "cudaGraphMemsetNodeSetParams": "hipGraphMemsetNodeSetParams",
    "cudaGraphNodeFindInClone": "hipGraphNodeFindInClone",
    "cudaGraphNodeGetDependencies": "hipGraphNodeGetDependencies",
    "cudaGraphNodeGetDependentNodes": "hipGraphNodeGetDependentNodes",
    "cudaGraphNodeGetType": "hipGraphNodeGetType",
    "cudaGraphReleaseUserObject": "hipGraphReleaseUserObject",
    "cudaGraphRemoveDependencies": "hipGraphRemoveDependencies",
    "cudaGraphRetainUserObject": "hipGraphRetainUserObject",
    "cudaGraphUpload": "hipGraphUpload",
    "cudaUserObjectCreate": "hipUserObjectCreate",
    "cudaUserObjectRelease": "hipUserObjectRelease",
    "cudaUserObjectRetain": "hipUserObjectRetain",
    "cuOccupancyMaxActiveBlocksPerMultiprocessor": "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",
    "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags": "hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
    "cuOccupancyMaxPotentialBlockSize": "hipModuleOccupancyMaxPotentialBlockSize",
    "cuOccupancyMaxPotentialBlockSizeWithFlags": "hipModuleOccupancyMaxPotentialBlockSizeWithFlags",
    "cudaOccupancyMaxActiveBlocksPerMultiprocessor": "hipOccupancyMaxActiveBlocksPerMultiprocessor",
    "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags": "hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
    "cudaOccupancyMaxPotentialBlockSize": "hipOccupancyMaxPotentialBlockSize",
    "cudaOccupancyMaxPotentialBlockSizeWithFlags": "hipOccupancyMaxPotentialBlockSizeWithFlags",
    "cuTexObjectCreate": "hipTexObjectCreate",
    "cuTexObjectDestroy": "hipTexObjectDestroy",
    "cuTexObjectGetResourceDesc": "hipTexObjectGetResourceDesc",
    "cuTexObjectGetResourceViewDesc": "hipTexObjectGetResourceViewDesc",
    "cuTexObjectGetTextureDesc": "hipTexObjectGetTextureDesc",
    "cuTexRefGetAddress": "hipTexRefGetAddress",
    "cuTexRefGetAddressMode": "hipTexRefGetAddressMode",
    "cuTexRefGetAddress_v2": "hipTexRefGetAddress",
    "cuTexRefGetArray": "hipTexRefGetArray",
    "cuTexRefGetFilterMode": "hipTexRefGetFilterMode",
    "cuTexRefGetFlags": "hipTexRefGetFlags",
    "cuTexRefGetFormat": "hipTexRefGetFormat",
    "cuTexRefGetMaxAnisotropy": "hipTexRefGetMaxAnisotropy",
    "cuTexRefGetMipmapFilterMode": "hipTexRefGetMipmapFilterMode",
    "cuTexRefGetMipmapLevelBias": "hipTexRefGetMipmapLevelBias",
    "cuTexRefGetMipmapLevelClamp": "hipTexRefGetMipmapLevelClamp",
    "cuTexRefGetMipmappedArray": "hipTexRefGetMipMappedArray",
    "cuTexRefSetAddress": "hipTexRefSetAddress",
    "cuTexRefSetAddress2D": "hipTexRefSetAddress2D",
    "cuTexRefSetAddress2D_v2": "hipTexRefSetAddress2D",
    "cuTexRefSetAddress2D_v3": "hipTexRefSetAddress2D",
    "cuTexRefSetAddressMode": "hipTexRefSetAddressMode",
    "cuTexRefSetAddress_v2": "hipTexRefSetAddress",
    "cuTexRefSetArray": "hipTexRefSetArray",
    "cuTexRefSetBorderColor": "hipTexRefSetBorderColor",
    "cuTexRefSetFilterMode": "hipTexRefSetFilterMode",
    "cuTexRefSetFlags": "hipTexRefSetFlags",
    "cuTexRefSetFormat": "hipTexRefSetFormat",
    "cuTexRefSetMaxAnisotropy": "hipTexRefSetMaxAnisotropy",
    "cuTexRefSetMipmapFilterMode": "hipTexRefSetMipmapFilterMode",
    "cuTexRefSetMipmapLevelBias": "hipTexRefSetMipmapLevelBias",
    "cuTexRefSetMipmapLevelClamp": "hipTexRefSetMipmapLevelClamp",
    "cuTexRefSetMipmappedArray": "hipTexRefSetMipmappedArray",
    "cudaBindTexture": "hipBindTexture",
    "cudaBindTexture2D": "hipBindTexture2D",
    "cudaBindTextureToArray": "hipBindTextureToArray",
    "cudaBindTextureToMipmappedArray": "hipBindTextureToMipmappedArray",
    "cudaCreateChannelDesc": "hipCreateChannelDesc",
    "cudaCreateTextureObject": "hipCreateTextureObject",
    "cudaDestroyTextureObject": "hipDestroyTextureObject",
    "cudaGetChannelDesc": "hipGetChannelDesc",
    "cudaGetTextureAlignmentOffset": "hipGetTextureAlignmentOffset",
    "cudaGetTextureObjectResourceDesc": "hipGetTextureObjectResourceDesc",
    "cudaGetTextureObjectResourceViewDesc": "hipGetTextureObjectResourceViewDesc",
    "cudaGetTextureObjectTextureDesc": "hipGetTextureObjectTextureDesc",
    "cudaGetTextureReference": "hipGetTextureReference",
    "cudaUnbindTexture": "hipUnbindTexture",
    "cudaCreateSurfaceObject": "hipCreateSurfaceObject",
    "cudaDestroySurfaceObject": "hipDestroySurfaceObject",
    "cuCtxDisablePeerAccess": "hipCtxDisablePeerAccess",
    "cuCtxEnablePeerAccess": "hipCtxEnablePeerAccess",
    "cuDeviceCanAccessPeer": "hipDeviceCanAccessPeer",
    "cuDeviceGetP2PAttribute": "hipDeviceGetP2PAttribute",
    "cudaDeviceCanAccessPeer": "hipDeviceCanAccessPeer",
    "cudaDeviceDisablePeerAccess": "hipDeviceDisablePeerAccess",
    "cudaDeviceEnablePeerAccess": "hipDeviceEnablePeerAccess",
    "cuGraphicsMapResources": "hipGraphicsMapResources",
    "cuGraphicsResourceGetMappedPointer": "hipGraphicsResourceGetMappedPointer",
    "cuGraphicsResourceGetMappedPointer_v2": "hipGraphicsResourceGetMappedPointer",
    "cuGraphicsSubResourceGetMappedArray": "hipGraphicsSubResourceGetMappedArray",
    "cuGraphicsUnmapResources": "hipGraphicsUnmapResources",
    "cuGraphicsUnregisterResource": "hipGraphicsUnregisterResource",
    "cudaGraphicsMapResources": "hipGraphicsMapResources",
    "cudaGraphicsResourceGetMappedPointer": "hipGraphicsResourceGetMappedPointer",
    "cudaGraphicsSubResourceGetMappedArray": "hipGraphicsSubResourceGetMappedArray",
    "cudaGraphicsUnmapResources": "hipGraphicsUnmapResources",
    "cudaGraphicsUnregisterResource": "hipGraphicsUnregisterResource",
    "cuProfilerStart": "hipProfilerStart",
    "cuProfilerStop": "hipProfilerStop",
    "cudaProfilerStart": "hipProfilerStart",
    "cudaProfilerStop": "hipProfilerStop",
    "cuGLGetDevices": "hipGLGetDevices",
    "cuGraphicsGLRegisterBuffer": "hipGraphicsGLRegisterBuffer",
    "cuGraphicsGLRegisterImage": "hipGraphicsGLRegisterImage",
    "cudaGLGetDevices": "hipGLGetDevices",
    "cudaGraphicsGLRegisterBuffer": "hipGraphicsGLRegisterBuffer",
    "cudaGraphicsGLRegisterImage": "hipGraphicsGLRegisterImage",
    "cudaThreadExit": "hipDeviceReset",
    "cudaThreadGetCacheConfig": "hipDeviceGetCacheConfig",
    "cudaThreadSetCacheConfig": "hipDeviceSetCacheConfig",
    "cudaThreadSynchronize": "hipDeviceSynchronize",
    "cuCabs": "hipCabs",
    "cuCabsf": "hipCabsf",
    "cuCadd": "hipCadd",
    "cuCaddf": "hipCaddf",
    "cuCdiv": "hipCdiv",
    "cuCdivf": "hipCdivf",
    "cuCfma": "hipCfma",
    "cuCfmaf": "hipCfmaf",
    "cuCimag": "hipCimag",
    "cuCimagf": "hipCimagf",
    "cuCmul": "hipCmul",
    "cuCmulf": "hipCmulf",
    "cuComplexDoubleToFloat": "hipComplexDoubleToFloat",
    "cuComplexFloatToDouble": "hipComplexFloatToDouble",
    "cuConj": "hipConj",
    "cuConjf": "hipConjf",
    "cuCreal": "hipCreal",
    "cuCrealf": "hipCrealf",
    "cuCsub": "hipCsub",
    "cuCsubf": "hipCsubf",
    "make_cuComplex": "make_hipComplex",
    "make_cuDoubleComplex": "make_hipDoubleComplex",
    "make_cuFloatComplex": "make_hipFloatComplex",
    "cublasCgeqrfBatched": "hipblasCgeqrfBatched",
    "cublasCgetrfBatched": "hipblasCgetrfBatched",
    "cublasCgetriBatched": "hipblasCgetriBatched",
    "cublasCgetrsBatched": "hipblasCgetrsBatched",
    "cublasDgeqrfBatched": "hipblasDgeqrfBatched",
    "cublasDgetrfBatched": "hipblasDgetrfBatched",
    "cublasDgetriBatched": "hipblasDgetriBatched",
    "cublasDgetrsBatched": "hipblasDgetrsBatched",
    "cublasSgeqrfBatched": "hipblasSgeqrfBatched",
    "cublasSgetrfBatched": "hipblasSgetrfBatched",
    "cublasSgetriBatched": "hipblasSgetriBatched",
    "cublasSgetrsBatched": "hipblasSgetrsBatched",
    "cublasZgeqrfBatched": "hipblasZgeqrfBatched",
    "cublasZgetrfBatched": "hipblasZgetrfBatched",
    "cublasZgetriBatched": "hipblasZgetriBatched",
    "cublasZgetrsBatched": "hipblasZgetrsBatched",
    "cuda_stream": "hip_stream",
    "cudnnActivationBackward": "hipdnnActivationBackward",
    "cudnnActivationForward": "hipdnnActivationForward",
    "cudnnAddTensor": "hipdnnAddTensor",
    "cudnnBatchNormalizationBackward": "hipdnnBatchNormalizationBackward",
    "cudnnBatchNormalizationForwardInference": "hipdnnBatchNormalizationForwardInference",
    "cudnnBatchNormalizationForwardTraining": "hipdnnBatchNormalizationForwardTraining",
    "cudnnConvolutionBackwardBias": "hipdnnConvolutionBackwardBias",
    "cudnnConvolutionBackwardData": "hipdnnConvolutionBackwardData",
    "cudnnConvolutionBackwardFilter": "hipdnnConvolutionBackwardFilter",
    "cudnnConvolutionForward": "hipdnnConvolutionForward",
    "cudnnCreate": "hipdnnCreate",
    "cudnnCreateActivationDescriptor": "hipdnnCreateActivationDescriptor",
    "cudnnCreateConvolutionDescriptor": "hipdnnCreateConvolutionDescriptor",
    "cudnnCreateDropoutDescriptor": "hipdnnCreateDropoutDescriptor",
    "cudnnCreateFilterDescriptor": "hipdnnCreateFilterDescriptor",
    "cudnnCreateLRNDescriptor": "hipdnnCreateLRNDescriptor",
    "cudnnCreateOpTensorDescriptor": "hipdnnCreateOpTensorDescriptor",
    "cudnnCreatePersistentRNNPlan": "hipdnnCreatePersistentRNNPlan",
    "cudnnCreatePoolingDescriptor": "hipdnnCreatePoolingDescriptor",
    "cudnnCreateRNNDescriptor": "hipdnnCreateRNNDescriptor",
    "cudnnCreateReduceTensorDescriptor": "hipdnnCreateReduceTensorDescriptor",
    "cudnnCreateTensorDescriptor": "hipdnnCreateTensorDescriptor",
    "cudnnDeriveBNTensorDescriptor": "hipdnnDeriveBNTensorDescriptor",
    "cudnnDestroy": "hipdnnDestroy",
    "cudnnDestroyActivationDescriptor": "hipdnnDestroyActivationDescriptor",
    "cudnnDestroyConvolutionDescriptor": "hipdnnDestroyConvolutionDescriptor",
    "cudnnDestroyDropoutDescriptor": "hipdnnDestroyDropoutDescriptor",
    "cudnnDestroyFilterDescriptor": "hipdnnDestroyFilterDescriptor",
    "cudnnDestroyLRNDescriptor": "hipdnnDestroyLRNDescriptor",
    "cudnnDestroyOpTensorDescriptor": "hipdnnDestroyOpTensorDescriptor",
    "cudnnDestroyPersistentRNNPlan": "hipdnnDestroyPersistentRNNPlan",
    "cudnnDestroyPoolingDescriptor": "hipdnnDestroyPoolingDescriptor",
    "cudnnDestroyRNNDescriptor": "hipdnnDestroyRNNDescriptor",
    "cudnnDestroyReduceTensorDescriptor": "hipdnnDestroyReduceTensorDescriptor",
    "cudnnDestroyTensorDescriptor": "hipdnnDestroyTensorDescriptor",
    "cudnnDropoutGetStatesSize": "hipdnnDropoutGetStatesSize",
    "cudnnFindConvolutionBackwardDataAlgorithm": "hipdnnFindConvolutionBackwardDataAlgorithm",
    "cudnnFindConvolutionBackwardDataAlgorithmEx": "hipdnnFindConvolutionBackwardDataAlgorithmEx",
    "cudnnFindConvolutionBackwardFilterAlgorithm": "hipdnnFindConvolutionBackwardFilterAlgorithm",
    "cudnnFindConvolutionBackwardFilterAlgorithmEx": "hipdnnFindConvolutionBackwardFilterAlgorithmEx",
    "cudnnFindConvolutionForwardAlgorithm": "hipdnnFindConvolutionForwardAlgorithm",
    "cudnnFindConvolutionForwardAlgorithmEx": "hipdnnFindConvolutionForwardAlgorithmEx",
    "cudnnGetActivationDescriptor": "hipdnnGetActivationDescriptor",
    "cudnnGetConvolution2dDescriptor": "hipdnnGetConvolution2dDescriptor",
    "cudnnGetConvolution2dForwardOutputDim": "hipdnnGetConvolution2dForwardOutputDim",
    "cudnnGetConvolutionBackwardDataAlgorithm": "hipdnnGetConvolutionBackwardDataAlgorithm",
    "cudnnGetConvolutionBackwardDataWorkspaceSize": "hipdnnGetConvolutionBackwardDataWorkspaceSize",
    "cudnnGetConvolutionBackwardFilterAlgorithm": "hipdnnGetConvolutionBackwardFilterAlgorithm",
    "cudnnGetConvolutionBackwardFilterWorkspaceSize": "hipdnnGetConvolutionBackwardFilterWorkspaceSize",
    "cudnnGetConvolutionForwardAlgorithm": "hipdnnGetConvolutionForwardAlgorithm",
    "cudnnGetConvolutionForwardWorkspaceSize": "hipdnnGetConvolutionForwardWorkspaceSize",
    "cudnnGetErrorString": "hipdnnGetErrorString",
    "cudnnGetFilter4dDescriptor": "hipdnnGetFilter4dDescriptor",
    "cudnnGetFilterNdDescriptor": "hipdnnGetFilterNdDescriptor",
    "cudnnGetLRNDescriptor": "hipdnnGetLRNDescriptor",
    "cudnnGetOpTensorDescriptor": "hipdnnGetOpTensorDescriptor",
    "cudnnGetPooling2dDescriptor": "hipdnnGetPooling2dDescriptor",
    "cudnnGetPooling2dForwardOutputDim": "hipdnnGetPooling2dForwardOutputDim",
    "cudnnGetRNNDescriptor": "hipdnnGetRNNDescriptor",
    "cudnnGetRNNLinLayerBiasParams": "hipdnnGetRNNLinLayerBiasParams",
    "cudnnGetRNNLinLayerMatrixParams": "hipdnnGetRNNLinLayerMatrixParams",
    "cudnnGetRNNParamsSize": "hipdnnGetRNNParamsSize",
    "cudnnGetRNNTrainingReserveSize": "hipdnnGetRNNTrainingReserveSize",
    "cudnnGetRNNWorkspaceSize": "hipdnnGetRNNWorkspaceSize",
    "cudnnGetReduceTensorDescriptor": "hipdnnGetReduceTensorDescriptor",
    "cudnnGetReductionWorkspaceSize": "hipdnnGetReductionWorkspaceSize",
    "cudnnGetStream": "hipdnnGetStream",
    "cudnnGetTensor4dDescriptor": "hipdnnGetTensor4dDescriptor",
    "cudnnGetTensorNdDescriptor": "hipdnnGetTensorNdDescriptor",
    "cudnnGetVersion": "hipdnnGetVersion",
    "cudnnLRNCrossChannelBackward": "hipdnnLRNCrossChannelBackward",
    "cudnnLRNCrossChannelForward": "hipdnnLRNCrossChannelForward",
    "cudnnOpTensor": "hipdnnOpTensor",
    "cudnnPoolingBackward": "hipdnnPoolingBackward",
    "cudnnPoolingForward": "hipdnnPoolingForward",
    "cudnnRNNBackwardData": "hipdnnRNNBackwardData",
    "cudnnRNNBackwardWeights": "hipdnnRNNBackwardWeights",
    "cudnnRNNForwardInference": "hipdnnRNNForwardInference",
    "cudnnRNNForwardTraining": "hipdnnRNNForwardTraining",
    "cudnnReduceTensor": "hipdnnReduceTensor",
    "cudnnScaleTensor": "hipdnnScaleTensor",
    "cudnnSetActivationDescriptor": "hipdnnSetActivationDescriptor",
    "cudnnSetConvolution2dDescriptor": "hipdnnSetConvolution2dDescriptor",
    "cudnnSetConvolutionGroupCount": "hipdnnSetConvolutionGroupCount",
    "cudnnSetConvolutionMathType": "hipdnnSetConvolutionMathType",
    "cudnnSetConvolutionNdDescriptor": "hipdnnSetConvolutionNdDescriptor",
    "cudnnSetDropoutDescriptor": "hipdnnSetDropoutDescriptor",
    "cudnnSetFilter4dDescriptor": "hipdnnSetFilter4dDescriptor",
    "cudnnSetFilterNdDescriptor": "hipdnnSetFilterNdDescriptor",
    "cudnnSetLRNDescriptor": "hipdnnSetLRNDescriptor",
    "cudnnSetOpTensorDescriptor": "hipdnnSetOpTensorDescriptor",
    "cudnnSetPersistentRNNPlan": "hipdnnSetPersistentRNNPlan",
    "cudnnSetPooling2dDescriptor": "hipdnnSetPooling2dDescriptor",
    "cudnnSetPoolingNdDescriptor": "hipdnnSetPoolingNdDescriptor",
    "cudnnSetRNNDescriptor": "hipdnnSetRNNDescriptor",
    "cudnnSetRNNDescriptor_v5": "hipdnnSetRNNDescriptor_v5",
    "cudnnSetRNNDescriptor_v6": "hipdnnSetRNNDescriptor_v6",
    "cudnnSetReduceTensorDescriptor": "hipdnnSetReduceTensorDescriptor",
    "cudnnSetStream": "hipdnnSetStream",
    "cudnnSetTensor": "hipdnnSetTensor",
    "cudnnSetTensor4dDescriptor": "hipdnnSetTensor4dDescriptor",
    "cudnnSetTensor4dDescriptorEx": "hipdnnSetTensor4dDescriptorEx",
    "cudnnSetTensorNdDescriptor": "hipdnnSetTensorNdDescriptor",
    "cudnnSoftmaxBackward": "hipdnnSoftmaxBackward",
    "cudnnSoftmaxForward": "hipdnnSoftmaxForward",
    "cufftCallbackLoadC": "hipfftCallbackLoadC",
    "cufftCallbackLoadD": "hipfftCallbackLoadD",
    "cufftCallbackLoadR": "hipfftCallbackLoadR",
    "cufftCallbackLoadZ": "hipfftCallbackLoadZ",
    "cufftCallbackStoreC": "hipfftCallbackStoreC",
    "cufftCallbackStoreD": "hipfftCallbackStoreD",
    "cufftCallbackStoreR": "hipfftCallbackStoreR",
    "cufftCallbackStoreZ": "hipfftCallbackStoreZ",
    "cufftCreate": "hipfftCreate",
    "cufftDestroy": "hipfftDestroy",
    "cufftEstimate1d": "hipfftEstimate1d",
    "cufftEstimate2d": "hipfftEstimate2d",
    "cufftEstimate3d": "hipfftEstimate3d",
    "cufftEstimateMany": "hipfftEstimateMany",
    "cufftExecC2C": "hipfftExecC2C",
    "cufftExecC2R": "hipfftExecC2R",
    "cufftExecD2Z": "hipfftExecD2Z",
    "cufftExecR2C": "hipfftExecR2C",
    "cufftExecZ2D": "hipfftExecZ2D",
    "cufftExecZ2Z": "hipfftExecZ2Z",
    "cufftGetProperty": "hipfftGetProperty",
    "cufftGetSize": "hipfftGetSize",
    "cufftGetSize1d": "hipfftGetSize1d",
    "cufftGetSize2d": "hipfftGetSize2d",
    "cufftGetSize3d": "hipfftGetSize3d",
    "cufftGetSizeMany": "hipfftGetSizeMany",
    "cufftGetSizeMany64": "hipfftGetSizeMany64",
    "cufftGetVersion": "hipfftGetVersion",
    "cufftMakePlan1d": "hipfftMakePlan1d",
    "cufftMakePlan2d": "hipfftMakePlan2d",
    "cufftMakePlan3d": "hipfftMakePlan3d",
    "cufftMakePlanMany": "hipfftMakePlanMany",
    "cufftMakePlanMany64": "hipfftMakePlanMany64",
    "cufftPlan1d": "hipfftPlan1d",
    "cufftPlan2d": "hipfftPlan2d",
    "cufftPlan3d": "hipfftPlan3d",
    "cufftPlanMany": "hipfftPlanMany",
    "cufftSetAutoAllocation": "hipfftSetAutoAllocation",
    "cufftSetStream": "hipfftSetStream",
    "cufftSetWorkArea": "hipfftSetWorkArea",
    "cufftXtClearCallback": "hipfftXtClearCallback",
    "cufftXtSetCallback": "hipfftXtSetCallback",
    "cufftXtSetCallbackSharedSize": "hipfftXtSetCallbackSharedSize",
    "curandCreateGenerator": "hiprandCreateGenerator",
    "curandCreateGeneratorHost": "hiprandCreateGeneratorHost",
    "curandCreatePoissonDistribution": "hiprandCreatePoissonDistribution",
    "curandDestroyDistribution": "hiprandDestroyDistribution",
    "curandDestroyGenerator": "hiprandDestroyGenerator",
    "curandGenerate": "hiprandGenerate",
    "curandGenerateLogNormal": "hiprandGenerateLogNormal",
    "curandGenerateLogNormalDouble": "hiprandGenerateLogNormalDouble",
    "curandGenerateNormal": "hiprandGenerateNormal",
    "curandGenerateNormalDouble": "hiprandGenerateNormalDouble",
    "curandGeneratePoisson": "hiprandGeneratePoisson",
    "curandGenerateSeeds": "hiprandGenerateSeeds",
    "curandGenerateUniform": "hiprandGenerateUniform",
    "curandGenerateUniformDouble": "hiprandGenerateUniformDouble",
    "curandGetVersion": "hiprandGetVersion",
    "curandMakeMTGP32Constants": "hiprandMakeMTGP32Constants",
    "curandMakeMTGP32KernelState": "hiprandMakeMTGP32KernelState",
    "curandSetGeneratorOffset": "hiprandSetGeneratorOffset",
    "curandSetPseudoRandomGeneratorSeed": "hiprandSetPseudoRandomGeneratorSeed",
    "curandSetQuasiRandomGeneratorDimensions": "hiprandSetQuasiRandomGeneratorDimensions",
    "curandSetStream": "hiprandSetStream",
    "cusparseAxpby": "hipsparseAxpby",
    "cusparseBlockedEllGet": "hipsparseBlockedEllGet",
    "cusparseCaxpyi": "hipsparseCaxpyi",
    "cusparseCbsr2csr": "hipsparseCbsr2csr",
    "cusparseCbsric02": "hipsparseCbsric02",
    "cusparseCbsric02_analysis": "hipsparseCbsric02_analysis",
    "cusparseCbsric02_bufferSize": "hipsparseCbsric02_bufferSize",
    "cusparseCbsrilu02": "hipsparseCbsrilu02",
    "cusparseCbsrilu02_analysis": "hipsparseCbsrilu02_analysis",
    "cusparseCbsrilu02_bufferSize": "hipsparseCbsrilu02_bufferSize",
    "cusparseCbsrilu02_numericBoost": "hipsparseCbsrilu02_numericBoost",
    "cusparseCbsrmm": "hipsparseCbsrmm",
    "cusparseCbsrmv": "hipsparseCbsrmv",
    "cusparseCbsrsm2_analysis": "hipsparseCbsrsm2_analysis",
    "cusparseCbsrsm2_bufferSize": "hipsparseCbsrsm2_bufferSize",
    "cusparseCbsrsm2_solve": "hipsparseCbsrsm2_solve",
    "cusparseCbsrsv2_analysis": "hipsparseCbsrsv2_analysis",
    "cusparseCbsrsv2_bufferSize": "hipsparseCbsrsv2_bufferSize",
    "cusparseCbsrsv2_bufferSizeExt": "hipsparseCbsrsv2_bufferSizeExt",
    "cusparseCbsrsv2_solve": "hipsparseCbsrsv2_solve",
    "cusparseCbsrxmv": "hipsparseCbsrxmv",
    "cusparseCcsc2dense": "hipsparseCcsc2dense",
    "cusparseCcsr2bsr": "hipsparseCcsr2bsr",
    "cusparseCcsr2csc": "hipsparseCcsr2csc",
    "cusparseCcsr2csr_compress": "hipsparseCcsr2csr_compress",
    "cusparseCcsr2csru": "hipsparseCcsr2csru",
    "cusparseCcsr2dense": "hipsparseCcsr2dense",
    "cusparseCcsr2gebsr": "hipsparseCcsr2gebsr",
    "cusparseCcsr2gebsr_bufferSize": "hipsparseCcsr2gebsr_bufferSize",
    "cusparseCcsr2hyb": "hipsparseCcsr2hyb",
    "cusparseCcsrcolor": "hipsparseCcsrcolor",
    "cusparseCcsrgeam": "hipsparseCcsrgeam",
    "cusparseCcsrgeam2": "hipsparseCcsrgeam2",
    "cusparseCcsrgeam2_bufferSizeExt": "hipsparseCcsrgeam2_bufferSizeExt",
    "cusparseCcsrgemm": "hipsparseCcsrgemm",
    "cusparseCcsrgemm2": "hipsparseCcsrgemm2",
    "cusparseCcsrgemm2_bufferSizeExt": "hipsparseCcsrgemm2_bufferSizeExt",
    "cusparseCcsric02": "hipsparseCcsric02",
    "cusparseCcsric02_analysis": "hipsparseCcsric02_analysis",
    "cusparseCcsric02_bufferSize": "hipsparseCcsric02_bufferSize",
    "cusparseCcsric02_bufferSizeExt": "hipsparseCcsric02_bufferSizeExt",
    "cusparseCcsrilu02": "hipsparseCcsrilu02",
    "cusparseCcsrilu02_analysis": "hipsparseCcsrilu02_analysis",
    "cusparseCcsrilu02_bufferSize": "hipsparseCcsrilu02_bufferSize",
    "cusparseCcsrilu02_bufferSizeExt": "hipsparseCcsrilu02_bufferSizeExt",
    "cusparseCcsrilu02_numericBoost": "hipsparseCcsrilu02_numericBoost",
    "cusparseCcsrmm": "hipsparseCcsrmm",
    "cusparseCcsrmm2": "hipsparseCcsrmm2",
    "cusparseCcsrmv": "hipsparseCcsrmv",
    "cusparseCcsrsm2_analysis": "hipsparseCcsrsm2_analysis",
    "cusparseCcsrsm2_bufferSizeExt": "hipsparseCcsrsm2_bufferSizeExt",
    "cusparseCcsrsm2_solve": "hipsparseCcsrsm2_solve",
    "cusparseCcsrsv2_analysis": "hipsparseCcsrsv2_analysis",
    "cusparseCcsrsv2_bufferSize": "hipsparseCcsrsv2_bufferSize",
    "cusparseCcsrsv2_bufferSizeExt": "hipsparseCcsrsv2_bufferSizeExt",
    "cusparseCcsrsv2_solve": "hipsparseCcsrsv2_solve",
    "cusparseCcsru2csr": "hipsparseCcsru2csr",
    "cusparseCcsru2csr_bufferSizeExt": "hipsparseCcsru2csr_bufferSizeExt",
    "cusparseCdense2csc": "hipsparseCdense2csc",
    "cusparseCdense2csr": "hipsparseCdense2csr",
    "cusparseCdotci": "hipsparseCdotci",
    "cusparseCdoti": "hipsparseCdoti",
    "cusparseCgebsr2csr": "hipsparseCgebsr2csr",
    "cusparseCgebsr2gebsc": "hipsparseCgebsr2gebsc",
    "cusparseCgebsr2gebsc_bufferSize": "hipsparseCgebsr2gebsc_bufferSize",
    "cusparseCgebsr2gebsr": "hipsparseCgebsr2gebsr",
    "cusparseCgebsr2gebsr_bufferSize": "hipsparseCgebsr2gebsr_bufferSize",
    "cusparseCgemmi": "hipsparseCgemmi",
    "cusparseCgemvi": "hipsparseCgemvi",
    "cusparseCgemvi_bufferSize": "hipsparseCgemvi_bufferSize",
    "cusparseCgpsvInterleavedBatch": "hipsparseCgpsvInterleavedBatch",
    "cusparseCgpsvInterleavedBatch_bufferSizeExt": "hipsparseCgpsvInterleavedBatch_bufferSizeExt",
    "cusparseCgthr": "hipsparseCgthr",
    "cusparseCgthrz": "hipsparseCgthrz",
    "cusparseCgtsv2": "hipsparseCgtsv2",
    "cusparseCgtsv2StridedBatch": "hipsparseCgtsv2StridedBatch",
    "cusparseCgtsv2StridedBatch_bufferSizeExt": "hipsparseCgtsv2StridedBatch_bufferSizeExt",
    "cusparseCgtsv2_bufferSizeExt": "hipsparseCgtsv2_bufferSizeExt",
    "cusparseCgtsv2_nopivot": "hipsparseCgtsv2_nopivot",
    "cusparseCgtsv2_nopivot_bufferSizeExt": "hipsparseCgtsv2_nopivot_bufferSizeExt",
    "cusparseCgtsvInterleavedBatch": "hipsparseCgtsvInterleavedBatch",
    "cusparseCgtsvInterleavedBatch_bufferSizeExt": "hipsparseCgtsvInterleavedBatch_bufferSizeExt",
    "cusparseChyb2csr": "hipsparseChyb2csr",
    "cusparseChybmv": "hipsparseChybmv",
    "cusparseCnnz": "hipsparseCnnz",
    "cusparseCnnz_compress": "hipsparseCnnz_compress",
    "cusparseCooAoSGet": "hipsparseCooAoSGet",
    "cusparseCooGet": "hipsparseCooGet",
    "cusparseCooSetPointers": "hipsparseCooSetPointers",
    "cusparseCooSetStridedBatch": "hipsparseCooSetStridedBatch",
    "cusparseCreate": "hipsparseCreate",
    "cusparseCreateBlockedEll": "hipsparseCreateBlockedEll",
    "cusparseCreateBsric02Info": "hipsparseCreateBsric02Info",
    "cusparseCreateBsrilu02Info": "hipsparseCreateBsrilu02Info",
    "cusparseCreateBsrsm2Info": "hipsparseCreateBsrsm2Info",
    "cusparseCreateBsrsv2Info": "hipsparseCreateBsrsv2Info",
    "cusparseCreateColorInfo": "hipsparseCreateColorInfo",
    "cusparseCreateCoo": "hipsparseCreateCoo",
    "cusparseCreateCooAoS": "hipsparseCreateCooAoS",
    "cusparseCreateCsc": "hipsparseCreateCsc",
    "cusparseCreateCsr": "hipsparseCreateCsr",
    "cusparseCreateCsrgemm2Info": "hipsparseCreateCsrgemm2Info",
    "cusparseCreateCsric02Info": "hipsparseCreateCsric02Info",
    "cusparseCreateCsrilu02Info": "hipsparseCreateCsrilu02Info",
    "cusparseCreateCsrsm2Info": "hipsparseCreateCsrsm2Info",
    "cusparseCreateCsrsv2Info": "hipsparseCreateCsrsv2Info",
    "cusparseCreateCsru2csrInfo": "hipsparseCreateCsru2csrInfo",
    "cusparseCreateDnMat": "hipsparseCreateDnMat",
    "cusparseCreateDnVec": "hipsparseCreateDnVec",
    "cusparseCreateHybMat": "hipsparseCreateHybMat",
    "cusparseCreateIdentityPermutation": "hipsparseCreateIdentityPermutation",
    "cusparseCreateMatDescr": "hipsparseCreateMatDescr",
    "cusparseCreatePruneInfo": "hipsparseCreatePruneInfo",
    "cusparseCreateSpVec": "hipsparseCreateSpVec",
    "cusparseCscSetPointers": "hipsparseCscSetPointers",
    "cusparseCsctr": "hipsparseCsctr",
    "cusparseCsrGet": "hipsparseCsrGet",
    "cusparseCsrSetPointers": "hipsparseCsrSetPointers",
    "cusparseCsrSetStridedBatch": "hipsparseCsrSetStridedBatch",
    "cusparseDaxpyi": "hipsparseDaxpyi",
    "cusparseDbsr2csr": "hipsparseDbsr2csr",
    "cusparseDbsric02": "hipsparseDbsric02",
    "cusparseDbsric02_analysis": "hipsparseDbsric02_analysis",
    "cusparseDbsric02_bufferSize": "hipsparseDbsric02_bufferSize",
    "cusparseDbsrilu02": "hipsparseDbsrilu02",
    "cusparseDbsrilu02_analysis": "hipsparseDbsrilu02_analysis",
    "cusparseDbsrilu02_bufferSize": "hipsparseDbsrilu02_bufferSize",
    "cusparseDbsrilu02_numericBoost": "hipsparseDbsrilu02_numericBoost",
    "cusparseDbsrmm": "hipsparseDbsrmm",
    "cusparseDbsrmv": "hipsparseDbsrmv",
    "cusparseDbsrsm2_analysis": "hipsparseDbsrsm2_analysis",
    "cusparseDbsrsm2_bufferSize": "hipsparseDbsrsm2_bufferSize",
    "cusparseDbsrsm2_solve": "hipsparseDbsrsm2_solve",
    "cusparseDbsrsv2_analysis": "hipsparseDbsrsv2_analysis",
    "cusparseDbsrsv2_bufferSize": "hipsparseDbsrsv2_bufferSize",
    "cusparseDbsrsv2_bufferSizeExt": "hipsparseDbsrsv2_bufferSizeExt",
    "cusparseDbsrsv2_solve": "hipsparseDbsrsv2_solve",
    "cusparseDbsrxmv": "hipsparseDbsrxmv",
    "cusparseDcsc2dense": "hipsparseDcsc2dense",
    "cusparseDcsr2bsr": "hipsparseDcsr2bsr",
    "cusparseDcsr2csc": "hipsparseDcsr2csc",
    "cusparseDcsr2csr_compress": "hipsparseDcsr2csr_compress",
    "cusparseDcsr2csru": "hipsparseDcsr2csru",
    "cusparseDcsr2dense": "hipsparseDcsr2dense",
    "cusparseDcsr2gebsr": "hipsparseDcsr2gebsr",
    "cusparseDcsr2gebsr_bufferSize": "hipsparseDcsr2gebsr_bufferSize",
    "cusparseDcsr2hyb": "hipsparseDcsr2hyb",
    "cusparseDcsrcolor": "hipsparseDcsrcolor",
    "cusparseDcsrgeam": "hipsparseDcsrgeam",
    "cusparseDcsrgeam2": "hipsparseDcsrgeam2",
    "cusparseDcsrgeam2_bufferSizeExt": "hipsparseDcsrgeam2_bufferSizeExt",
    "cusparseDcsrgemm": "hipsparseDcsrgemm",
    "cusparseDcsrgemm2": "hipsparseDcsrgemm2",
    "cusparseDcsrgemm2_bufferSizeExt": "hipsparseDcsrgemm2_bufferSizeExt",
    "cusparseDcsric02": "hipsparseDcsric02",
    "cusparseDcsric02_analysis": "hipsparseDcsric02_analysis",
    "cusparseDcsric02_bufferSize": "hipsparseDcsric02_bufferSize",
    "cusparseDcsric02_bufferSizeExt": "hipsparseDcsric02_bufferSizeExt",
    "cusparseDcsrilu02": "hipsparseDcsrilu02",
    "cusparseDcsrilu02_analysis": "hipsparseDcsrilu02_analysis",
    "cusparseDcsrilu02_bufferSize": "hipsparseDcsrilu02_bufferSize",
    "cusparseDcsrilu02_bufferSizeExt": "hipsparseDcsrilu02_bufferSizeExt",
    "cusparseDcsrilu02_numericBoost": "hipsparseDcsrilu02_numericBoost",
    "cusparseDcsrmm": "hipsparseDcsrmm",
    "cusparseDcsrmm2": "hipsparseDcsrmm2",
    "cusparseDcsrmv": "hipsparseDcsrmv",
    "cusparseDcsrsm2_analysis": "hipsparseDcsrsm2_analysis",
    "cusparseDcsrsm2_bufferSizeExt": "hipsparseDcsrsm2_bufferSizeExt",
    "cusparseDcsrsm2_solve": "hipsparseDcsrsm2_solve",
    "cusparseDcsrsv2_analysis": "hipsparseDcsrsv2_analysis",
    "cusparseDcsrsv2_bufferSize": "hipsparseDcsrsv2_bufferSize",
    "cusparseDcsrsv2_bufferSizeExt": "hipsparseDcsrsv2_bufferSizeExt",
    "cusparseDcsrsv2_solve": "hipsparseDcsrsv2_solve",
    "cusparseDcsru2csr": "hipsparseDcsru2csr",
    "cusparseDcsru2csr_bufferSizeExt": "hipsparseDcsru2csr_bufferSizeExt",
    "cusparseDdense2csc": "hipsparseDdense2csc",
    "cusparseDdense2csr": "hipsparseDdense2csr",
    "cusparseDdoti": "hipsparseDdoti",
    "cusparseDenseToSparse_analysis": "hipsparseDenseToSparse_analysis",
    "cusparseDenseToSparse_bufferSize": "hipsparseDenseToSparse_bufferSize",
    "cusparseDenseToSparse_convert": "hipsparseDenseToSparse_convert",
    "cusparseDestroy": "hipsparseDestroy",
    "cusparseDestroyBsric02Info": "hipsparseDestroyBsric02Info",
    "cusparseDestroyBsrilu02Info": "hipsparseDestroyBsrilu02Info",
    "cusparseDestroyBsrsm2Info": "hipsparseDestroyBsrsm2Info",
    "cusparseDestroyBsrsv2Info": "hipsparseDestroyBsrsv2Info",
    "cusparseDestroyColorInfo": "hipsparseDestroyColorInfo",
    "cusparseDestroyCsrgemm2Info": "hipsparseDestroyCsrgemm2Info",
    "cusparseDestroyCsric02Info": "hipsparseDestroyCsric02Info",
    "cusparseDestroyCsrilu02Info": "hipsparseDestroyCsrilu02Info",
    "cusparseDestroyCsrsm2Info": "hipsparseDestroyCsrsm2Info",
    "cusparseDestroyCsrsv2Info": "hipsparseDestroyCsrsv2Info",
    "cusparseDestroyCsru2csrInfo": "hipsparseDestroyCsru2csrInfo",
    "cusparseDestroyDnMat": "hipsparseDestroyDnMat",
    "cusparseDestroyDnVec": "hipsparseDestroyDnVec",
    "cusparseDestroyHybMat": "hipsparseDestroyHybMat",
    "cusparseDestroyMatDescr": "hipsparseDestroyMatDescr",
    "cusparseDestroyPruneInfo": "hipsparseDestroyPruneInfo",
    "cusparseDestroySpMat": "hipsparseDestroySpMat",
    "cusparseDestroySpVec": "hipsparseDestroySpVec",
    "cusparseDgebsr2csr": "hipsparseDgebsr2csr",
    "cusparseDgebsr2gebsc": "hipsparseDgebsr2gebsc",
    "cusparseDgebsr2gebsc_bufferSize": "hipsparseDgebsr2gebsc_bufferSize",
    "cusparseDgebsr2gebsr": "hipsparseDgebsr2gebsr",
    "cusparseDgebsr2gebsr_bufferSize": "hipsparseDgebsr2gebsr_bufferSize",
    "cusparseDgemmi": "hipsparseDgemmi",
    "cusparseDgemvi": "hipsparseDgemvi",
    "cusparseDgemvi_bufferSize": "hipsparseDgemvi_bufferSize",
    "cusparseDgpsvInterleavedBatch": "hipsparseDgpsvInterleavedBatch",
    "cusparseDgpsvInterleavedBatch_bufferSizeExt": "hipsparseDgpsvInterleavedBatch_bufferSizeExt",
    "cusparseDgthr": "hipsparseDgthr",
    "cusparseDgthrz": "hipsparseDgthrz",
    "cusparseDgtsv2": "hipsparseDgtsv2",
    "cusparseDgtsv2StridedBatch": "hipsparseDgtsv2StridedBatch",
    "cusparseDgtsv2StridedBatch_bufferSizeExt": "hipsparseDgtsv2StridedBatch_bufferSizeExt",
    "cusparseDgtsv2_bufferSizeExt": "hipsparseDgtsv2_bufferSizeExt",
    "cusparseDgtsv2_nopivot": "hipsparseDgtsv2_nopivot",
    "cusparseDgtsv2_nopivot_bufferSizeExt": "hipsparseDgtsv2_nopivot_bufferSizeExt",
    "cusparseDgtsvInterleavedBatch": "hipsparseDgtsvInterleavedBatch",
    "cusparseDgtsvInterleavedBatch_bufferSizeExt": "hipsparseDgtsvInterleavedBatch_bufferSizeExt",
    "cusparseDhyb2csr": "hipsparseDhyb2csr",
    "cusparseDhybmv": "hipsparseDhybmv",
    "cusparseDnMatGet": "hipsparseDnMatGet",
    "cusparseDnMatGetStridedBatch": "hipsparseDnMatGetStridedBatch",
    "cusparseDnMatGetValues": "hipsparseDnMatGetValues",
    "cusparseDnMatSetStridedBatch": "hipsparseDnMatSetStridedBatch",
    "cusparseDnMatSetValues": "hipsparseDnMatSetValues",
    "cusparseDnVecGet": "hipsparseDnVecGet",
    "cusparseDnVecGetValues": "hipsparseDnVecGetValues",
    "cusparseDnVecSetValues": "hipsparseDnVecSetValues",
    "cusparseDnnz": "hipsparseDnnz",
    "cusparseDnnz_compress": "hipsparseDnnz_compress",
    "cusparseDpruneCsr2csr": "hipsparseDpruneCsr2csr",
    "cusparseDpruneCsr2csrByPercentage": "hipsparseDpruneCsr2csrByPercentage",
    "cusparseDpruneCsr2csrByPercentage_bufferSizeExt": "hipsparseDpruneCsr2csrByPercentage_bufferSizeExt",
    "cusparseDpruneCsr2csrNnz": "hipsparseDpruneCsr2csrNnz",
    "cusparseDpruneCsr2csrNnzByPercentage": "hipsparseDpruneCsr2csrNnzByPercentage",
    "cusparseDpruneCsr2csr_bufferSizeExt": "hipsparseDpruneCsr2csr_bufferSizeExt",
    "cusparseDpruneDense2csr": "hipsparseDpruneDense2csr",
    "cusparseDpruneDense2csrByPercentage": "hipsparseDpruneDense2csrByPercentage",
    "cusparseDpruneDense2csrByPercentage_bufferSizeExt": "hipsparseDpruneDense2csrByPercentage_bufferSizeExt",
    "cusparseDpruneDense2csrNnz": "hipsparseDpruneDense2csrNnz",
    "cusparseDpruneDense2csrNnzByPercentage": "hipsparseDpruneDense2csrNnzByPercentage",
    "cusparseDpruneDense2csr_bufferSizeExt": "hipsparseDpruneDense2csr_bufferSizeExt",
    "cusparseDroti": "hipsparseDroti",
    "cusparseDsctr": "hipsparseDsctr",
    "cusparseGather": "hipsparseGather",
    "cusparseGetMatDiagType": "hipsparseGetMatDiagType",
    "cusparseGetMatFillMode": "hipsparseGetMatFillMode",
    "cusparseGetMatIndexBase": "hipsparseGetMatIndexBase",
    "cusparseGetMatType": "hipsparseGetMatType",
    "cusparseGetPointerMode": "hipsparseGetPointerMode",
    "cusparseGetStream": "hipsparseGetStream",
    "cusparseGetVersion": "hipsparseGetVersion",
    "cusparseRot": "hipsparseRot",
    "cusparseSDDMM": "hipsparseSDDMM",
    "cusparseSDDMM_bufferSize": "hipsparseSDDMM_bufferSize",
    "cusparseSDDMM_preprocess": "hipsparseSDDMM_preprocess",
    "cusparseSaxpyi": "hipsparseSaxpyi",
    "cusparseSbsr2csr": "hipsparseSbsr2csr",
    "cusparseSbsric02": "hipsparseSbsric02",
    "cusparseSbsric02_analysis": "hipsparseSbsric02_analysis",
    "cusparseSbsric02_bufferSize": "hipsparseSbsric02_bufferSize",
    "cusparseSbsrilu02": "hipsparseSbsrilu02",
    "cusparseSbsrilu02_analysis": "hipsparseSbsrilu02_analysis",
    "cusparseSbsrilu02_bufferSize": "hipsparseSbsrilu02_bufferSize",
    "cusparseSbsrilu02_numericBoost": "hipsparseSbsrilu02_numericBoost",
    "cusparseSbsrmm": "hipsparseSbsrmm",
    "cusparseSbsrmv": "hipsparseSbsrmv",
    "cusparseSbsrsm2_analysis": "hipsparseSbsrsm2_analysis",
    "cusparseSbsrsm2_bufferSize": "hipsparseSbsrsm2_bufferSize",
    "cusparseSbsrsm2_solve": "hipsparseSbsrsm2_solve",
    "cusparseSbsrsv2_analysis": "hipsparseSbsrsv2_analysis",
    "cusparseSbsrsv2_bufferSize": "hipsparseSbsrsv2_bufferSize",
    "cusparseSbsrsv2_bufferSizeExt": "hipsparseSbsrsv2_bufferSizeExt",
    "cusparseSbsrsv2_solve": "hipsparseSbsrsv2_solve",
    "cusparseSbsrxmv": "hipsparseSbsrxmv",
    "cusparseScatter": "hipsparseScatter",
    "cusparseScsc2dense": "hipsparseScsc2dense",
    "cusparseScsr2bsr": "hipsparseScsr2bsr",
    "cusparseScsr2csc": "hipsparseScsr2csc",
    "cusparseScsr2csr_compress": "hipsparseScsr2csr_compress",
    "cusparseScsr2csru": "hipsparseScsr2csru",
    "cusparseScsr2dense": "hipsparseScsr2dense",
    "cusparseScsr2gebsr": "hipsparseScsr2gebsr",
    "cusparseScsr2gebsr_bufferSize": "hipsparseScsr2gebsr_bufferSize",
    "cusparseScsr2hyb": "hipsparseScsr2hyb",
    "cusparseScsrcolor": "hipsparseScsrcolor",
    "cusparseScsrgeam": "hipsparseScsrgeam",
    "cusparseScsrgeam2": "hipsparseScsrgeam2",
    "cusparseScsrgeam2_bufferSizeExt": "hipsparseScsrgeam2_bufferSizeExt",
    "cusparseScsrgemm": "hipsparseScsrgemm",
    "cusparseScsrgemm2": "hipsparseScsrgemm2",
    "cusparseScsrgemm2_bufferSizeExt": "hipsparseScsrgemm2_bufferSizeExt",
    "cusparseScsric02": "hipsparseScsric02",
    "cusparseScsric02_analysis": "hipsparseScsric02_analysis",
    "cusparseScsric02_bufferSize": "hipsparseScsric02_bufferSize",
    "cusparseScsric02_bufferSizeExt": "hipsparseScsric02_bufferSizeExt",
    "cusparseScsrilu02": "hipsparseScsrilu02",
    "cusparseScsrilu02_analysis": "hipsparseScsrilu02_analysis",
    "cusparseScsrilu02_bufferSize": "hipsparseScsrilu02_bufferSize",
    "cusparseScsrilu02_bufferSizeExt": "hipsparseScsrilu02_bufferSizeExt",
    "cusparseScsrilu02_numericBoost": "hipsparseScsrilu02_numericBoost",
    "cusparseScsrmm": "hipsparseScsrmm",
    "cusparseScsrmm2": "hipsparseScsrmm2",
    "cusparseScsrmv": "hipsparseScsrmv",
    "cusparseScsrsm2_analysis": "hipsparseScsrsm2_analysis",
    "cusparseScsrsm2_bufferSizeExt": "hipsparseScsrsm2_bufferSizeExt",
    "cusparseScsrsm2_solve": "hipsparseScsrsm2_solve",
    "cusparseScsrsv2_analysis": "hipsparseScsrsv2_analysis",
    "cusparseScsrsv2_bufferSize": "hipsparseScsrsv2_bufferSize",
    "cusparseScsrsv2_bufferSizeExt": "hipsparseScsrsv2_bufferSizeExt",
    "cusparseScsrsv2_solve": "hipsparseScsrsv2_solve",
    "cusparseScsru2csr": "hipsparseScsru2csr",
    "cusparseScsru2csr_bufferSizeExt": "hipsparseScsru2csr_bufferSizeExt",
    "cusparseSdense2csc": "hipsparseSdense2csc",
    "cusparseSdense2csr": "hipsparseSdense2csr",
    "cusparseSdoti": "hipsparseSdoti",
    "cusparseSetMatDiagType": "hipsparseSetMatDiagType",
    "cusparseSetMatFillMode": "hipsparseSetMatFillMode",
    "cusparseSetMatIndexBase": "hipsparseSetMatIndexBase",
    "cusparseSetMatType": "hipsparseSetMatType",
    "cusparseSetPointerMode": "hipsparseSetPointerMode",
    "cusparseSetStream": "hipsparseSetStream",
    "cusparseSgebsr2csr": "hipsparseSgebsr2csr",
    "cusparseSgebsr2gebsc": "hipsparseSgebsr2gebsc",
    "cusparseSgebsr2gebsc_bufferSize": "hipsparseSgebsr2gebsc_bufferSize",
    "cusparseSgebsr2gebsr": "hipsparseSgebsr2gebsr",
    "cusparseSgebsr2gebsr_bufferSize": "hipsparseSgebsr2gebsr_bufferSize",
    "cusparseSgemmi": "hipsparseSgemmi",
    "cusparseSgemvi": "hipsparseSgemvi",
    "cusparseSgemvi_bufferSize": "hipsparseSgemvi_bufferSize",
    "cusparseSgpsvInterleavedBatch": "hipsparseSgpsvInterleavedBatch",
    "cusparseSgpsvInterleavedBatch_bufferSizeExt": "hipsparseSgpsvInterleavedBatch_bufferSizeExt",
    "cusparseSgthr": "hipsparseSgthr",
    "cusparseSgthrz": "hipsparseSgthrz",
    "cusparseSgtsv2": "hipsparseSgtsv2",
    "cusparseSgtsv2StridedBatch": "hipsparseSgtsv2StridedBatch",
    "cusparseSgtsv2StridedBatch_bufferSizeExt": "hipsparseSgtsv2StridedBatch_bufferSizeExt",
    "cusparseSgtsv2_bufferSizeExt": "hipsparseSgtsv2_bufferSizeExt",
    "cusparseSgtsv2_nopivot": "hipsparseSgtsv2_nopivot",
    "cusparseSgtsv2_nopivot_bufferSizeExt": "hipsparseSgtsv2_nopivot_bufferSizeExt",
    "cusparseSgtsvInterleavedBatch": "hipsparseSgtsvInterleavedBatch",
    "cusparseSgtsvInterleavedBatch_bufferSizeExt": "hipsparseSgtsvInterleavedBatch_bufferSizeExt",
    "cusparseShyb2csr": "hipsparseShyb2csr",
    "cusparseShybmv": "hipsparseShybmv",
    "cusparseSnnz": "hipsparseSnnz",
    "cusparseSnnz_compress": "hipsparseSnnz_compress",
    "cusparseSpGEMM_compute": "hipsparseSpGEMM_compute",
    "cusparseSpGEMM_copy": "hipsparseSpGEMM_copy",
    "cusparseSpGEMM_createDescr": "hipsparseSpGEMM_createDescr",
    "cusparseSpGEMM_destroyDescr": "hipsparseSpGEMM_destroyDescr",
    "cusparseSpGEMM_workEstimation": "hipsparseSpGEMM_workEstimation",
    "cusparseSpMM": "hipsparseSpMM",
    "cusparseSpMM_bufferSize": "hipsparseSpMM_bufferSize",
    "cusparseSpMM_preprocess": "hipsparseSpMM_preprocess",
    "cusparseSpMV": "hipsparseSpMV",
    "cusparseSpMV_bufferSize": "hipsparseSpMV_bufferSize",
    "cusparseSpMatGetAttribute": "hipsparseSpMatGetAttribute",
    "cusparseSpMatGetFormat": "hipsparseSpMatGetFormat",
    "cusparseSpMatGetIndexBase": "hipsparseSpMatGetIndexBase",
    "cusparseSpMatGetSize": "hipsparseSpMatGetSize",
    "cusparseSpMatGetStridedBatch": "hipsparseSpMatGetStridedBatch",
    "cusparseSpMatGetValues": "hipsparseSpMatGetValues",
    "cusparseSpMatSetAttribute": "hipsparseSpMatSetAttribute",
    "cusparseSpMatSetStridedBatch": "hipsparseSpMatSetStridedBatch",
    "cusparseSpMatSetValues": "hipsparseSpMatSetValues",
    "cusparseSpSM_analysis": "hipsparseSpSM_analysis",
    "cusparseSpSM_bufferSize": "hipsparseSpSM_bufferSize",
    "cusparseSpSM_createDescr": "hipsparseSpSM_createDescr",
    "cusparseSpSM_destroyDescr": "hipsparseSpSM_destroyDescr",
    "cusparseSpSM_solve": "hipsparseSpSM_solve",
    "cusparseSpSV_analysis": "hipsparseSpSV_analysis",
    "cusparseSpSV_bufferSize": "hipsparseSpSV_bufferSize",
    "cusparseSpSV_createDescr": "hipsparseSpSV_createDescr",
    "cusparseSpSV_destroyDescr": "hipsparseSpSV_destroyDescr",
    "cusparseSpSV_solve": "hipsparseSpSV_solve",
    "cusparseSpVV": "hipsparseSpVV",
    "cusparseSpVV_bufferSize": "hipsparseSpVV_bufferSize",
    "cusparseSpVecGet": "hipsparseSpVecGet",
    "cusparseSpVecGetIndexBase": "hipsparseSpVecGetIndexBase",
    "cusparseSpVecGetValues": "hipsparseSpVecGetValues",
    "cusparseSpVecSetValues": "hipsparseSpVecSetValues",
    "cusparseSparseToDense": "hipsparseSparseToDense",
    "cusparseSparseToDense_bufferSize": "hipsparseSparseToDense_bufferSize",
    "cusparseSpruneCsr2csr": "hipsparseSpruneCsr2csr",
    "cusparseSpruneCsr2csrByPercentage": "hipsparseSpruneCsr2csrByPercentage",
    "cusparseSpruneCsr2csrByPercentage_bufferSizeExt": "hipsparseSpruneCsr2csrByPercentage_bufferSizeExt",
    "cusparseSpruneCsr2csrNnz": "hipsparseSpruneCsr2csrNnz",
    "cusparseSpruneCsr2csrNnzByPercentage": "hipsparseSpruneCsr2csrNnzByPercentage",
    "cusparseSpruneCsr2csr_bufferSizeExt": "hipsparseSpruneCsr2csr_bufferSizeExt",
    "cusparseSpruneDense2csr": "hipsparseSpruneDense2csr",
    "cusparseSpruneDense2csrByPercentage": "hipsparseSpruneDense2csrByPercentage",
    "cusparseSpruneDense2csrByPercentage_bufferSizeExt": "hipsparseSpruneDense2csrByPercentage_bufferSizeExt",
    "cusparseSpruneDense2csrNnz": "hipsparseSpruneDense2csrNnz",
    "cusparseSpruneDense2csrNnzByPercentage": "hipsparseSpruneDense2csrNnzByPercentage",
    "cusparseSpruneDense2csr_bufferSizeExt": "hipsparseSpruneDense2csr_bufferSizeExt",
    "cusparseSroti": "hipsparseSroti",
    "cusparseSsctr": "hipsparseSsctr",
    "cusparseXbsric02_zeroPivot": "hipsparseXbsric02_zeroPivot",
    "cusparseXbsrilu02_zeroPivot": "hipsparseXbsrilu02_zeroPivot",
    "cusparseXbsrsm2_zeroPivot": "hipsparseXbsrsm2_zeroPivot",
    "cusparseXbsrsv2_zeroPivot": "hipsparseXbsrsv2_zeroPivot",
    "cusparseXcoo2csr": "hipsparseXcoo2csr",
    "cusparseXcoosortByColumn": "hipsparseXcoosortByColumn",
    "cusparseXcoosortByRow": "hipsparseXcoosortByRow",
    "cusparseXcoosort_bufferSizeExt": "hipsparseXcoosort_bufferSizeExt",
    "cusparseXcscsort": "hipsparseXcscsort",
    "cusparseXcscsort_bufferSizeExt": "hipsparseXcscsort_bufferSizeExt",
    "cusparseXcsr2bsrNnz": "hipsparseXcsr2bsrNnz",
    "cusparseXcsr2coo": "hipsparseXcsr2coo",
    "cusparseXcsr2gebsrNnz": "hipsparseXcsr2gebsrNnz",
    "cusparseXcsrgeam2Nnz": "hipsparseXcsrgeam2Nnz",
    "cusparseXcsrgeamNnz": "hipsparseXcsrgeamNnz",
    "cusparseXcsrgemm2Nnz": "hipsparseXcsrgemm2Nnz",
    "cusparseXcsrgemmNnz": "hipsparseXcsrgemmNnz",
    "cusparseXcsric02_zeroPivot": "hipsparseXcsric02_zeroPivot",
    "cusparseXcsrilu02_zeroPivot": "hipsparseXcsrilu02_zeroPivot",
    "cusparseXcsrsm2_zeroPivot": "hipsparseXcsrsm2_zeroPivot",
    "cusparseXcsrsort": "hipsparseXcsrsort",
    "cusparseXcsrsort_bufferSizeExt": "hipsparseXcsrsort_bufferSizeExt",
    "cusparseXcsrsv2_zeroPivot": "hipsparseXcsrsv2_zeroPivot",
    "cusparseXgebsr2gebsrNnz": "hipsparseXgebsr2gebsrNnz",
    "cusparseZaxpyi": "hipsparseZaxpyi",
    "cusparseZbsr2csr": "hipsparseZbsr2csr",
    "cusparseZbsric02": "hipsparseZbsric02",
    "cusparseZbsric02_analysis": "hipsparseZbsric02_analysis",
    "cusparseZbsric02_bufferSize": "hipsparseZbsric02_bufferSize",
    "cusparseZbsrilu02": "hipsparseZbsrilu02",
    "cusparseZbsrilu02_analysis": "hipsparseZbsrilu02_analysis",
    "cusparseZbsrilu02_bufferSize": "hipsparseZbsrilu02_bufferSize",
    "cusparseZbsrilu02_numericBoost": "hipsparseZbsrilu02_numericBoost",
    "cusparseZbsrmm": "hipsparseZbsrmm",
    "cusparseZbsrmv": "hipsparseZbsrmv",
    "cusparseZbsrsm2_analysis": "hipsparseZbsrsm2_analysis",
    "cusparseZbsrsm2_bufferSize": "hipsparseZbsrsm2_bufferSize",
    "cusparseZbsrsm2_solve": "hipsparseZbsrsm2_solve",
    "cusparseZbsrsv2_analysis": "hipsparseZbsrsv2_analysis",
    "cusparseZbsrsv2_bufferSize": "hipsparseZbsrsv2_bufferSize",
    "cusparseZbsrsv2_bufferSizeExt": "hipsparseZbsrsv2_bufferSizeExt",
    "cusparseZbsrsv2_solve": "hipsparseZbsrsv2_solve",
    "cusparseZbsrxmv": "hipsparseZbsrxmv",
    "cusparseZcsc2dense": "hipsparseZcsc2dense",
    "cusparseZcsr2bsr": "hipsparseZcsr2bsr",
    "cusparseZcsr2csc": "hipsparseZcsr2csc",
    "cusparseZcsr2csr_compress": "hipsparseZcsr2csr_compress",
    "cusparseZcsr2csru": "hipsparseZcsr2csru",
    "cusparseZcsr2dense": "hipsparseZcsr2dense",
    "cusparseZcsr2gebsr": "hipsparseZcsr2gebsr",
    "cusparseZcsr2gebsr_bufferSize": "hipsparseZcsr2gebsr_bufferSize",
    "cusparseZcsr2hyb": "hipsparseZcsr2hyb",
    "cusparseZcsrcolor": "hipsparseZcsrcolor",
    "cusparseZcsrgeam": "hipsparseZcsrgeam",
    "cusparseZcsrgeam2": "hipsparseZcsrgeam2",
    "cusparseZcsrgeam2_bufferSizeExt": "hipsparseZcsrgeam2_bufferSizeExt",
    "cusparseZcsrgemm": "hipsparseZcsrgemm",
    "cusparseZcsrgemm2": "hipsparseZcsrgemm2",
    "cusparseZcsrgemm2_bufferSizeExt": "hipsparseZcsrgemm2_bufferSizeExt",
    "cusparseZcsric02": "hipsparseZcsric02",
    "cusparseZcsric02_analysis": "hipsparseZcsric02_analysis",
    "cusparseZcsric02_bufferSize": "hipsparseZcsric02_bufferSize",
    "cusparseZcsric02_bufferSizeExt": "hipsparseZcsric02_bufferSizeExt",
    "cusparseZcsrilu02": "hipsparseZcsrilu02",
    "cusparseZcsrilu02_analysis": "hipsparseZcsrilu02_analysis",
    "cusparseZcsrilu02_bufferSize": "hipsparseZcsrilu02_bufferSize",
    "cusparseZcsrilu02_bufferSizeExt": "hipsparseZcsrilu02_bufferSizeExt",
    "cusparseZcsrilu02_numericBoost": "hipsparseZcsrilu02_numericBoost",
    "cusparseZcsrmm": "hipsparseZcsrmm",
    "cusparseZcsrmm2": "hipsparseZcsrmm2",
    "cusparseZcsrmv": "hipsparseZcsrmv",
    "cusparseZcsrsm2_analysis": "hipsparseZcsrsm2_analysis",
    "cusparseZcsrsm2_bufferSizeExt": "hipsparseZcsrsm2_bufferSizeExt",
    "cusparseZcsrsm2_solve": "hipsparseZcsrsm2_solve",
    "cusparseZcsrsv2_analysis": "hipsparseZcsrsv2_analysis",
    "cusparseZcsrsv2_bufferSize": "hipsparseZcsrsv2_bufferSize",
    "cusparseZcsrsv2_bufferSizeExt": "hipsparseZcsrsv2_bufferSizeExt",
    "cusparseZcsrsv2_solve": "hipsparseZcsrsv2_solve",
    "cusparseZcsru2csr": "hipsparseZcsru2csr",
    "cusparseZcsru2csr_bufferSizeExt": "hipsparseZcsru2csr_bufferSizeExt",
    "cusparseZdense2csc": "hipsparseZdense2csc",
    "cusparseZdense2csr": "hipsparseZdense2csr",
    "cusparseZdotci": "hipsparseZdotci",
    "cusparseZdoti": "hipsparseZdoti",
    "cusparseZgebsr2csr": "hipsparseZgebsr2csr",
    "cusparseZgebsr2gebsc": "hipsparseZgebsr2gebsc",
    "cusparseZgebsr2gebsc_bufferSize": "hipsparseZgebsr2gebsc_bufferSize",
    "cusparseZgebsr2gebsr": "hipsparseZgebsr2gebsr",
    "cusparseZgebsr2gebsr_bufferSize": "hipsparseZgebsr2gebsr_bufferSize",
    "cusparseZgemmi": "hipsparseZgemmi",
    "cusparseZgemvi": "hipsparseZgemvi",
    "cusparseZgemvi_bufferSize": "hipsparseZgemvi_bufferSize",
    "cusparseZgpsvInterleavedBatch": "hipsparseZgpsvInterleavedBatch",
    "cusparseZgpsvInterleavedBatch_bufferSizeExt": "hipsparseZgpsvInterleavedBatch_bufferSizeExt",
    "cusparseZgthr": "hipsparseZgthr",
    "cusparseZgthrz": "hipsparseZgthrz",
    "cusparseZgtsv2": "hipsparseZgtsv2",
    "cusparseZgtsv2StridedBatch": "hipsparseZgtsv2StridedBatch",
    "cusparseZgtsv2StridedBatch_bufferSizeExt": "hipsparseZgtsv2StridedBatch_bufferSizeExt",
    "cusparseZgtsv2_bufferSizeExt": "hipsparseZgtsv2_bufferSizeExt",
    "cusparseZgtsv2_nopivot": "hipsparseZgtsv2_nopivot",
    "cusparseZgtsv2_nopivot_bufferSizeExt": "hipsparseZgtsv2_nopivot_bufferSizeExt",
    "cusparseZgtsvInterleavedBatch": "hipsparseZgtsvInterleavedBatch",
    "cusparseZgtsvInterleavedBatch_bufferSizeExt": "hipsparseZgtsvInterleavedBatch_bufferSizeExt",
    "cusparseZhyb2csr": "hipsparseZhyb2csr",
    "cusparseZhybmv": "hipsparseZhybmv",
    "cusparseZnnz": "hipsparseZnnz",
    "cusparseZnnz_compress": "hipsparseZnnz_compress",
    "cusparseZsctr": "hipsparseZsctr",
    "nvrtcAddNameExpression": "hiprtcAddNameExpression",
    "nvrtcCompileProgram": "hiprtcCompileProgram",
    "nvrtcCreateProgram": "hiprtcCreateProgram",
    "nvrtcDestroyProgram": "hiprtcDestroyProgram",
    "nvrtcGetCUBIN": "hiprtcGetBitcode",
    "nvrtcGetCUBINSize": "hiprtcGetBitcodeSize",
    "nvrtcGetErrorString": "hiprtcGetErrorString",
    "nvrtcGetLoweredName": "hiprtcGetLoweredName",
    "nvrtcGetPTX": "hiprtcGetCode",
    "nvrtcGetPTXSize": "hiprtcGetCodeSize",
    "nvrtcGetProgramLog": "hiprtcGetProgramLog",
    "nvrtcGetProgramLogSize": "hiprtcGetProgramLogSize",
    "nvrtcVersion": "hiprtcVersion",
    "curand": "hiprand",
    "curand_discrete": "hiprand_discrete",
    "curand_discrete4": "hiprand_discrete4",
    "curand_init": "hiprand_init",
    "curand_log_normal": "hiprand_log_normal",
    "curand_log_normal2": "hiprand_log_normal2",
    "curand_log_normal2_double": "hiprand_log_normal2_double",
    "curand_log_normal4": "hiprand_log_normal4",
    "curand_log_normal4_double": "hiprand_log_normal4_double",
    "curand_log_normal_double": "hiprand_log_normal_double",
    "curand_normal": "hiprand_normal",
    "curand_normal2": "hiprand_normal2",
    "curand_normal2_double": "hiprand_normal2_double",
    "curand_normal4": "hiprand_normal4",
    "curand_normal4_double": "hiprand_normal4_double",
    "curand_normal_double": "hiprand_normal_double",
    "curand_poisson": "hiprand_poisson",
    "curand_poisson4": "hiprand_poisson4",
    "curand_uniform": "hiprand_uniform",
    "curand_uniform2_double": "hiprand_uniform2_double",
    "curand_uniform4": "hiprand_uniform4",
    "curand_uniform4_double": "hiprand_uniform4_double",
    "curand_uniform_double": "hiprand_uniform_double",
    "__half": "__half",
    "__half2": "__half2",
    "__half2_raw": "__half2_raw",
    "__half_raw": "__half_raw",
    "CUDAContext": "HIPContext",
    "CUDA_ARRAY3D_DESCRIPTOR": "HIP_ARRAY3D_DESCRIPTOR",
    "CUDA_ARRAY3D_DESCRIPTOR_st": "HIP_ARRAY3D_DESCRIPTOR",
    "CUDA_ARRAY3D_DESCRIPTOR_v2": "HIP_ARRAY3D_DESCRIPTOR",
    "CUDA_ARRAY_DESCRIPTOR": "HIP_ARRAY_DESCRIPTOR",
    "CUDA_ARRAY_DESCRIPTOR_st": "HIP_ARRAY_DESCRIPTOR",
    "CUDA_ARRAY_DESCRIPTOR_v1": "HIP_ARRAY_DESCRIPTOR",
    "CUDA_ARRAY_DESCRIPTOR_v1_st": "HIP_ARRAY_DESCRIPTOR",
    "CUDA_ARRAY_DESCRIPTOR_v2": "HIP_ARRAY_DESCRIPTOR",
    "CUDA_EXTERNAL_MEMORY_BUFFER_DESC": "hipExternalMemoryBufferDesc",
    "CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st": "hipExternalMemoryBufferDesc_st",
    "CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1": "hipExternalMemoryBufferDesc",
    "CUDA_EXTERNAL_MEMORY_HANDLE_DESC": "hipExternalMemoryHandleDesc",
    "CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st": "hipExternalMemoryHandleDesc_st",
    "CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1": "hipExternalMemoryHandleDesc",
    "CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC": "hipExternalSemaphoreHandleDesc",
    "CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st": "hipExternalSemaphoreHandleDesc_st",
    "CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1": "hipExternalSemaphoreHandleDesc",
    "CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS": "hipExternalSemaphoreSignalParams",
    "CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st": "hipExternalSemaphoreSignalParams_st",
    "CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1": "hipExternalSemaphoreSignalParams",
    "CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS": "hipExternalSemaphoreWaitParams",
    "CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st": "hipExternalSemaphoreWaitParams_st",
    "CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1": "hipExternalSemaphoreWaitParams",
    "CUDA_HOST_NODE_PARAMS": "hipHostNodeParams",
    "CUDA_HOST_NODE_PARAMS_st": "hipHostNodeParams",
    "CUDA_HOST_NODE_PARAMS_v1": "hipHostNodeParams",
    "CUDA_KERNEL_NODE_PARAMS": "hipKernelNodeParams",
    "CUDA_KERNEL_NODE_PARAMS_st": "hipKernelNodeParams",
    "CUDA_KERNEL_NODE_PARAMS_v1": "hipKernelNodeParams",
    "CUDA_MEMCPY2D": "hip_Memcpy2D",
    "CUDA_MEMCPY2D_st": "hip_Memcpy2D",
    "CUDA_MEMCPY2D_v1": "hip_Memcpy2D",
    "CUDA_MEMCPY2D_v1_st": "hip_Memcpy2D",
    "CUDA_MEMCPY2D_v2": "hip_Memcpy2D",
    "CUDA_MEMCPY3D": "HIP_MEMCPY3D",
    "CUDA_MEMCPY3D_st": "HIP_MEMCPY3D",
    "CUDA_MEMCPY3D_v1": "HIP_MEMCPY3D",
    "CUDA_MEMCPY3D_v1_st": "HIP_MEMCPY3D",
    "CUDA_MEMCPY3D_v2": "HIP_MEMCPY3D",
    "CUDA_MEMSET_NODE_PARAMS": "hipMemsetParams",
    "CUDA_MEMSET_NODE_PARAMS_st": "hipMemsetParams",
    "CUDA_MEMSET_NODE_PARAMS_v1": "hipMemsetParams",
    "CUDA_RESOURCE_DESC": "HIP_RESOURCE_DESC",
    "CUDA_RESOURCE_DESC_st": "HIP_RESOURCE_DESC_st",
    "CUDA_RESOURCE_DESC_v1": "HIP_RESOURCE_DESC",
    "CUDA_RESOURCE_VIEW_DESC": "HIP_RESOURCE_VIEW_DESC",
    "CUDA_RESOURCE_VIEW_DESC_st": "HIP_RESOURCE_VIEW_DESC_st",
    "CUDA_RESOURCE_VIEW_DESC_v1": "HIP_RESOURCE_VIEW_DESC",
    "CUDA_TEXTURE_DESC": "HIP_TEXTURE_DESC",
    "CUDA_TEXTURE_DESC_st": "HIP_TEXTURE_DESC_st",
    "CUDA_TEXTURE_DESC_v1": "HIP_TEXTURE_DESC",
    "CUGLDeviceList": "hipGLDeviceList",
    "CUGLDeviceList_enum": "hipGLDeviceList",
    "CUaccessPolicyWindow": "hipAccessPolicyWindow",
    "CUaccessPolicyWindow_st": "hipAccessPolicyWindow",
    "CUaccessProperty": "hipAccessProperty",
    "CUaccessProperty_enum": "hipAccessProperty",
    "CUaddress_mode": "HIPaddress_mode",
    "CUaddress_mode_enum": "HIPaddress_mode_enum",
    "CUarray": "hipArray_t",
    "CUarrayMapInfo": "hipArrayMapInfo",
    "CUarrayMapInfo_st": "hipArrayMapInfo",
    "CUarrayMapInfo_v1": "hipArrayMapInfo",
    "CUarraySparseSubresourceType": "hipArraySparseSubresourceType",
    "CUarraySparseSubresourceType_enum": "hipArraySparseSubresourceType",
    "CUarray_format": "hipArray_Format",
    "CUarray_format_enum": "hipArray_Format",
    "CUarray_st": "hipArray",
    "CUcomputemode": "hipComputeMode",
    "CUcomputemode_enum": "hipComputeMode",
    "CUcontext": "hipCtx_t",
    "CUctx_st": "ihipCtx_t",
    "CUdevice": "hipDevice_t",
    "CUdevice_P2PAttribute": "hipDeviceP2PAttr",
    "CUdevice_P2PAttribute_enum": "hipDeviceP2PAttr",
    "CUdevice_attribute": "hipDeviceAttribute_t",
    "CUdevice_attribute_enum": "hipDeviceAttribute_t",
    "CUdevice_v1": "hipDevice_t",
    "CUdeviceptr": "hipDeviceptr_t",
    "CUdeviceptr_v1": "hipDeviceptr_t",
    "CUdeviceptr_v2": "hipDeviceptr_t",
    "CUevent": "hipEvent_t",
    "CUevent_st": "ihipEvent_t",
    "CUexternalMemory": "hipExternalMemory_t",
    "CUexternalMemoryHandleType": "hipExternalMemoryHandleType",
    "CUexternalMemoryHandleType_enum": "hipExternalMemoryHandleType_enum",
    "CUexternalSemaphore": "hipExternalSemaphore_t",
    "CUexternalSemaphoreHandleType": "hipExternalSemaphoreHandleType",
    "CUexternalSemaphoreHandleType_enum": "hipExternalSemaphoreHandleType_enum",
    "CUfilter_mode": "HIPfilter_mode",
    "CUfilter_mode_enum": "HIPfilter_mode_enum",
    "CUfunc_cache": "hipFuncCache_t",
    "CUfunc_cache_enum": "hipFuncCache_t",
    "CUfunc_st": "ihipModuleSymbol_t",
    "CUfunction": "hipFunction_t",
    "CUfunction_attribute": "hipFunction_attribute",
    "CUfunction_attribute_enum": "hipFunction_attribute",
    "CUgraph": "hipGraph_t",
    "CUgraphExec": "hipGraphExec_t",
    "CUgraphExecUpdateResult": "hipGraphExecUpdateResult",
    "CUgraphExecUpdateResult_enum": "hipGraphExecUpdateResult",
    "CUgraphExec_st": "hipGraphExec",
    "CUgraphInstantiate_flags": "hipGraphInstantiateFlags",
    "CUgraphInstantiate_flags_enum": "hipGraphInstantiateFlags",
    "CUgraphMem_attribute": "hipGraphMemAttributeType",
    "CUgraphMem_attribute_enum": "hipGraphMemAttributeType",
    "CUgraphNode": "hipGraphNode_t",
    "CUgraphNodeType": "hipGraphNodeType",
    "CUgraphNodeType_enum": "hipGraphNodeType",
    "CUgraphNode_st": "hipGraphNode",
    "CUgraph_st": "ihipGraph",
    "CUgraphicsRegisterFlags": "hipGraphicsRegisterFlags",
    "CUgraphicsRegisterFlags_enum": "hipGraphicsRegisterFlags",
    "CUgraphicsResource": "hipGraphicsResource_t",
    "CUgraphicsResource_st": "hipGraphicsResource",
    "CUhostFn": "hipHostFn_t",
    "CUipcEventHandle": "hipIpcEventHandle_t",
    "CUipcEventHandle_st": "hipIpcEventHandle_st",
    "CUipcEventHandle_v1": "hipIpcEventHandle_t",
    "CUipcMemHandle": "hipIpcMemHandle_t",
    "CUipcMemHandle_st": "hipIpcMemHandle_st",
    "CUipcMemHandle_v1": "hipIpcMemHandle_t",
    "CUjitInputType": "hiprtcJITInputType",
    "CUjitInputType_enum": "hiprtcJITInputType",
    "CUjit_option": "hipJitOption",
    "CUjit_option_enum": "hipJitOption",
    "CUkernelNodeAttrID": "hipKernelNodeAttrID",
    "CUkernelNodeAttrID_enum": "hipKernelNodeAttrID",
    "CUkernelNodeAttrValue": "hipKernelNodeAttrValue",
    "CUkernelNodeAttrValue_union": "hipKernelNodeAttrValue",
    "CUkernelNodeAttrValue_v1": "hipKernelNodeAttrValue",
    "CUlimit": "hipLimit_t",
    "CUlimit_enum": "hipLimit_t",
    "CUlinkState": "hiprtcLinkState",
    "CUlinkState_st": "ihiprtcLinkState",
    "CUmemAccessDesc": "hipMemAccessDesc",
    "CUmemAccessDesc_st": "hipMemAccessDesc",
    "CUmemAccessDesc_v1": "hipMemAccessDesc",
    "CUmemAccess_flags": "hipMemAccessFlags",
    "CUmemAccess_flags_enum": "hipMemAccessFlags",
    "CUmemAllocationGranularity_flags": "hipMemAllocationGranularity_flags",
    "CUmemAllocationGranularity_flags_enum": "hipMemAllocationGranularity_flags",
    "CUmemAllocationHandleType": "hipMemAllocationHandleType",
    "CUmemAllocationHandleType_enum": "hipMemAllocationHandleType",
    "CUmemAllocationProp": "hipMemAllocationProp",
    "CUmemAllocationProp_st": "hipMemAllocationProp",
    "CUmemAllocationProp_v1": "hipMemAllocationProp",
    "CUmemAllocationType": "hipMemAllocationType",
    "CUmemAllocationType_enum": "hipMemAllocationType",
    "CUmemGenericAllocationHandle": "hipMemGenericAllocationHandle_t",
    "CUmemGenericAllocationHandle_v1": "hipMemGenericAllocationHandle_t",
    "CUmemHandleType": "hipMemHandleType",
    "CUmemHandleType_enum": "hipMemHandleType",
    "CUmemLocation": "hipMemLocation",
    "CUmemLocationType": "hipMemLocationType",
    "CUmemLocationType_enum": "hipMemLocationType",
    "CUmemLocation_st": "hipMemLocation",
    "CUmemLocation_v1": "hipMemLocation",
    "CUmemOperationType": "hipMemOperationType",
    "CUmemOperationType_enum": "hipMemOperationType",
    "CUmemPoolHandle_st": "ihipMemPoolHandle_t",
    "CUmemPoolProps": "hipMemPoolProps",
    "CUmemPoolProps_st": "hipMemPoolProps",
    "CUmemPoolProps_v1": "hipMemPoolProps",
    "CUmemPoolPtrExportData": "hipMemPoolPtrExportData",
    "CUmemPoolPtrExportData_st": "hipMemPoolPtrExportData",
    "CUmemPoolPtrExportData_v1": "hipMemPoolPtrExportData",
    "CUmemPool_attribute": "hipMemPoolAttr",
    "CUmemPool_attribute_enum": "hipMemPoolAttr",
    "CUmem_advise": "hipMemoryAdvise",
    "CUmem_advise_enum": "hipMemoryAdvise",
    "CUmem_range_attribute": "hipMemRangeAttribute",
    "CUmem_range_attribute_enum": "hipMemRangeAttribute",
    "CUmemoryPool": "hipMemPool_t",
    "CUmemorytype": "hipMemoryType",
    "CUmemorytype_enum": "hipMemoryType",
    "CUmipmappedArray": "hipMipmappedArray_t",
    "CUmipmappedArray_st": "hipMipmappedArray",
    "CUmod_st": "ihipModule_t",
    "CUmodule": "hipModule_t",
    "CUpointer_attribute": "hipPointer_attribute",
    "CUpointer_attribute_enum": "hipPointer_attribute",
    "CUresourceViewFormat": "HIPresourceViewFormat",
    "CUresourceViewFormat_enum": "HIPresourceViewFormat_enum",
    "CUresourcetype": "HIPresourcetype",
    "CUresourcetype_enum": "HIPresourcetype_enum",
    "CUresult": "hipError_t",
    "CUsharedconfig": "hipSharedMemConfig",
    "CUsharedconfig_enum": "hipSharedMemConfig",
    "CUstream": "hipStream_t",
    "CUstreamCallback": "hipStreamCallback_t",
    "CUstreamCaptureMode": "hipStreamCaptureMode",
    "CUstreamCaptureMode_enum": "hipStreamCaptureMode",
    "CUstreamCaptureStatus": "hipStreamCaptureStatus",
    "CUstreamCaptureStatus_enum": "hipStreamCaptureStatus",
    "CUstreamUpdateCaptureDependencies_flags": "hipStreamUpdateCaptureDependenciesFlags",
    "CUstreamUpdateCaptureDependencies_flags_enum": "hipStreamUpdateCaptureDependenciesFlags",
    "CUstream_st": "ihipStream_t",
    "CUsurfObject": "hipSurfaceObject_t",
    "CUsurfObject_v1": "hipSurfaceObject_t",
    "CUtexObject": "hipTextureObject_t",
    "CUtexObject_v1": "hipTextureObject_t",
    "CUtexref": "hipTexRef",
    "CUtexref_st": "textureReference",
    "CUuserObject": "hipUserObject_t",
    "CUuserObjectRetain_flags": "hipUserObjectRetainFlags",
    "CUuserObjectRetain_flags_enum": "hipUserObjectRetainFlags",
    "CUuserObject_flags": "hipUserObjectFlags",
    "CUuserObject_flags_enum": "hipUserObjectFlags",
    "CUuserObject_st": "hipUserObject",
    "CUuuid": "hipUUID",
    "CUuuid_st": "hipUUID_t",
    "GLenum": "GLenum",
    "GLuint": "GLuint",
    "bsric02Info_t": "bsric02Info_t",
    "bsrilu02Info_t": "bsrilu02Info_t",
    "bsrsm2Info": "bsrsm2Info",
    "bsrsm2Info_t": "bsrsm2Info_t",
    "bsrsv2Info_t": "bsrsv2Info_t",
    "csrgemm2Info_t": "csrgemm2Info_t",
    "csrilu02Info_t": "csrilu02Info_t",
    "csrsm2Info_t": "csrsm2Info_t",
    "csrsv2Info_t": "csrsv2Info_t",
    "csru2csrInfo": "csru2csrInfo",
    "csru2csrInfo_t": "csru2csrInfo_t",
    "cuComplex": "hipComplex",
    "cuDoubleComplex": "hipDoubleComplex",
    "cuFloatComplex": "hipFloatComplex",
    "cublasComputeType_t": "hipblasDatatype_t",
    "cudaAccessPolicyWindow": "hipAccessPolicyWindow",
    "cudaAccessProperty": "hipAccessProperty",
    "cudaArray": "hipArray",
    "cudaArray_const_t": "hipArray_const_t",
    "cudaArray_t": "hipArray_t",
    "cudaChannelFormatDesc": "hipChannelFormatDesc",
    "cudaChannelFormatKind": "hipChannelFormatKind",
    "cudaComputeMode": "hipComputeMode",
    "cudaDeviceAttr": "hipDeviceAttribute_t",
    "cudaDeviceP2PAttr": "hipDeviceP2PAttr",
    "cudaDeviceProp": "hipDeviceProp_t",
    "cudaError": "hipError_t",
    "cudaError_enum": "hipError_t",
    "cudaError_t": "hipError_t",
    "cudaEvent_t": "hipEvent_t",
    "cudaExtent": "hipExtent",
    "cudaExternalMemoryBufferDesc": "hipExternalMemoryBufferDesc",
    "cudaExternalMemoryHandleDesc": "hipExternalMemoryHandleDesc",
    "cudaExternalMemoryHandleType": "hipExternalMemoryHandleType",
    "cudaExternalMemory_t": "hipExternalMemory_t",
    "cudaExternalSemaphoreHandleDesc": "hipExternalSemaphoreHandleDesc",
    "cudaExternalSemaphoreHandleType": "hipExternalSemaphoreHandleType",
    "cudaExternalSemaphoreSignalParams": "hipExternalSemaphoreSignalParams",
    "cudaExternalSemaphoreSignalParams_v1": "hipExternalSemaphoreSignalParams",
    "cudaExternalSemaphoreWaitParams": "hipExternalSemaphoreWaitParams",
    "cudaExternalSemaphoreWaitParams_v1": "hipExternalSemaphoreWaitParams",
    "cudaExternalSemaphore_t": "hipExternalSemaphore_t",
    "cudaFuncAttribute": "hipFuncAttribute",
    "cudaFuncAttributes": "hipFuncAttributes",
    "cudaFuncCache": "hipFuncCache_t",
    "cudaFunction_t": "hipFunction_t",
    "cudaGLDeviceList": "hipGLDeviceList",
    "cudaGraphExecUpdateResult": "hipGraphExecUpdateResult",
    "cudaGraphExec_t": "hipGraphExec_t",
    "cudaGraphInstantiateFlags": "hipGraphInstantiateFlags",
    "cudaGraphMemAttributeType": "hipGraphMemAttributeType",
    "cudaGraphNodeType": "hipGraphNodeType",
    "cudaGraphNode_t": "hipGraphNode_t",
    "cudaGraph_t": "hipGraph_t",
    "cudaGraphicsRegisterFlags": "hipGraphicsRegisterFlags",
    "cudaGraphicsResource": "hipGraphicsResource",
    "cudaGraphicsResource_t": "hipGraphicsResource_t",
    "cudaHostFn_t": "hipHostFn_t",
    "cudaHostNodeParams": "hipHostNodeParams",
    "cudaIpcEventHandle_st": "hipIpcEventHandle_st",
    "cudaIpcEventHandle_t": "hipIpcEventHandle_t",
    "cudaIpcMemHandle_st": "hipIpcMemHandle_st",
    "cudaIpcMemHandle_t": "hipIpcMemHandle_t",
    "cudaKernelNodeAttrID": "hipKernelNodeAttrID",
    "cudaKernelNodeAttrValue": "hipKernelNodeAttrValue",
    "cudaKernelNodeParams": "hipKernelNodeParams",
    "cudaLaunchParams": "hipLaunchParams",
    "cudaLimit": "hipLimit_t",
    "cudaMemAccessDesc": "hipMemAccessDesc",
    "cudaMemAccessFlags": "hipMemAccessFlags",
    "cudaMemAllocationHandleType": "hipMemAllocationHandleType",
    "cudaMemAllocationType": "hipMemAllocationType",
    "cudaMemLocation": "hipMemLocation",
    "cudaMemLocationType": "hipMemLocationType",
    "cudaMemPoolAttr": "hipMemPoolAttr",
    "cudaMemPoolProps": "hipMemPoolProps",
    "cudaMemPoolPtrExportData": "hipMemPoolPtrExportData",
    "cudaMemPool_t": "hipMemPool_t",
    "cudaMemRangeAttribute": "hipMemRangeAttribute",
    "cudaMemcpy3DParms": "hipMemcpy3DParms",
    "cudaMemcpyKind": "hipMemcpyKind",
    "cudaMemoryAdvise": "hipMemoryAdvise",
    "cudaMemoryType": "hipMemoryType",
    "cudaMemsetParams": "hipMemsetParams",
    "cudaMipmappedArray": "hipMipmappedArray",
    "cudaMipmappedArray_const_t": "hipMipmappedArray_const_t",
    "cudaMipmappedArray_t": "hipMipmappedArray_t",
    "cudaPitchedPtr": "hipPitchedPtr",
    "cudaPointerAttributes": "hipPointerAttribute_t",
    "cudaPos": "hipPos",
    "cudaResourceDesc": "hipResourceDesc",
    "cudaResourceType": "hipResourceType",
    "cudaResourceViewDesc": "hipResourceViewDesc",
    "cudaResourceViewFormat": "hipResourceViewFormat",
    "cudaSharedMemConfig": "hipSharedMemConfig",
    "cudaStreamCallback_t": "hipStreamCallback_t",
    "cudaStreamCaptureMode": "hipStreamCaptureMode",
    "cudaStreamCaptureStatus": "hipStreamCaptureStatus",
    "cudaStreamUpdateCaptureDependenciesFlags": "hipStreamUpdateCaptureDependenciesFlags",
    "cudaStream_t": "hipStream_t",
    "cudaSurfaceBoundaryMode": "hipSurfaceBoundaryMode",
    "cudaSurfaceObject_t": "hipSurfaceObject_t",
    "cudaTextureAddressMode": "hipTextureAddressMode",
    "cudaTextureDesc": "hipTextureDesc",
    "cudaTextureFilterMode": "hipTextureFilterMode",
    "cudaTextureObject_t": "hipTextureObject_t",
    "cudaTextureReadMode": "hipTextureReadMode",
    "cudaUUID_t": "hipUUID",
    "cudaUserObjectFlags": "hipUserObjectFlags",
    "cudaUserObjectRetainFlags": "hipUserObjectRetainFlags",
    "cudaUserObject_t": "hipUserObject_t",
    "cudnnActivationDescriptor_t": "hipdnnActivationDescriptor_t",
    "cudnnActivationMode_t": "hipdnnActivationMode_t",
    "cudnnBatchNormMode_t": "hipdnnBatchNormMode_t",
    "cudnnConvolutionBwdDataAlgoPerfStruct": "hipdnnConvolutionBwdDataAlgoPerf_t",
    "cudnnConvolutionBwdDataAlgoPerf_t": "hipdnnConvolutionBwdDataAlgoPerf_t",
    "cudnnConvolutionBwdDataAlgo_t": "hipdnnConvolutionBwdDataAlgo_t",
    "cudnnConvolutionBwdDataPreference_t": "hipdnnConvolutionBwdDataPreference_t",
    "cudnnConvolutionBwdFilterAlgoPerfStruct": "hipdnnConvolutionBwdFilterAlgoPerf_t",
    "cudnnConvolutionBwdFilterAlgoPerf_t": "hipdnnConvolutionBwdFilterAlgoPerf_t",
    "cudnnConvolutionBwdFilterAlgo_t": "hipdnnConvolutionBwdFilterAlgo_t",
    "cudnnConvolutionBwdFilterPreference_t": "hipdnnConvolutionBwdFilterPreference_t",
    "cudnnConvolutionDescriptor_t": "hipdnnConvolutionDescriptor_t",
    "cudnnConvolutionFwdAlgoPerfStruct": "hipdnnConvolutionFwdAlgoPerf_t",
    "cudnnConvolutionFwdAlgoPerf_t": "hipdnnConvolutionFwdAlgoPerf_t",
    "cudnnConvolutionFwdAlgo_t": "hipdnnConvolutionFwdAlgo_t",
    "cudnnConvolutionFwdPreference_t": "hipdnnConvolutionFwdPreference_t",
    "cudnnConvolutionMode_t": "hipdnnConvolutionMode_t",
    "cudnnDataType_t": "hipdnnDataType_t",
    "cudnnDirectionMode_t": "hipdnnDirectionMode_t",
    "cudnnDropoutDescriptor_t": "hipdnnDropoutDescriptor_t",
    "cudnnFilterDescriptor_t": "hipdnnFilterDescriptor_t",
    "cudnnHandle_t": "hipdnnHandle_t",
    "cudnnIndicesType_t": "hipdnnIndicesType_t",
    "cudnnLRNDescriptor_t": "hipdnnLRNDescriptor_t",
    "cudnnLRNMode_t": "hipdnnLRNMode_t",
    "cudnnMathType_t": "hipdnnMathType_t",
    "cudnnNanPropagation_t": "hipdnnNanPropagation_t",
    "cudnnOpTensorDescriptor_t": "hipdnnOpTensorDescriptor_t",
    "cudnnOpTensorOp_t": "hipdnnOpTensorOp_t",
    "cudnnPersistentRNNPlan_t": "hipdnnPersistentRNNPlan_t",
    "cudnnPoolingDescriptor_t": "hipdnnPoolingDescriptor_t",
    "cudnnPoolingMode_t": "hipdnnPoolingMode_t",
    "cudnnRNNAlgo_t": "hipdnnRNNAlgo_t",
    "cudnnRNNBiasMode_t": "hipdnnRNNBiasMode_t",
    "cudnnRNNDescriptor_t": "hipdnnRNNDescriptor_t",
    "cudnnRNNInputMode_t": "hipdnnRNNInputMode_t",
    "cudnnRNNMode_t": "hipdnnRNNMode_t",
    "cudnnReduceTensorDescriptor_t": "hipdnnReduceTensorDescriptor_t",
    "cudnnReduceTensorIndices_t": "hipdnnReduceTensorIndices_t",
    "cudnnReduceTensorOp_t": "hipdnnReduceTensorOp_t",
    "cudnnSoftmaxAlgorithm_t": "hipdnnSoftmaxAlgorithm_t",
    "cudnnSoftmaxMode_t": "hipdnnSoftmaxMode_t",
    "cudnnStatus_t": "hipdnnStatus_t",
    "cudnnTensorDescriptor_t": "hipdnnTensorDescriptor_t",
    "cudnnTensorFormat_t": "hipdnnTensorFormat_t",
    "cufftComplex": "hipfftComplex",
    "cufftDoubleComplex": "hipfftDoubleComplex",
    "cufftDoubleReal": "hipfftDoubleReal",
    "cufftHandle": "hipfftHandle",
    "cufftReal": "hipfftReal",
    "cufftResult": "hipfftResult",
    "cufftResult_t": "hipfftResult_t",
    "cufftType": "hipfftType",
    "cufftType_t": "hipfftType_t",
    "cufftXtCallbackType": "hipfftXtCallbackType",
    "cufftXtCallbackType_t": "hipfftXtCallbackType_t",
    "curandDirectionVectors32_t": "hiprandDirectionVectors32_t",
    "curandDiscreteDistribution_st": "hiprandDiscreteDistribution_st",
    "curandDiscreteDistribution_t": "hiprandDiscreteDistribution_t",
    "curandGenerator_st": "hiprandGenerator_st",
    "curandGenerator_t": "hiprandGenerator_t",
    "curandRngType": "hiprandRngType_t",
    "curandRngType_t": "hiprandRngType_t",
    "curandState": "hiprandState",
    "curandStateMRG32k3a": "hiprandStateMRG32k3a",
    "curandStateMRG32k3a_t": "hiprandStateMRG32k3a_t",
    "curandStateMtgp32": "hiprandStateMtgp32",
    "curandStateMtgp32_t": "hiprandStateMtgp32_t",
    "curandStatePhilox4_32_10": "hiprandStatePhilox4_32_10",
    "curandStatePhilox4_32_10_t": "hiprandStatePhilox4_32_10_t",
    "curandStateSobol32": "hiprandStateSobol32",
    "curandStateSobol32_t": "hiprandStateSobol32_t",
    "curandStateXORWOW": "hiprandStateXORWOW",
    "curandStateXORWOW_t": "hiprandStateXORWOW_t",
    "curandState_t": "hiprandState_t",
    "curandStatus": "hiprandStatus_t",
    "curandStatus_t": "hiprandStatus_t",
    "cusparseAction_t": "hipsparseAction_t",
    "cusparseColorInfo_t": "hipsparseColorInfo_t",
    "cusparseDiagType_t": "hipsparseDiagType_t",
    "cusparseDirection_t": "hipsparseDirection_t",
    "cusparseDnMatDescr": "hipsparseDnMatDescr",
    "cusparseDnMatDescr_t": "hipsparseDnMatDescr_t",
    "cusparseDnVecDescr_t": "hipsparseDnVecDescr_t",
    "cusparseFillMode_t": "hipsparseFillMode_t",
    "cusparseFormat_t": "hipsparseFormat_t",
    "cusparseHandle_t": "hipsparseHandle_t",
    "cusparseHybMat_t": "hipsparseHybMat_t",
    "cusparseHybPartition_t": "hipsparseHybPartition_t",
    "cusparseIndexBase_t": "hipsparseIndexBase_t",
    "cusparseIndexType_t": "hipsparseIndexType_t",
    "cusparseMatDescr_t": "hipsparseMatDescr_t",
    "cusparseMatrixType_t": "hipsparseMatrixType_t",
    "cusparseOperation_t": "hipsparseOperation_t",
    "cusparseOrder_t": "hipsparseOrder_t",
    "cusparsePointerMode_t": "hipsparsePointerMode_t",
    "cusparseSDDMMAlg_t": "hipsparseSDDMMAlg_t",
    "cusparseSolvePolicy_t": "hipsparseSolvePolicy_t",
    "cusparseSpGEMMAlg_t": "hipsparseSpGEMMAlg_t",
    "cusparseSpGEMMDescr": "hipsparseSpGEMMDescr",
    "cusparseSpGEMMDescr_t": "hipsparseSpGEMMDescr_t",
    "cusparseSpMMAlg_t": "hipsparseSpMMAlg_t",
    "cusparseSpMVAlg_t": "hipsparseSpMVAlg_t",
    "cusparseSpMatAttribute_t": "hipsparseSpMatAttribute_t",
    "cusparseSpMatDescr_t": "hipsparseSpMatDescr_t",
    "cusparseSpSMAlg_t": "hipsparseSpSMAlg_t",
    "cusparseSpSMDescr": "hipsparseSpSMDescr",
    "cusparseSpSMDescr_t": "hipsparseSpSMDescr_t",
    "cusparseSpSVAlg_t": "hipsparseSpSVAlg_t",
    "cusparseSpSVDescr": "hipsparseSpSVDescr",
    "cusparseSpSVDescr_t": "hipsparseSpSVDescr_t",
    "cusparseSpVecDescr_t": "hipsparseSpVecDescr_t",
    "cusparseSparseToDenseAlg_t": "hipsparseSparseToDenseAlg_t",
    "cusparseStatus_t": "hipsparseStatus_t",
    "nvrtcProgram": "hiprtcProgram",
    "nvrtcResult": "hiprtcResult",
    "pruneInfo_t": "pruneInfo_t",
    "surfaceReference": "surfaceReference",
    "textureReference": "textureReference",
    "CUBLAS_STATUS_LICENSE_ERROR": "HIPBLAS_STATUS_UNKNOWN",
    "CUDA_ERROR_ALREADY_ACQUIRED": "hipErrorAlreadyAcquired",
    "CUDA_ERROR_ALREADY_MAPPED": "hipErrorAlreadyMapped",
    "CUDA_ERROR_ARRAY_IS_MAPPED": "hipErrorArrayIsMapped",
    "CUDA_ERROR_ASSERT": "hipErrorAssert",
    "CUDA_ERROR_CAPTURED_EVENT": "hipErrorCapturedEvent",
    "CUDA_ERROR_CONTEXT_ALREADY_CURRENT": "hipErrorContextAlreadyCurrent",
    "CUDA_ERROR_CONTEXT_ALREADY_IN_USE": "hipErrorContextAlreadyInUse",
    "CUDA_ERROR_CONTEXT_IS_DESTROYED": "hipErrorContextIsDestroyed",
    "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE": "hipErrorCooperativeLaunchTooLarge",
    "CUDA_ERROR_DEINITIALIZED": "hipErrorDeinitialized",
    "CUDA_ERROR_ECC_UNCORRECTABLE": "hipErrorECCNotCorrectable",
    "CUDA_ERROR_FILE_NOT_FOUND": "hipErrorFileNotFound",
    "CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE": "hipErrorGraphExecUpdateFailure",
    "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED": "hipErrorHostMemoryAlreadyRegistered",
    "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED": "hipErrorHostMemoryNotRegistered",
    "CUDA_ERROR_ILLEGAL_ADDRESS": "hipErrorIllegalAddress",
    "CUDA_ERROR_ILLEGAL_STATE": "hipErrorIllegalState",
    "CUDA_ERROR_INVALID_CONTEXT": "hipErrorInvalidContext",
    "CUDA_ERROR_INVALID_DEVICE": "hipErrorInvalidDevice",
    "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT": "hipErrorInvalidGraphicsContext",
    "CUDA_ERROR_INVALID_HANDLE": "hipErrorInvalidHandle",
    "CUDA_ERROR_INVALID_IMAGE": "hipErrorInvalidImage",
    "CUDA_ERROR_INVALID_PTX": "hipErrorInvalidKernelFile",
    "CUDA_ERROR_INVALID_SOURCE": "hipErrorInvalidSource",
    "CUDA_ERROR_INVALID_VALUE": "hipErrorInvalidValue",
    "CUDA_ERROR_LAUNCH_FAILED": "hipErrorLaunchFailure",
    "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES": "hipErrorLaunchOutOfResources",
    "CUDA_ERROR_LAUNCH_TIMEOUT": "hipErrorLaunchTimeOut",
    "CUDA_ERROR_MAP_FAILED": "hipErrorMapFailed",
    "CUDA_ERROR_NOT_FOUND": "hipErrorNotFound",
    "CUDA_ERROR_NOT_INITIALIZED": "hipErrorNotInitialized",
    "CUDA_ERROR_NOT_MAPPED": "hipErrorNotMapped",
    "CUDA_ERROR_NOT_MAPPED_AS_ARRAY": "hipErrorNotMappedAsArray",
    "CUDA_ERROR_NOT_MAPPED_AS_POINTER": "hipErrorNotMappedAsPointer",
    "CUDA_ERROR_NOT_READY": "hipErrorNotReady",
    "CUDA_ERROR_NOT_SUPPORTED": "hipErrorNotSupported",
    "CUDA_ERROR_NO_BINARY_FOR_GPU": "hipErrorNoBinaryForGpu",
    "CUDA_ERROR_NO_DEVICE": "hipErrorNoDevice",
    "CUDA_ERROR_OPERATING_SYSTEM": "hipErrorOperatingSystem",
    "CUDA_ERROR_OUT_OF_MEMORY": "hipErrorOutOfMemory",
    "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED": "hipErrorPeerAccessAlreadyEnabled",
    "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED": "hipErrorPeerAccessNotEnabled",
    "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED": "hipErrorPeerAccessUnsupported",
    "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE": "hipErrorSetOnActiveProcess",
    "CUDA_ERROR_PROFILER_ALREADY_STARTED": "hipErrorProfilerAlreadyStarted",
    "CUDA_ERROR_PROFILER_ALREADY_STOPPED": "hipErrorProfilerAlreadyStopped",
    "CUDA_ERROR_PROFILER_DISABLED": "hipErrorProfilerDisabled",
    "CUDA_ERROR_PROFILER_NOT_INITIALIZED": "hipErrorProfilerNotInitialized",
    "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED": "hipErrorSharedObjectInitFailed",
    "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND": "hipErrorSharedObjectSymbolNotFound",
    "CUDA_ERROR_STREAM_CAPTURE_IMPLICIT": "hipErrorStreamCaptureImplicit",
    "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED": "hipErrorStreamCaptureInvalidated",
    "CUDA_ERROR_STREAM_CAPTURE_ISOLATION": "hipErrorStreamCaptureIsolation",
    "CUDA_ERROR_STREAM_CAPTURE_MERGE": "hipErrorStreamCaptureMerge",
    "CUDA_ERROR_STREAM_CAPTURE_UNJOINED": "hipErrorStreamCaptureUnjoined",
    "CUDA_ERROR_STREAM_CAPTURE_UNMATCHED": "hipErrorStreamCaptureUnmatched",
    "CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED": "hipErrorStreamCaptureUnsupported",
    "CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD": "hipErrorStreamCaptureWrongThread",
    "CUDA_ERROR_UNKNOWN": "hipErrorUnknown",
    "CUDA_ERROR_UNMAP_FAILED": "hipErrorUnmapFailed",
    "CUDA_ERROR_UNSUPPORTED_LIMIT": "hipErrorUnsupportedLimit",
    "CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH": "hipGraphInstantiateFlagAutoFreeOnLaunch",
    "CUDA_SUCCESS": "hipSuccess",
    "CUDNN_16BIT_INDICES": "HIPDNN_16BIT_INDICES",
    "CUDNN_32BIT_INDICES": "HIPDNN_32BIT_INDICES",
    "CUDNN_64BIT_INDICES": "HIPDNN_64BIT_INDICES",
    "CUDNN_8BIT_INDICES": "HIPDNN_8BIT_INDICES",
    "CUDNN_ACTIVATION_CLIPPED_RELU": "HIPDNN_ACTIVATION_CLIPPED_RELU",
    "CUDNN_ACTIVATION_ELU": "HIPDNN_ACTIVATION_ELU",
    "CUDNN_ACTIVATION_IDENTITY": "HIPDNN_ACTIVATION_PATHTRU",
    "CUDNN_ACTIVATION_RELU": "HIPDNN_ACTIVATION_RELU",
    "CUDNN_ACTIVATION_SIGMOID": "HIPDNN_ACTIVATION_SIGMOID",
    "CUDNN_ACTIVATION_SWISH": "HIPDNN_ACTIVATION_SWISH",
    "CUDNN_ACTIVATION_TANH": "HIPDNN_ACTIVATION_TANH",
    "CUDNN_BATCHNORM_PER_ACTIVATION": "HIPDNN_BATCHNORM_PER_ACTIVATION",
    "CUDNN_BATCHNORM_SPATIAL": "HIPDNN_BATCHNORM_SPATIAL",
    "CUDNN_BATCHNORM_SPATIAL_PERSISTENT": "HIPDNN_BATCHNORM_SPATIAL_PERSISTENT",
    "CUDNN_BIDIRECTIONAL": "HIPDNN_BIDIRECTIONAL",
    "CUDNN_BN_MIN_EPSILON": "HIPDNN_BN_MIN_EPSILON",
    "CUDNN_CONVOLUTION": "HIPDNN_CONVOLUTION",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_0": "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_0",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1": "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT": "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_TRANSPOSE_GEMM",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT": "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING": "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD": "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED": "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED",
    "CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE": "HIPDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE",
    "CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST": "HIPDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST",
    "CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT": "HIPDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0": "HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1": "HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3": "HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_3",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT": "HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT": "HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING": "HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD": "HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED": "HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED",
    "CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE": "HIPDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE",
    "CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST": "HIPDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST",
    "CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT": "HIPDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT",
    "CUDNN_CONVOLUTION_FWD_ALGO_COUNT": "HIPDNN_CONVOLUTION_FWD_ALGO_COUNT",
    "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT": "HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT",
    "CUDNN_CONVOLUTION_FWD_ALGO_FFT": "HIPDNN_CONVOLUTION_FWD_ALGO_FFT",
    "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING": "HIPDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
    "CUDNN_CONVOLUTION_FWD_ALGO_GEMM": "HIPDNN_CONVOLUTION_FWD_ALGO_GEMM",
    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM": "HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM": "HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
    "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD": "HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
    "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED": "HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED",
    "CUDNN_CONVOLUTION_FWD_NO_WORKSPACE": "HIPDNN_CONVOLUTION_FWD_NO_WORKSPACE",
    "CUDNN_CONVOLUTION_FWD_PREFER_FASTEST": "HIPDNN_CONVOLUTION_FWD_PREFER_FASTEST",
    "CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT": "HIPDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT",
    "CUDNN_CROSS_CORRELATION": "HIPDNN_CROSS_CORRELATION",
    "CUDNN_DATA_DOUBLE": "HIPDNN_DATA_DOUBLE",
    "CUDNN_DATA_FLOAT": "HIPDNN_DATA_FLOAT",
    "CUDNN_DATA_HALF": "HIPDNN_DATA_HALF",
    "CUDNN_DATA_INT32": "HIPDNN_DATA_INT32",
    "CUDNN_DATA_INT8": "HIPDNN_DATA_INT8",
    "CUDNN_DATA_INT8x4": "HIPDNN_DATA_INT8x4",
    "CUDNN_DEFAULT_MATH": "HIPDNN_DEFAULT_MATH",
    "CUDNN_GRU": "HIPDNN_GRU",
    "CUDNN_LINEAR_INPUT": "HIPDNN_LINEAR_INPUT",
    "CUDNN_LRN_CROSS_CHANNEL_DIM1": "HIPDNN_LRN_CROSS_CHANNEL",
    "CUDNN_LSTM": "HIPDNN_LSTM",
    "CUDNN_NOT_PROPAGATE_NAN": "HIPDNN_NOT_PROPAGATE_NAN",
    "CUDNN_OP_TENSOR_ADD": "HIPDNN_OP_TENSOR_ADD",
    "CUDNN_OP_TENSOR_MAX": "HIPDNN_OP_TENSOR_MAX",
    "CUDNN_OP_TENSOR_MIN": "HIPDNN_OP_TENSOR_MIN",
    "CUDNN_OP_TENSOR_MUL": "HIPDNN_OP_TENSOR_MUL",
    "CUDNN_OP_TENSOR_SQRT": "HIPDNN_OP_TENSOR_SQRT",
    "CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING": "HIPDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING",
    "CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING": "HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING",
    "CUDNN_POOLING_MAX": "HIPDNN_POOLING_MAX",
    "CUDNN_POOLING_MAX_DETERMINISTIC": "HIPDNN_POOLING_MAX_DETERMINISTIC",
    "CUDNN_PROPAGATE_NAN": "HIPDNN_PROPAGATE_NAN",
    "CUDNN_REDUCE_TENSOR_ADD": "HIPDNN_REDUCE_TENSOR_ADD",
    "CUDNN_REDUCE_TENSOR_AMAX": "HIPDNN_REDUCE_TENSOR_AMAX",
    "CUDNN_REDUCE_TENSOR_AVG": "HIPDNN_REDUCE_TENSOR_AVG",
    "CUDNN_REDUCE_TENSOR_FLATTENED_INDICES": "HIPDNN_REDUCE_TENSOR_FLATTENED_INDICES",
    "CUDNN_REDUCE_TENSOR_MAX": "HIPDNN_REDUCE_TENSOR_MAX",
    "CUDNN_REDUCE_TENSOR_MIN": "HIPDNN_REDUCE_TENSOR_MIN",
    "CUDNN_REDUCE_TENSOR_MUL": "HIPDNN_REDUCE_TENSOR_MUL",
    "CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS": "HIPDNN_REDUCE_TENSOR_MUL_NO_ZEROS",
    "CUDNN_REDUCE_TENSOR_NORM1": "HIPDNN_REDUCE_TENSOR_NORM1",
    "CUDNN_REDUCE_TENSOR_NORM2": "HIPDNN_REDUCE_TENSOR_NORM2",
    "CUDNN_REDUCE_TENSOR_NO_INDICES": "HIPDNN_REDUCE_TENSOR_NO_INDICES",
    "CUDNN_RNN_ALGO_PERSIST_DYNAMIC": "HIPDNN_RNN_ALGO_PERSIST_DYNAMIC",
    "CUDNN_RNN_ALGO_PERSIST_STATIC": "HIPDNN_RNN_ALGO_PERSIST_STATIC",
    "CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H": "HIPDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H",
    "CUDNN_RNN_ALGO_STANDARD": "HIPDNN_RNN_ALGO_STANDARD",
    "CUDNN_RNN_DOUBLE_BIAS": "HIPDNN_RNN_WITH_BIAS",
    "CUDNN_RNN_NO_BIAS": "HIPDNN_RNN_NO_BIAS",
    "CUDNN_RNN_RELU": "HIPDNN_RNN_RELU",
    "CUDNN_RNN_SINGLE_INP_BIAS": "HIPDNN_RNN_WITH_BIAS",
    "CUDNN_RNN_SINGLE_REC_BIAS": "HIPDNN_RNN_WITH_BIAS",
    "CUDNN_RNN_TANH": "HIPDNN_RNN_TANH",
    "CUDNN_SKIP_INPUT": "HIPDNN_SKIP_INPUT",
    "CUDNN_SOFTMAX_ACCURATE": "HIPDNN_SOFTMAX_ACCURATE",
    "CUDNN_SOFTMAX_FAST": "HIPDNN_SOFTMAX_FAST",
    "CUDNN_SOFTMAX_LOG": "HIPDNN_SOFTMAX_LOG",
    "CUDNN_SOFTMAX_MODE_CHANNEL": "HIPDNN_SOFTMAX_MODE_CHANNEL",
    "CUDNN_SOFTMAX_MODE_INSTANCE": "HIPDNN_SOFTMAX_MODE_INSTANCE",
    "CUDNN_STATUS_ALLOC_FAILED": "HIPDNN_STATUS_ALLOC_FAILED",
    "CUDNN_STATUS_ARCH_MISMATCH": "HIPDNN_STATUS_ARCH_MISMATCH",
    "CUDNN_STATUS_BAD_PARAM": "HIPDNN_STATUS_BAD_PARAM",
    "CUDNN_STATUS_EXECUTION_FAILED": "HIPDNN_STATUS_EXECUTION_FAILED",
    "CUDNN_STATUS_INTERNAL_ERROR": "HIPDNN_STATUS_INTERNAL_ERROR",
    "CUDNN_STATUS_INVALID_VALUE": "HIPDNN_STATUS_INVALID_VALUE",
    "CUDNN_STATUS_LICENSE_ERROR": "HIPDNN_STATUS_LICENSE_ERROR",
    "CUDNN_STATUS_MAPPING_ERROR": "HIPDNN_STATUS_MAPPING_ERROR",
    "CUDNN_STATUS_NOT_INITIALIZED": "HIPDNN_STATUS_NOT_INITIALIZED",
    "CUDNN_STATUS_NOT_SUPPORTED": "HIPDNN_STATUS_NOT_SUPPORTED",
    "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING": "HIPDNN_STATUS_RUNTIME_PREREQUISITE_MISSING",
    "CUDNN_STATUS_SUCCESS": "HIPDNN_STATUS_SUCCESS",
    "CUDNN_TENSOR_NCHW": "HIPDNN_TENSOR_NCHW",
    "CUDNN_TENSOR_NCHW_VECT_C": "HIPDNN_TENSOR_NCHW_VECT_C",
    "CUDNN_TENSOR_NHWC": "HIPDNN_TENSOR_NHWC",
    "CUDNN_TENSOR_OP_MATH": "HIPDNN_TENSOR_OP_MATH",
    "CUDNN_UNIDIRECTIONAL": "HIPDNN_UNIDIRECTIONAL",
    "CUDNN_VERSION": "HIPDNN_VERSION",
    "CUFFT_ALLOC_FAILED": "HIPFFT_ALLOC_FAILED",
    "CUFFT_C2C": "HIPFFT_C2C",
    "CUFFT_C2R": "HIPFFT_C2R",
    "CUFFT_CB_LD_COMPLEX": "HIPFFT_CB_LD_COMPLEX",
    "CUFFT_CB_LD_COMPLEX_DOUBLE": "HIPFFT_CB_LD_COMPLEX_DOUBLE",
    "CUFFT_CB_LD_REAL": "HIPFFT_CB_LD_REAL",
    "CUFFT_CB_LD_REAL_DOUBLE": "HIPFFT_CB_LD_REAL_DOUBLE",
    "CUFFT_CB_ST_COMPLEX": "HIPFFT_CB_ST_COMPLEX",
    "CUFFT_CB_ST_COMPLEX_DOUBLE": "HIPFFT_CB_ST_COMPLEX_DOUBLE",
    "CUFFT_CB_ST_REAL": "HIPFFT_CB_ST_REAL",
    "CUFFT_CB_ST_REAL_DOUBLE": "HIPFFT_CB_ST_REAL_DOUBLE",
    "CUFFT_CB_UNDEFINED": "HIPFFT_CB_UNDEFINED",
    "CUFFT_D2Z": "HIPFFT_D2Z",
    "CUFFT_EXEC_FAILED": "HIPFFT_EXEC_FAILED",
    "CUFFT_FORWARD": "HIPFFT_FORWARD",
    "CUFFT_INCOMPLETE_PARAMETER_LIST": "HIPFFT_INCOMPLETE_PARAMETER_LIST",
    "CUFFT_INTERNAL_ERROR": "HIPFFT_INTERNAL_ERROR",
    "CUFFT_INVALID_DEVICE": "HIPFFT_INVALID_DEVICE",
    "CUFFT_INVALID_PLAN": "HIPFFT_INVALID_PLAN",
    "CUFFT_INVALID_SIZE": "HIPFFT_INVALID_SIZE",
    "CUFFT_INVALID_TYPE": "HIPFFT_INVALID_TYPE",
    "CUFFT_INVALID_VALUE": "HIPFFT_INVALID_VALUE",
    "CUFFT_INVERSE": "HIPFFT_BACKWARD",
    "CUFFT_NOT_IMPLEMENTED": "HIPFFT_NOT_IMPLEMENTED",
    "CUFFT_NOT_SUPPORTED": "HIPFFT_NOT_SUPPORTED",
    "CUFFT_NO_WORKSPACE": "HIPFFT_NO_WORKSPACE",
    "CUFFT_PARSE_ERROR": "HIPFFT_PARSE_ERROR",
    "CUFFT_R2C": "HIPFFT_R2C",
    "CUFFT_SETUP_FAILED": "HIPFFT_SETUP_FAILED",
    "CUFFT_SUCCESS": "HIPFFT_SUCCESS",
    "CUFFT_UNALIGNED_DATA": "HIPFFT_UNALIGNED_DATA",
    "CUFFT_Z2D": "HIPFFT_Z2D",
    "CUFFT_Z2Z": "HIPFFT_Z2Z",
    "CURAND_RNG_PSEUDO_DEFAULT": "HIPRAND_RNG_PSEUDO_DEFAULT",
    "CURAND_RNG_PSEUDO_MRG32K3A": "HIPRAND_RNG_PSEUDO_MRG32K3A",
    "CURAND_RNG_PSEUDO_MT19937": "HIPRAND_RNG_PSEUDO_MT19937",
    "CURAND_RNG_PSEUDO_MTGP32": "HIPRAND_RNG_PSEUDO_MTGP32",
    "CURAND_RNG_PSEUDO_PHILOX4_32_10": "HIPRAND_RNG_PSEUDO_PHILOX4_32_10",
    "CURAND_RNG_PSEUDO_XORWOW": "HIPRAND_RNG_PSEUDO_XORWOW",
    "CURAND_RNG_QUASI_DEFAULT": "HIPRAND_RNG_QUASI_DEFAULT",
    "CURAND_RNG_QUASI_SCRAMBLED_SOBOL32": "HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32",
    "CURAND_RNG_QUASI_SCRAMBLED_SOBOL64": "HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64",
    "CURAND_RNG_QUASI_SOBOL32": "HIPRAND_RNG_QUASI_SOBOL32",
    "CURAND_RNG_QUASI_SOBOL64": "HIPRAND_RNG_QUASI_SOBOL64",
    "CURAND_RNG_TEST": "HIPRAND_RNG_TEST",
    "CURAND_STATUS_ALLOCATION_FAILED": "HIPRAND_STATUS_ALLOCATION_FAILED",
    "CURAND_STATUS_ARCH_MISMATCH": "HIPRAND_STATUS_ARCH_MISMATCH",
    "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED": "HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED",
    "CURAND_STATUS_INITIALIZATION_FAILED": "HIPRAND_STATUS_INITIALIZATION_FAILED",
    "CURAND_STATUS_INTERNAL_ERROR": "HIPRAND_STATUS_INTERNAL_ERROR",
    "CURAND_STATUS_LAUNCH_FAILURE": "HIPRAND_STATUS_LAUNCH_FAILURE",
    "CURAND_STATUS_LENGTH_NOT_MULTIPLE": "HIPRAND_STATUS_LENGTH_NOT_MULTIPLE",
    "CURAND_STATUS_NOT_INITIALIZED": "HIPRAND_STATUS_NOT_INITIALIZED",
    "CURAND_STATUS_OUT_OF_RANGE": "HIPRAND_STATUS_OUT_OF_RANGE",
    "CURAND_STATUS_PREEXISTING_FAILURE": "HIPRAND_STATUS_PREEXISTING_FAILURE",
    "CURAND_STATUS_SUCCESS": "HIPRAND_STATUS_SUCCESS",
    "CURAND_STATUS_TYPE_ERROR": "HIPRAND_STATUS_TYPE_ERROR",
    "CURAND_STATUS_VERSION_MISMATCH": "HIPRAND_STATUS_VERSION_MISMATCH",
    "CUSPARSE_ACTION_NUMERIC": "HIPSPARSE_ACTION_NUMERIC",
    "CUSPARSE_ACTION_SYMBOLIC": "HIPSPARSE_ACTION_SYMBOLIC",
    "CUSPARSE_COOMM_ALG1": "HIPSPARSE_COOMM_ALG1",
    "CUSPARSE_COOMM_ALG2": "HIPSPARSE_COOMM_ALG2",
    "CUSPARSE_COOMM_ALG3": "HIPSPARSE_COOMM_ALG3",
    "CUSPARSE_COOMV_ALG": "HIPSPARSE_COOMV_ALG",
    "CUSPARSE_CSRMM_ALG1": "HIPSPARSE_CSRMM_ALG1",
    "CUSPARSE_CSRMV_ALG1": "HIPSPARSE_CSRMV_ALG1",
    "CUSPARSE_CSRMV_ALG2": "HIPSPARSE_CSRMV_ALG2",
    "CUSPARSE_DIAG_TYPE_NON_UNIT": "HIPSPARSE_DIAG_TYPE_NON_UNIT",
    "CUSPARSE_DIAG_TYPE_UNIT": "HIPSPARSE_DIAG_TYPE_UNIT",
    "CUSPARSE_DIRECTION_COLUMN": "HIPSPARSE_DIRECTION_COLUMN",
    "CUSPARSE_DIRECTION_ROW": "HIPSPARSE_DIRECTION_ROW",
    "CUSPARSE_FILL_MODE_LOWER": "HIPSPARSE_FILL_MODE_LOWER",
    "CUSPARSE_FILL_MODE_UPPER": "HIPSPARSE_FILL_MODE_UPPER",
    "CUSPARSE_FORMAT_BLOCKED_ELL": "HIPSPARSE_FORMAT_BLOCKED_ELL",
    "CUSPARSE_FORMAT_COO": "HIPSPARSE_FORMAT_COO",
    "CUSPARSE_FORMAT_COO_AOS": "HIPSPARSE_FORMAT_COO_AOS",
    "CUSPARSE_FORMAT_CSC": "HIPSPARSE_FORMAT_CSC",
    "CUSPARSE_FORMAT_CSR": "HIPSPARSE_FORMAT_CSR",
    "CUSPARSE_HYB_PARTITION_AUTO": "HIPSPARSE_HYB_PARTITION_AUTO",
    "CUSPARSE_HYB_PARTITION_MAX": "HIPSPARSE_HYB_PARTITION_MAX",
    "CUSPARSE_HYB_PARTITION_USER": "HIPSPARSE_HYB_PARTITION_USER",
    "CUSPARSE_INDEX_16U": "HIPSPARSE_INDEX_16U",
    "CUSPARSE_INDEX_32I": "HIPSPARSE_INDEX_32I",
    "CUSPARSE_INDEX_64I": "HIPSPARSE_INDEX_64I",
    "CUSPARSE_INDEX_BASE_ONE": "HIPSPARSE_INDEX_BASE_ONE",
    "CUSPARSE_INDEX_BASE_ZERO": "HIPSPARSE_INDEX_BASE_ZERO",
    "CUSPARSE_MATRIX_TYPE_GENERAL": "HIPSPARSE_MATRIX_TYPE_GENERAL",
    "CUSPARSE_MATRIX_TYPE_HERMITIAN": "HIPSPARSE_MATRIX_TYPE_HERMITIAN",
    "CUSPARSE_MATRIX_TYPE_SYMMETRIC": "HIPSPARSE_MATRIX_TYPE_SYMMETRIC",
    "CUSPARSE_MATRIX_TYPE_TRIANGULAR": "HIPSPARSE_MATRIX_TYPE_TRIANGULAR",
    "CUSPARSE_MM_ALG_DEFAULT": "HIPSPARSE_MM_ALG_DEFAULT",
    "CUSPARSE_MV_ALG_DEFAULT": "HIPSPARSE_MV_ALG_DEFAULT",
    "CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE": "HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE",
    "CUSPARSE_OPERATION_NON_TRANSPOSE": "HIPSPARSE_OPERATION_NON_TRANSPOSE",
    "CUSPARSE_OPERATION_TRANSPOSE": "HIPSPARSE_OPERATION_TRANSPOSE",
    "CUSPARSE_ORDER_COL": "HIPSPARSE_ORDER_COL",
    "CUSPARSE_ORDER_ROW": "HIPSPARSE_ORDER_ROW",
    "CUSPARSE_POINTER_MODE_DEVICE": "HIPSPARSE_POINTER_MODE_DEVICE",
    "CUSPARSE_POINTER_MODE_HOST": "HIPSPARSE_POINTER_MODE_HOST",
    "CUSPARSE_SDDMM_ALG_DEFAULT": "HIPSPARSE_SDDMM_ALG_DEFAULT",
    "CUSPARSE_SOLVE_POLICY_NO_LEVEL": "HIPSPARSE_SOLVE_POLICY_NO_LEVEL",
    "CUSPARSE_SOLVE_POLICY_USE_LEVEL": "HIPSPARSE_SOLVE_POLICY_USE_LEVEL",
    "CUSPARSE_SPARSETODENSE_ALG_DEFAULT": "HIPSPARSE_SPARSETODENSE_ALG_DEFAULT",
    "CUSPARSE_SPGEMM_DEFAULT": "HIPSPARSE_SPGEMM_DEFAULT",
    "CUSPARSE_SPMAT_DIAG_TYPE": "HIPSPARSE_SPMAT_DIAG_TYPE",
    "CUSPARSE_SPMAT_FILL_MODE": "HIPSPARSE_SPMAT_FILL_MODE",
    "CUSPARSE_SPMM_ALG_DEFAULT": "HIPSPARSE_SPMM_ALG_DEFAULT",
    "CUSPARSE_SPMM_BLOCKED_ELL_ALG1": "HIPSPARSE_SPMM_BLOCKED_ELL_ALG1",
    "CUSPARSE_SPMM_COO_ALG1": "HIPSPARSE_SPMM_COO_ALG1",
    "CUSPARSE_SPMM_COO_ALG2": "HIPSPARSE_SPMM_COO_ALG2",
    "CUSPARSE_SPMM_COO_ALG3": "HIPSPARSE_SPMM_COO_ALG3",
    "CUSPARSE_SPMM_COO_ALG4": "HIPSPARSE_SPMM_COO_ALG4",
    "CUSPARSE_SPMM_CSR_ALG1": "HIPSPARSE_SPMM_CSR_ALG1",
    "CUSPARSE_SPMM_CSR_ALG2": "HIPSPARSE_SPMM_CSR_ALG2",
    "CUSPARSE_SPMM_CSR_ALG3": "HIPSPARSE_SPMM_CSR_ALG3",
    "CUSPARSE_SPMV_ALG_DEFAULT": "HIPSPARSE_SPMV_ALG_DEFAULT",
    "CUSPARSE_SPMV_COO_ALG1": "HIPSPARSE_SPMV_COO_ALG1",
    "CUSPARSE_SPMV_COO_ALG2": "HIPSPARSE_SPMV_COO_ALG2",
    "CUSPARSE_SPMV_CSR_ALG1": "HIPSPARSE_SPMV_CSR_ALG1",
    "CUSPARSE_SPMV_CSR_ALG2": "HIPSPARSE_SPMV_CSR_ALG2",
    "CUSPARSE_SPSM_ALG_DEFAULT": "HIPSPARSE_SPSM_ALG_DEFAULT",
    "CUSPARSE_SPSV_ALG_DEFAULT": "HIPSPARSE_SPSV_ALG_DEFAULT",
    "CUSPARSE_STATUS_ALLOC_FAILED": "HIPSPARSE_STATUS_ALLOC_FAILED",
    "CUSPARSE_STATUS_ARCH_MISMATCH": "HIPSPARSE_STATUS_ARCH_MISMATCH",
    "CUSPARSE_STATUS_EXECUTION_FAILED": "HIPSPARSE_STATUS_EXECUTION_FAILED",
    "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES": "HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES",
    "CUSPARSE_STATUS_INTERNAL_ERROR": "HIPSPARSE_STATUS_INTERNAL_ERROR",
    "CUSPARSE_STATUS_INVALID_VALUE": "HIPSPARSE_STATUS_INVALID_VALUE",
    "CUSPARSE_STATUS_MAPPING_ERROR": "HIPSPARSE_STATUS_MAPPING_ERROR",
    "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED": "HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED",
    "CUSPARSE_STATUS_NOT_INITIALIZED": "HIPSPARSE_STATUS_NOT_INITIALIZED",
    "CUSPARSE_STATUS_NOT_SUPPORTED": "HIPSPARSE_STATUS_NOT_SUPPORTED",
    "CUSPARSE_STATUS_SUCCESS": "HIPSPARSE_STATUS_SUCCESS",
    "CUSPARSE_STATUS_ZERO_PIVOT": "HIPSPARSE_STATUS_ZERO_PIVOT",
    "CU_ACCESS_PROPERTY_NORMAL": "hipAccessPropertyNormal",
    "CU_ACCESS_PROPERTY_PERSISTING": "hipAccessPropertyPersisting",
    "CU_ACCESS_PROPERTY_STREAMING": "hipAccessPropertyStreaming",
    "CU_AD_FORMAT_FLOAT": "HIP_AD_FORMAT_FLOAT",
    "CU_AD_FORMAT_HALF": "HIP_AD_FORMAT_HALF",
    "CU_AD_FORMAT_SIGNED_INT16": "HIP_AD_FORMAT_SIGNED_INT16",
    "CU_AD_FORMAT_SIGNED_INT32": "HIP_AD_FORMAT_SIGNED_INT32",
    "CU_AD_FORMAT_SIGNED_INT8": "HIP_AD_FORMAT_SIGNED_INT8",
    "CU_AD_FORMAT_UNSIGNED_INT16": "HIP_AD_FORMAT_UNSIGNED_INT16",
    "CU_AD_FORMAT_UNSIGNED_INT32": "HIP_AD_FORMAT_UNSIGNED_INT32",
    "CU_AD_FORMAT_UNSIGNED_INT8": "HIP_AD_FORMAT_UNSIGNED_INT8",
    "CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL": "hipArraySparseSubresourceTypeMiptail",
    "CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL": "hipArraySparseSubresourceTypeSparseLevel",
    "CU_COMPUTEMODE_DEFAULT": "hipComputeModeDefault",
    "CU_COMPUTEMODE_EXCLUSIVE": "hipComputeModeExclusive",
    "CU_COMPUTEMODE_EXCLUSIVE_PROCESS": "hipComputeModeExclusiveProcess",
    "CU_COMPUTEMODE_PROHIBITED": "hipComputeModeProhibited",
    "CU_CTX_BLOCKING_SYNC": "hipDeviceScheduleBlockingSync",
    "CU_CTX_LMEM_RESIZE_TO_MAX": "hipDeviceLmemResizeToMax",
    "CU_CTX_MAP_HOST": "hipDeviceMapHost",
    "CU_CTX_SCHED_AUTO": "hipDeviceScheduleAuto",
    "CU_CTX_SCHED_BLOCKING_SYNC": "hipDeviceScheduleBlockingSync",
    "CU_CTX_SCHED_MASK": "hipDeviceScheduleMask",
    "CU_CTX_SCHED_SPIN": "hipDeviceScheduleSpin",
    "CU_CTX_SCHED_YIELD": "hipDeviceScheduleYield",
    "CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT": "hipDeviceAttributeAsyncEngineCount",
    "CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY": "hipDeviceAttributeCanMapHostMemory",
    "CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM": "hipDeviceAttributeCanUseHostPointerForRegisteredMem",
    "CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR": "hipDeviceAttributeCanUseStreamWaitValue",
    "CU_DEVICE_ATTRIBUTE_CLOCK_RATE": "hipDeviceAttributeClockRate",
    "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR": "hipDeviceAttributeComputeCapabilityMajor",
    "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR": "hipDeviceAttributeComputeCapabilityMinor",
    "CU_DEVICE_ATTRIBUTE_COMPUTE_MODE": "hipDeviceAttributeComputeMode",
    "CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED": "hipDeviceAttributeComputePreemptionSupported",
    "CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS": "hipDeviceAttributeConcurrentKernels",
    "CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS": "hipDeviceAttributeConcurrentManagedAccess",
    "CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH": "hipDeviceAttributeCooperativeLaunch",
    "CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH": "hipDeviceAttributeCooperativeMultiDeviceLaunch",
    "CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST": "hipDeviceAttributeDirectManagedMemAccessFromHost",
    "CU_DEVICE_ATTRIBUTE_ECC_ENABLED": "hipDeviceAttributeEccEnabled",
    "CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED": "hipDeviceAttributeGlobalL1CacheSupported",
    "CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH": "hipDeviceAttributeMemoryBusWidth",
    "CU_DEVICE_ATTRIBUTE_GPU_OVERLAP": "hipDeviceAttributeAsyncEngineCount",
    "CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED": "hipDeviceAttributeHostNativeAtomicSupported",
    "CU_DEVICE_ATTRIBUTE_INTEGRATED": "hipDeviceAttributeIntegrated",
    "CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT": "hipDeviceAttributeKernelExecTimeout",
    "CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE": "hipDeviceAttributeL2CacheSize",
    "CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED": "hipDeviceAttributeLocalL1CacheSupported",
    "CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY": "hipDeviceAttributeManagedMemory",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH": "hipDeviceAttributeMaxSurface1DLayered",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH": "hipDeviceAttributeMaxSurface1D",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT": "hipDeviceAttributeMaxSurface2D",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT": "hipDeviceAttributeMaxSurface2DLayered",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH": "hipDeviceAttributeMaxSurface2DLayered",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH": "hipDeviceAttributeMaxSurface2D",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH": "hipDeviceAttributeMaxSurface3D",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT": "hipDeviceAttributeMaxSurface3D",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH": "hipDeviceAttributeMaxSurface3D",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH": "hipDeviceAttributeMaxSurfaceCubemapLayered",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH": "hipDeviceAttributeMaxSurfaceCubemap",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH": "hipDeviceAttributeMaxTexture1DLayered",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH": "hipDeviceAttributeMaxTexture1DLinear",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH": "hipDeviceAttributeMaxTexture1DMipmap",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH": "hipDeviceAttributeMaxTexture1DWidth",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT": "hipDeviceAttributeMaxTexture2DLayered",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH": "hipDeviceAttributeMaxTexture2DLayered",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT": "hipDeviceAttributeMaxTexture2DGather",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH": "hipDeviceAttributeMaxTexture2DGather",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT": "hipDeviceAttributeMaxTexture2DHeight",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT": "hipDeviceAttributeMaxTexture2DLayered",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH": "hipDeviceAttributeMaxTexture2DLayered",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT": "hipDeviceAttributeMaxTexture2DLinear",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH": "hipDeviceAttributeMaxTexture2DLinear",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH": "hipDeviceAttributeMaxTexture2DLinear",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT": "hipDeviceAttributeMaxTexture2DMipmap",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH": "hipDeviceAttributeMaxTexture2DMipmap",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH": "hipDeviceAttributeMaxTexture2DWidth",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH": "hipDeviceAttributeMaxTexture3DDepth",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE": "hipDeviceAttributeMaxTexture3DAlt",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT": "hipDeviceAttributeMaxTexture3DHeight",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE": "hipDeviceAttributeMaxTexture3DAlt",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH": "hipDeviceAttributeMaxTexture3DWidth",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE": "hipDeviceAttributeMaxTexture3DAlt",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH": "hipDeviceAttributeMaxTextureCubemapLayered",
    "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH": "hipDeviceAttributeMaxTextureCubemap",
    "CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR": "hipDeviceAttributeMaxBlocksPerMultiprocessor",
    "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X": "hipDeviceAttributeMaxBlockDimX",
    "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y": "hipDeviceAttributeMaxBlockDimY",
    "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z": "hipDeviceAttributeMaxBlockDimZ",
    "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X": "hipDeviceAttributeMaxGridDimX",
    "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y": "hipDeviceAttributeMaxGridDimY",
    "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z": "hipDeviceAttributeMaxGridDimZ",
    "CU_DEVICE_ATTRIBUTE_MAX_PITCH": "hipDeviceAttributeMaxPitch",
    "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK": "hipDeviceAttributeMaxRegistersPerBlock",
    "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR": "hipDeviceAttributeMaxRegistersPerMultiprocessor",
    "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK": "hipDeviceAttributeMaxSharedMemoryPerBlock",
    "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN": "hipDeviceAttributeSharedMemPerBlockOptin",
    "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR": "hipDeviceAttributeMaxSharedMemoryPerMultiprocessor",
    "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK": "hipDeviceAttributeMaxThreadsPerBlock",
    "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR": "hipDeviceAttributeMaxThreadsPerMultiProcessor",
    "CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE": "hipDeviceAttributeMemoryClockRate",
    "CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED": "hipDeviceAttributeMemoryPoolsSupported",
    "CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT": "hipDeviceAttributeMultiprocessorCount",
    "CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD": "hipDeviceAttributeIsMultiGpuBoard",
    "CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID": "hipDeviceAttributeMultiGpuBoardGroupId",
    "CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS": "hipDeviceAttributePageableMemoryAccess",
    "CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES": "hipDeviceAttributePageableMemoryAccessUsesHostPageTables",
    "CU_DEVICE_ATTRIBUTE_PCI_BUS_ID": "hipDeviceAttributePciBusId",
    "CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID": "hipDeviceAttributePciDeviceId",
    "CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID": "hipDeviceAttributePciDomainID",
    "CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK": "hipDeviceAttributeMaxRegistersPerBlock",
    "CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK": "hipDeviceAttributeMaxSharedMemoryPerBlock",
    "CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO": "hipDeviceAttributeSingleToDoublePrecisionPerfRatio",
    "CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED": "hipDeviceAttributeStreamPrioritiesSupported",
    "CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT": "hipDeviceAttributeSurfaceAlignment",
    "CU_DEVICE_ATTRIBUTE_TCC_DRIVER": "hipDeviceAttributeTccDriver",
    "CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT": "hipDeviceAttributeTextureAlignment",
    "CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT": "hipDeviceAttributeTexturePitchAlignment",
    "CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY": "hipDeviceAttributeTotalConstantMemory",
    "CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING": "hipDeviceAttributeUnifiedAddressing",
    "CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED": "hipDeviceAttributeVirtualMemoryManagementSupported",
    "CU_DEVICE_ATTRIBUTE_WARP_SIZE": "hipDeviceAttributeWarpSize",
    "CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED": "hipDevP2PAttrHipArrayAccessSupported",
    "CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED": "hipDevP2PAttrAccessSupported",
    "CU_DEVICE_P2P_ATTRIBUTE_ARRAY_ACCESS_ACCESS_SUPPORTED": "hipDevP2PAttrHipArrayAccessSupported",
    "CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED": "hipDevP2PAttrHipArrayAccessSupported",
    "CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED": "hipDevP2PAttrNativeAtomicSupported",
    "CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK": "hipDevP2PAttrPerformanceRank",
    "CU_EVENT_BLOCKING_SYNC": "hipEventBlockingSync",
    "CU_EVENT_DEFAULT": "hipEventDefault",
    "CU_EVENT_DISABLE_TIMING": "hipEventDisableTiming",
    "CU_EVENT_INTERPROCESS": "hipEventInterprocess",
    "CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE": "hipExternalMemoryHandleTypeD3D11Resource",
    "CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT": "hipExternalMemoryHandleTypeD3D11ResourceKmt",
    "CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP": "hipExternalMemoryHandleTypeD3D12Heap",
    "CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE": "hipExternalMemoryHandleTypeD3D12Resource",
    "CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD": "hipExternalMemoryHandleTypeOpaqueFd",
    "CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32": "hipExternalMemoryHandleTypeOpaqueWin32",
    "CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT": "hipExternalMemoryHandleTypeOpaqueWin32Kmt",
    "CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE": "hipExternalSemaphoreHandleTypeD3D12Fence",
    "CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD": "hipExternalSemaphoreHandleTypeOpaqueFd",
    "CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32": "hipExternalSemaphoreHandleTypeOpaqueWin32",
    "CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT": "hipExternalSemaphoreHandleTypeOpaqueWin32Kmt",
    "CU_FUNC_ATTRIBUTE_BINARY_VERSION": "HIP_FUNC_ATTRIBUTE_BINARY_VERSION",
    "CU_FUNC_ATTRIBUTE_CACHE_MODE_CA": "HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA",
    "CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES": "HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES",
    "CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES": "HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES",
    "CU_FUNC_ATTRIBUTE_MAX": "HIP_FUNC_ATTRIBUTE_MAX",
    "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES": "HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES",
    "CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK": "HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK",
    "CU_FUNC_ATTRIBUTE_NUM_REGS": "HIP_FUNC_ATTRIBUTE_NUM_REGS",
    "CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT": "HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT",
    "CU_FUNC_ATTRIBUTE_PTX_VERSION": "HIP_FUNC_ATTRIBUTE_PTX_VERSION",
    "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES": "HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES",
    "CU_FUNC_CACHE_PREFER_EQUAL": "hipFuncCachePreferEqual",
    "CU_FUNC_CACHE_PREFER_L1": "hipFuncCachePreferL1",
    "CU_FUNC_CACHE_PREFER_NONE": "hipFuncCachePreferNone",
    "CU_FUNC_CACHE_PREFER_SHARED": "hipFuncCachePreferShared",
    "CU_GL_DEVICE_LIST_ALL": "hipGLDeviceListAll",
    "CU_GL_DEVICE_LIST_CURRENT_FRAME": "hipGLDeviceListCurrentFrame",
    "CU_GL_DEVICE_LIST_NEXT_FRAME": "hipGLDeviceListNextFrame",
    "CU_GRAPHICS_REGISTER_FLAGS_NONE": "hipGraphicsRegisterFlagsNone",
    "CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY": "hipGraphicsRegisterFlagsReadOnly",
    "CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST": "hipGraphicsRegisterFlagsSurfaceLoadStore",
    "CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER": "hipGraphicsRegisterFlagsTextureGather",
    "CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD": "hipGraphicsRegisterFlagsWriteDiscard",
    "CU_GRAPH_EXEC_UPDATE_ERROR": "hipGraphExecUpdateError",
    "CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED": "hipGraphExecUpdateErrorFunctionChanged",
    "CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED": "hipGraphExecUpdateErrorNodeTypeChanged",
    "CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED": "hipGraphExecUpdateErrorNotSupported",
    "CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED": "hipGraphExecUpdateErrorParametersChanged",
    "CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED": "hipGraphExecUpdateErrorTopologyChanged",
    "CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE": "hipGraphExecUpdateErrorUnsupportedFunctionChange",
    "CU_GRAPH_EXEC_UPDATE_SUCCESS": "hipGraphExecUpdateSuccess",
    "CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT": "hipGraphMemAttrReservedMemCurrent",
    "CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH": "hipGraphMemAttrReservedMemHigh",
    "CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT": "hipGraphMemAttrUsedMemCurrent",
    "CU_GRAPH_MEM_ATTR_USED_MEM_HIGH": "hipGraphMemAttrUsedMemHigh",
    "CU_GRAPH_NODE_TYPE_COUNT": "hipGraphNodeTypeCount",
    "CU_GRAPH_NODE_TYPE_EMPTY": "hipGraphNodeTypeEmpty",
    "CU_GRAPH_NODE_TYPE_EVENT_RECORD": "hipGraphNodeTypeEventRecord",
    "CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL": "hipGraphNodeTypeExtSemaphoreSignal",
    "CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT": "hipGraphNodeTypeExtSemaphoreWait",
    "CU_GRAPH_NODE_TYPE_GRAPH": "hipGraphNodeTypeGraph",
    "CU_GRAPH_NODE_TYPE_HOST": "hipGraphNodeTypeHost",
    "CU_GRAPH_NODE_TYPE_KERNEL": "hipGraphNodeTypeKernel",
    "CU_GRAPH_NODE_TYPE_MEMCPY": "hipGraphNodeTypeMemcpy",
    "CU_GRAPH_NODE_TYPE_MEMSET": "hipGraphNodeTypeMemset",
    "CU_GRAPH_NODE_TYPE_WAIT_EVENT": "hipGraphNodeTypeWaitEvent",
    "CU_GRAPH_USER_OBJECT_MOVE": "hipGraphUserObjectMove",
    "CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS": "hipIpcMemLazyEnablePeerAccess",
    "CU_JIT_CACHE_MODE": "HIPRTC_JIT_CACHE_MODE",
    "CU_JIT_ERROR_LOG_BUFFER": "HIPRTC_JIT_ERROR_LOG_BUFFER",
    "CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES": "HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES",
    "CU_JIT_FALLBACK_STRATEGY": "HIPRTC_JIT_FALLBACK_STRATEGY",
    "CU_JIT_FAST_COMPILE": "HIPRTC_JIT_FAST_COMPILE",
    "CU_JIT_GENERATE_DEBUG_INFO": "HIPRTC_JIT_GENERATE_DEBUG_INFO",
    "CU_JIT_GENERATE_LINE_INFO": "HIPRTC_JIT_GENERATE_LINE_INFO",
    "CU_JIT_INFO_LOG_BUFFER": "HIPRTC_JIT_INFO_LOG_BUFFER",
    "CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES": "HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES",
    "CU_JIT_INPUT_CUBIN": "HIPRTC_JIT_INPUT_CUBIN",
    "CU_JIT_INPUT_FATBINARY": "HIPRTC_JIT_INPUT_FATBINARY",
    "CU_JIT_INPUT_LIBRARY": "HIPRTC_JIT_INPUT_LIBRARY",
    "CU_JIT_INPUT_NVVM": "HIPRTC_JIT_INPUT_NVVM",
    "CU_JIT_INPUT_OBJECT": "HIPRTC_JIT_INPUT_OBJECT",
    "CU_JIT_INPUT_PTX": "HIPRTC_JIT_INPUT_PTX",
    "CU_JIT_LOG_VERBOSE": "HIPRTC_JIT_LOG_VERBOSE",
    "CU_JIT_MAX_REGISTERS": "HIPRTC_JIT_MAX_REGISTERS",
    "CU_JIT_NEW_SM3X_OPT": "HIPRTC_JIT_NEW_SM3X_OPT",
    "CU_JIT_NUM_INPUT_TYPES": "HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES",
    "CU_JIT_NUM_OPTIONS": "HIPRTC_JIT_NUM_OPTIONS",
    "CU_JIT_OPTIMIZATION_LEVEL": "HIPRTC_JIT_OPTIMIZATION_LEVEL",
    "CU_JIT_TARGET": "HIPRTC_JIT_TARGET",
    "CU_JIT_TARGET_FROM_CUCONTEXT": "HIPRTC_JIT_TARGET_FROM_HIPCONTEXT",
    "CU_JIT_THREADS_PER_BLOCK": "HIPRTC_JIT_THREADS_PER_BLOCK",
    "CU_JIT_WALL_TIME": "HIPRTC_JIT_WALL_TIME",
    "CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW": "hipKernelNodeAttributeAccessPolicyWindow",
    "CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE": "hipKernelNodeAttributeCooperative",
    "CU_LIMIT_MALLOC_HEAP_SIZE": "hipLimitMallocHeapSize",
    "CU_LIMIT_PRINTF_FIFO_SIZE": "hipLimitPrintfFifoSize",
    "CU_LIMIT_STACK_SIZE": "hipLimitStackSize",
    "CU_MEMORYTYPE_ARRAY": "hipMemoryTypeArray",
    "CU_MEMORYTYPE_DEVICE": "hipMemoryTypeDevice",
    "CU_MEMORYTYPE_HOST": "hipMemoryTypeHost",
    "CU_MEMORYTYPE_UNIFIED": "hipMemoryTypeUnified",
    "CU_MEMPOOL_ATTR_RELEASE_THRESHOLD": "hipMemPoolAttrReleaseThreshold",
    "CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT": "hipMemPoolAttrReservedMemCurrent",
    "CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH": "hipMemPoolAttrReservedMemHigh",
    "CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES": "hipMemPoolReuseAllowInternalDependencies",
    "CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC": "hipMemPoolReuseAllowOpportunistic",
    "CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES": "hipMemPoolReuseFollowEventDependencies",
    "CU_MEMPOOL_ATTR_USED_MEM_CURRENT": "hipMemPoolAttrUsedMemCurrent",
    "CU_MEMPOOL_ATTR_USED_MEM_HIGH": "hipMemPoolAttrUsedMemHigh",
    "CU_MEM_ACCESS_FLAGS_PROT_NONE": "hipMemAccessFlagsProtNone",
    "CU_MEM_ACCESS_FLAGS_PROT_READ": "hipMemAccessFlagsProtRead",
    "CU_MEM_ACCESS_FLAGS_PROT_READWRITE": "hipMemAccessFlagsProtReadWrite",
    "CU_MEM_ADVISE_SET_ACCESSED_BY": "hipMemAdviseSetAccessedBy",
    "CU_MEM_ADVISE_SET_PREFERRED_LOCATION": "hipMemAdviseSetPreferredLocation",
    "CU_MEM_ADVISE_SET_READ_MOSTLY": "hipMemAdviseSetReadMostly",
    "CU_MEM_ADVISE_UNSET_ACCESSED_BY": "hipMemAdviseUnsetAccessedBy",
    "CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION": "hipMemAdviseUnsetPreferredLocation",
    "CU_MEM_ADVISE_UNSET_READ_MOSTLY": "hipMemAdviseUnsetReadMostly",
    "CU_MEM_ALLOCATION_TYPE_INVALID": "hipMemAllocationTypeInvalid",
    "CU_MEM_ALLOCATION_TYPE_MAX": "hipMemAllocationTypeMax",
    "CU_MEM_ALLOCATION_TYPE_PINNED": "hipMemAllocationTypePinned",
    "CU_MEM_ALLOC_GRANULARITY_MINIMUM": "hipMemAllocationGranularityMinimum",
    "CU_MEM_ALLOC_GRANULARITY_RECOMMENDED": "hipMemAllocationGranularityRecommended",
    "CU_MEM_ATTACH_GLOBAL": "hipMemAttachGlobal",
    "CU_MEM_ATTACH_HOST": "hipMemAttachHost",
    "CU_MEM_ATTACH_SINGLE": "hipMemAttachSingle",
    "CU_MEM_HANDLE_TYPE_GENERIC": "hipMemHandleTypeGeneric",
    "CU_MEM_HANDLE_TYPE_NONE": "hipMemHandleTypeNone",
    "CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR": "hipMemHandleTypePosixFileDescriptor",
    "CU_MEM_HANDLE_TYPE_WIN32": "hipMemHandleTypeWin32",
    "CU_MEM_HANDLE_TYPE_WIN32_KMT": "hipMemHandleTypeWin32Kmt",
    "CU_MEM_LOCATION_TYPE_DEVICE": "hipMemLocationTypeDevice",
    "CU_MEM_LOCATION_TYPE_INVALID": "hipMemLocationTypeInvalid",
    "CU_MEM_OPERATION_TYPE_MAP": "hipMemOperationTypeMap",
    "CU_MEM_OPERATION_TYPE_UNMAP": "hipMemOperationTypeUnmap",
    "CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY": "hipMemRangeAttributeAccessedBy",
    "CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION": "hipMemRangeAttributeLastPrefetchLocation",
    "CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION": "hipMemRangeAttributePreferredLocation",
    "CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY": "hipMemRangeAttributeReadMostly",
    "CU_OCCUPANCY_DEFAULT": "hipOccupancyDefault",
    "CU_POINTER_ATTRIBUTE_ACCESS_FLAGS": "HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS",
    "CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES": "HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES",
    "CU_POINTER_ATTRIBUTE_BUFFER_ID": "HIP_POINTER_ATTRIBUTE_BUFFER_ID",
    "CU_POINTER_ATTRIBUTE_CONTEXT": "HIP_POINTER_ATTRIBUTE_CONTEXT",
    "CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL": "HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL",
    "CU_POINTER_ATTRIBUTE_DEVICE_POINTER": "HIP_POINTER_ATTRIBUTE_DEVICE_POINTER",
    "CU_POINTER_ATTRIBUTE_HOST_POINTER": "HIP_POINTER_ATTRIBUTE_HOST_POINTER",
    "CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE": "HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE",
    "CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE": "HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE",
    "CU_POINTER_ATTRIBUTE_IS_MANAGED": "HIP_POINTER_ATTRIBUTE_IS_MANAGED",
    "CU_POINTER_ATTRIBUTE_MAPPED": "HIP_POINTER_ATTRIBUTE_MAPPED",
    "CU_POINTER_ATTRIBUTE_MEMORY_TYPE": "HIP_POINTER_ATTRIBUTE_MEMORY_TYPE",
    "CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE": "HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE",
    "CU_POINTER_ATTRIBUTE_P2P_TOKENS": "HIP_POINTER_ATTRIBUTE_P2P_TOKENS",
    "CU_POINTER_ATTRIBUTE_RANGE_SIZE": "HIP_POINTER_ATTRIBUTE_RANGE_SIZE",
    "CU_POINTER_ATTRIBUTE_RANGE_START_ADDR": "HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR",
    "CU_POINTER_ATTRIBUTE_SYNC_MEMOPS": "HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS",
    "CU_RESOURCE_TYPE_ARRAY": "HIP_RESOURCE_TYPE_ARRAY",
    "CU_RESOURCE_TYPE_LINEAR": "HIP_RESOURCE_TYPE_LINEAR",
    "CU_RESOURCE_TYPE_MIPMAPPED_ARRAY": "HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY",
    "CU_RESOURCE_TYPE_PITCH2D": "HIP_RESOURCE_TYPE_PITCH2D",
    "CU_RES_VIEW_FORMAT_FLOAT_1X16": "HIP_RES_VIEW_FORMAT_FLOAT_1X16",
    "CU_RES_VIEW_FORMAT_FLOAT_1X32": "HIP_RES_VIEW_FORMAT_FLOAT_1X32",
    "CU_RES_VIEW_FORMAT_FLOAT_2X16": "HIP_RES_VIEW_FORMAT_FLOAT_2X16",
    "CU_RES_VIEW_FORMAT_FLOAT_2X32": "HIP_RES_VIEW_FORMAT_FLOAT_2X32",
    "CU_RES_VIEW_FORMAT_FLOAT_4X16": "HIP_RES_VIEW_FORMAT_FLOAT_4X16",
    "CU_RES_VIEW_FORMAT_FLOAT_4X32": "HIP_RES_VIEW_FORMAT_FLOAT_4X32",
    "CU_RES_VIEW_FORMAT_NONE": "HIP_RES_VIEW_FORMAT_NONE",
    "CU_RES_VIEW_FORMAT_SIGNED_BC4": "HIP_RES_VIEW_FORMAT_SIGNED_BC4",
    "CU_RES_VIEW_FORMAT_SIGNED_BC5": "HIP_RES_VIEW_FORMAT_SIGNED_BC5",
    "CU_RES_VIEW_FORMAT_SIGNED_BC6H": "HIP_RES_VIEW_FORMAT_SIGNED_BC6H",
    "CU_RES_VIEW_FORMAT_SINT_1X16": "HIP_RES_VIEW_FORMAT_SINT_1X16",
    "CU_RES_VIEW_FORMAT_SINT_1X32": "HIP_RES_VIEW_FORMAT_SINT_1X32",
    "CU_RES_VIEW_FORMAT_SINT_1X8": "HIP_RES_VIEW_FORMAT_SINT_1X8",
    "CU_RES_VIEW_FORMAT_SINT_2X16": "HIP_RES_VIEW_FORMAT_SINT_2X16",
    "CU_RES_VIEW_FORMAT_SINT_2X32": "HIP_RES_VIEW_FORMAT_SINT_2X32",
    "CU_RES_VIEW_FORMAT_SINT_2X8": "HIP_RES_VIEW_FORMAT_SINT_2X8",
    "CU_RES_VIEW_FORMAT_SINT_4X16": "HIP_RES_VIEW_FORMAT_SINT_4X16",
    "CU_RES_VIEW_FORMAT_SINT_4X32": "HIP_RES_VIEW_FORMAT_SINT_4X32",
    "CU_RES_VIEW_FORMAT_SINT_4X8": "HIP_RES_VIEW_FORMAT_SINT_4X8",
    "CU_RES_VIEW_FORMAT_UINT_1X16": "HIP_RES_VIEW_FORMAT_UINT_1X16",
    "CU_RES_VIEW_FORMAT_UINT_1X32": "HIP_RES_VIEW_FORMAT_UINT_1X32",
    "CU_RES_VIEW_FORMAT_UINT_1X8": "HIP_RES_VIEW_FORMAT_UINT_1X8",
    "CU_RES_VIEW_FORMAT_UINT_2X16": "HIP_RES_VIEW_FORMAT_UINT_2X16",
    "CU_RES_VIEW_FORMAT_UINT_2X32": "HIP_RES_VIEW_FORMAT_UINT_2X32",
    "CU_RES_VIEW_FORMAT_UINT_2X8": "HIP_RES_VIEW_FORMAT_UINT_2X8",
    "CU_RES_VIEW_FORMAT_UINT_4X16": "HIP_RES_VIEW_FORMAT_UINT_4X16",
    "CU_RES_VIEW_FORMAT_UINT_4X32": "HIP_RES_VIEW_FORMAT_UINT_4X32",
    "CU_RES_VIEW_FORMAT_UINT_4X8": "HIP_RES_VIEW_FORMAT_UINT_4X8",
    "CU_RES_VIEW_FORMAT_UNSIGNED_BC1": "HIP_RES_VIEW_FORMAT_UNSIGNED_BC1",
    "CU_RES_VIEW_FORMAT_UNSIGNED_BC2": "HIP_RES_VIEW_FORMAT_UNSIGNED_BC2",
    "CU_RES_VIEW_FORMAT_UNSIGNED_BC3": "HIP_RES_VIEW_FORMAT_UNSIGNED_BC3",
    "CU_RES_VIEW_FORMAT_UNSIGNED_BC4": "HIP_RES_VIEW_FORMAT_UNSIGNED_BC4",
    "CU_RES_VIEW_FORMAT_UNSIGNED_BC5": "HIP_RES_VIEW_FORMAT_UNSIGNED_BC5",
    "CU_RES_VIEW_FORMAT_UNSIGNED_BC6H": "HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H",
    "CU_RES_VIEW_FORMAT_UNSIGNED_BC7": "HIP_RES_VIEW_FORMAT_UNSIGNED_BC7",
    "CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE": "hipSharedMemBankSizeDefault",
    "CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE": "hipSharedMemBankSizeEightByte",
    "CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE": "hipSharedMemBankSizeFourByte",
    "CU_STREAM_ADD_CAPTURE_DEPENDENCIES": "hipStreamAddCaptureDependencies",
    "CU_STREAM_CAPTURE_MODE_GLOBAL": "hipStreamCaptureModeGlobal",
    "CU_STREAM_CAPTURE_MODE_RELAXED": "hipStreamCaptureModeRelaxed",
    "CU_STREAM_CAPTURE_MODE_THREAD_LOCAL": "hipStreamCaptureModeThreadLocal",
    "CU_STREAM_CAPTURE_STATUS_ACTIVE": "hipStreamCaptureStatusActive",
    "CU_STREAM_CAPTURE_STATUS_INVALIDATED": "hipStreamCaptureStatusInvalidated",
    "CU_STREAM_CAPTURE_STATUS_NONE": "hipStreamCaptureStatusNone",
    "CU_STREAM_DEFAULT": "hipStreamDefault",
    "CU_STREAM_NON_BLOCKING": "hipStreamNonBlocking",
    "CU_STREAM_SET_CAPTURE_DEPENDENCIES": "hipStreamSetCaptureDependencies",
    "CU_STREAM_WAIT_VALUE_AND": "hipStreamWaitValueAnd",
    "CU_STREAM_WAIT_VALUE_EQ": "hipStreamWaitValueEq",
    "CU_STREAM_WAIT_VALUE_GEQ": "hipStreamWaitValueGte",
    "CU_STREAM_WAIT_VALUE_NOR": "hipStreamWaitValueNor",
    "CU_TR_ADDRESS_MODE_BORDER": "HIP_TR_ADDRESS_MODE_BORDER",
    "CU_TR_ADDRESS_MODE_CLAMP": "HIP_TR_ADDRESS_MODE_CLAMP",
    "CU_TR_ADDRESS_MODE_MIRROR": "HIP_TR_ADDRESS_MODE_MIRROR",
    "CU_TR_ADDRESS_MODE_WRAP": "HIP_TR_ADDRESS_MODE_WRAP",
    "CU_TR_FILTER_MODE_LINEAR": "HIP_TR_FILTER_MODE_LINEAR",
    "CU_TR_FILTER_MODE_POINT": "HIP_TR_FILTER_MODE_POINT",
    "CU_USER_OBJECT_NO_DESTRUCTOR_SYNC": "hipUserObjectNoDestructorSync",
    "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE": "HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE",
    "NVRTC_ERROR_COMPILATION": "HIPRTC_ERROR_COMPILATION",
    "NVRTC_ERROR_INTERNAL_ERROR": "HIPRTC_ERROR_INTERNAL_ERROR",
    "NVRTC_ERROR_INVALID_INPUT": "HIPRTC_ERROR_INVALID_INPUT",
    "NVRTC_ERROR_INVALID_OPTION": "HIPRTC_ERROR_INVALID_OPTION",
    "NVRTC_ERROR_INVALID_PROGRAM": "HIPRTC_ERROR_INVALID_PROGRAM",
    "NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID": "HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID",
    "NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION": "HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION",
    "NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION": "HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION",
    "NVRTC_ERROR_OUT_OF_MEMORY": "HIPRTC_ERROR_OUT_OF_MEMORY",
    "NVRTC_ERROR_PROGRAM_CREATION_FAILURE": "HIPRTC_ERROR_PROGRAM_CREATION_FAILURE",
    "NVRTC_SUCCESS": "HIPRTC_SUCCESS",
    "cudaAccessPropertyNormal": "hipAccessPropertyNormal",
    "cudaAccessPropertyPersisting": "hipAccessPropertyPersisting",
    "cudaAccessPropertyStreaming": "hipAccessPropertyStreaming",
    "cudaAddressModeBorder": "hipAddressModeBorder",
    "cudaAddressModeClamp": "hipAddressModeClamp",
    "cudaAddressModeMirror": "hipAddressModeMirror",
    "cudaAddressModeWrap": "hipAddressModeWrap",
    "cudaBoundaryModeClamp": "hipBoundaryModeClamp",
    "cudaBoundaryModeTrap": "hipBoundaryModeTrap",
    "cudaBoundaryModeZero": "hipBoundaryModeZero",
    "cudaChannelFormatKindFloat": "hipChannelFormatKindFloat",
    "cudaChannelFormatKindNone": "hipChannelFormatKindNone",
    "cudaChannelFormatKindSigned": "hipChannelFormatKindSigned",
    "cudaChannelFormatKindUnsigned": "hipChannelFormatKindUnsigned",
    "cudaComputeModeDefault": "hipComputeModeDefault",
    "cudaComputeModeExclusive": "hipComputeModeExclusive",
    "cudaComputeModeExclusiveProcess": "hipComputeModeExclusiveProcess",
    "cudaComputeModeProhibited": "hipComputeModeProhibited",
    "cudaDevAttrAsyncEngineCount": "hipDeviceAttributeAsyncEngineCount",
    "cudaDevAttrCanMapHostMemory": "hipDeviceAttributeCanMapHostMemory",
    "cudaDevAttrCanUseHostPointerForRegisteredMem": "hipDeviceAttributeCanUseHostPointerForRegisteredMem",
    "cudaDevAttrClockRate": "hipDeviceAttributeClockRate",
    "cudaDevAttrComputeCapabilityMajor": "hipDeviceAttributeComputeCapabilityMajor",
    "cudaDevAttrComputeCapabilityMinor": "hipDeviceAttributeComputeCapabilityMinor",
    "cudaDevAttrComputeMode": "hipDeviceAttributeComputeMode",
    "cudaDevAttrComputePreemptionSupported": "hipDeviceAttributeComputePreemptionSupported",
    "cudaDevAttrConcurrentKernels": "hipDeviceAttributeConcurrentKernels",
    "cudaDevAttrConcurrentManagedAccess": "hipDeviceAttributeConcurrentManagedAccess",
    "cudaDevAttrCooperativeLaunch": "hipDeviceAttributeCooperativeLaunch",
    "cudaDevAttrCooperativeMultiDeviceLaunch": "hipDeviceAttributeCooperativeMultiDeviceLaunch",
    "cudaDevAttrDirectManagedMemAccessFromHost": "hipDeviceAttributeDirectManagedMemAccessFromHost",
    "cudaDevAttrEccEnabled": "hipDeviceAttributeEccEnabled",
    "cudaDevAttrGlobalL1CacheSupported": "hipDeviceAttributeGlobalL1CacheSupported",
    "cudaDevAttrGlobalMemoryBusWidth": "hipDeviceAttributeMemoryBusWidth",
    "cudaDevAttrGpuOverlap": "hipDeviceAttributeAsyncEngineCount",
    "cudaDevAttrHostNativeAtomicSupported": "hipDeviceAttributeHostNativeAtomicSupported",
    "cudaDevAttrIntegrated": "hipDeviceAttributeIntegrated",
    "cudaDevAttrIsMultiGpuBoard": "hipDeviceAttributeIsMultiGpuBoard",
    "cudaDevAttrKernelExecTimeout": "hipDeviceAttributeKernelExecTimeout",
    "cudaDevAttrL2CacheSize": "hipDeviceAttributeL2CacheSize",
    "cudaDevAttrLocalL1CacheSupported": "hipDeviceAttributeLocalL1CacheSupported",
    "cudaDevAttrManagedMemory": "hipDeviceAttributeManagedMemory",
    "cudaDevAttrMaxBlockDimX": "hipDeviceAttributeMaxBlockDimX",
    "cudaDevAttrMaxBlockDimY": "hipDeviceAttributeMaxBlockDimY",
    "cudaDevAttrMaxBlockDimZ": "hipDeviceAttributeMaxBlockDimZ",
    "cudaDevAttrMaxBlocksPerMultiprocessor": "hipDeviceAttributeMaxBlocksPerMultiprocessor",
    "cudaDevAttrMaxGridDimX": "hipDeviceAttributeMaxGridDimX",
    "cudaDevAttrMaxGridDimY": "hipDeviceAttributeMaxGridDimY",
    "cudaDevAttrMaxGridDimZ": "hipDeviceAttributeMaxGridDimZ",
    "cudaDevAttrMaxPitch": "hipDeviceAttributeMaxPitch",
    "cudaDevAttrMaxRegistersPerBlock": "hipDeviceAttributeMaxRegistersPerBlock",
    "cudaDevAttrMaxRegistersPerMultiprocessor": "hipDeviceAttributeMaxRegistersPerMultiprocessor",
    "cudaDevAttrMaxSharedMemoryPerBlock": "hipDeviceAttributeMaxSharedMemoryPerBlock",
    "cudaDevAttrMaxSharedMemoryPerBlockOptin": "hipDeviceAttributeSharedMemPerBlockOptin",
    "cudaDevAttrMaxSharedMemoryPerMultiprocessor": "hipDeviceAttributeMaxSharedMemoryPerMultiprocessor",
    "cudaDevAttrMaxSurface1DLayeredWidth": "hipDeviceAttributeMaxSurface1DLayered",
    "cudaDevAttrMaxSurface1DWidth": "hipDeviceAttributeMaxSurface1D",
    "cudaDevAttrMaxSurface2DHeight": "hipDeviceAttributeMaxSurface2D",
    "cudaDevAttrMaxSurface2DLayeredHeight": "hipDeviceAttributeMaxSurface2DLayered",
    "cudaDevAttrMaxSurface2DLayeredWidth": "hipDeviceAttributeMaxSurface2DLayered",
    "cudaDevAttrMaxSurface2DWidth": "hipDeviceAttributeMaxSurface2D",
    "cudaDevAttrMaxSurface3DDepth": "hipDeviceAttributeMaxSurface3D",
    "cudaDevAttrMaxSurface3DHeight": "hipDeviceAttributeMaxSurface3D",
    "cudaDevAttrMaxSurface3DWidth": "hipDeviceAttributeMaxSurface3D",
    "cudaDevAttrMaxSurfaceCubemapLayeredWidth": "hipDeviceAttributeMaxSurfaceCubemapLayered",
    "cudaDevAttrMaxSurfaceCubemapWidth": "hipDeviceAttributeMaxSurfaceCubemap",
    "cudaDevAttrMaxTexture1DLayeredWidth": "hipDeviceAttributeMaxTexture1DLayered",
    "cudaDevAttrMaxTexture1DLinearWidth": "hipDeviceAttributeMaxTexture1DLinear",
    "cudaDevAttrMaxTexture1DMipmappedWidth": "hipDeviceAttributeMaxTexture1DMipmap",
    "cudaDevAttrMaxTexture1DWidth": "hipDeviceAttributeMaxTexture1DWidth",
    "cudaDevAttrMaxTexture2DGatherHeight": "hipDeviceAttributeMaxTexture2DGather",
    "cudaDevAttrMaxTexture2DGatherWidth": "hipDeviceAttributeMaxTexture2DGather",
    "cudaDevAttrMaxTexture2DHeight": "hipDeviceAttributeMaxTexture2DHeight",
    "cudaDevAttrMaxTexture2DLayeredHeight": "hipDeviceAttributeMaxTexture2DLayered",
    "cudaDevAttrMaxTexture2DLayeredWidth": "hipDeviceAttributeMaxTexture2DLayered",
    "cudaDevAttrMaxTexture2DLinearHeight": "hipDeviceAttributeMaxTexture2DLinear",
    "cudaDevAttrMaxTexture2DLinearPitch": "hipDeviceAttributeMaxTexture2DLinear",
    "cudaDevAttrMaxTexture2DLinearWidth": "hipDeviceAttributeMaxTexture2DLinear",
    "cudaDevAttrMaxTexture2DMipmappedHeight": "hipDeviceAttributeMaxTexture2DMipmap",
    "cudaDevAttrMaxTexture2DMipmappedWidth": "hipDeviceAttributeMaxTexture2DMipmap",
    "cudaDevAttrMaxTexture2DWidth": "hipDeviceAttributeMaxTexture2DWidth",
    "cudaDevAttrMaxTexture3DDepth": "hipDeviceAttributeMaxTexture3DDepth",
    "cudaDevAttrMaxTexture3DDepthAlt": "hipDeviceAttributeMaxTexture3DAlt",
    "cudaDevAttrMaxTexture3DHeight": "hipDeviceAttributeMaxTexture3DHeight",
    "cudaDevAttrMaxTexture3DHeightAlt": "hipDeviceAttributeMaxTexture3DAlt",
    "cudaDevAttrMaxTexture3DWidth": "hipDeviceAttributeMaxTexture3DWidth",
    "cudaDevAttrMaxTexture3DWidthAlt": "hipDeviceAttributeMaxTexture3DAlt",
    "cudaDevAttrMaxTextureCubemapLayeredWidth": "hipDeviceAttributeMaxTextureCubemapLayered",
    "cudaDevAttrMaxTextureCubemapWidth": "hipDeviceAttributeMaxTextureCubemap",
    "cudaDevAttrMaxThreadsPerBlock": "hipDeviceAttributeMaxThreadsPerBlock",
    "cudaDevAttrMaxThreadsPerMultiProcessor": "hipDeviceAttributeMaxThreadsPerMultiProcessor",
    "cudaDevAttrMemoryClockRate": "hipDeviceAttributeMemoryClockRate",
    "cudaDevAttrMemoryPoolsSupported": "hipDeviceAttributeMemoryPoolsSupported",
    "cudaDevAttrMultiGpuBoardGroupID": "hipDeviceAttributeMultiGpuBoardGroupID",
    "cudaDevAttrMultiProcessorCount": "hipDeviceAttributeMultiprocessorCount",
    "cudaDevAttrPageableMemoryAccess": "hipDeviceAttributePageableMemoryAccess",
    "cudaDevAttrPageableMemoryAccessUsesHostPageTables": "hipDeviceAttributePageableMemoryAccessUsesHostPageTables",
    "cudaDevAttrPciBusId": "hipDeviceAttributePciBusId",
    "cudaDevAttrPciDeviceId": "hipDeviceAttributePciDeviceId",
    "cudaDevAttrPciDomainId": "hipDeviceAttributePciDomainID",
    "cudaDevAttrReserved94": "hipDeviceAttributeCanUseStreamWaitValue",
    "cudaDevAttrSingleToDoublePrecisionPerfRatio": "hipDeviceAttributeSingleToDoublePrecisionPerfRatio",
    "cudaDevAttrStreamPrioritiesSupported": "hipDeviceAttributeStreamPrioritiesSupported",
    "cudaDevAttrSurfaceAlignment": "hipDeviceAttributeSurfaceAlignment",
    "cudaDevAttrTccDriver": "hipDeviceAttributeTccDriver",
    "cudaDevAttrTextureAlignment": "hipDeviceAttributeTextureAlignment",
    "cudaDevAttrTexturePitchAlignment": "hipDeviceAttributeTexturePitchAlignment",
    "cudaDevAttrTotalConstantMemory": "hipDeviceAttributeTotalConstantMemory",
    "cudaDevAttrUnifiedAddressing": "hipDeviceAttributeUnifiedAddressing",
    "cudaDevAttrWarpSize": "hipDeviceAttributeWarpSize",
    "cudaDevP2PAttrAccessSupported": "hipDevP2PAttrAccessSupported",
    "cudaDevP2PAttrCudaArrayAccessSupported": "hipDevP2PAttrHipArrayAccessSupported",
    "cudaDevP2PAttrNativeAtomicSupported": "hipDevP2PAttrNativeAtomicSupported",
    "cudaDevP2PAttrPerformanceRank": "hipDevP2PAttrPerformanceRank",
    "cudaErrorAlreadyAcquired": "hipErrorAlreadyAcquired",
    "cudaErrorAlreadyMapped": "hipErrorAlreadyMapped",
    "cudaErrorArrayIsMapped": "hipErrorArrayIsMapped",
    "cudaErrorAssert": "hipErrorAssert",
    "cudaErrorCapturedEvent": "hipErrorCapturedEvent",
    "cudaErrorContextIsDestroyed": "hipErrorContextIsDestroyed",
    "cudaErrorCooperativeLaunchTooLarge": "hipErrorCooperativeLaunchTooLarge",
    "cudaErrorCudartUnloading": "hipErrorDeinitialized",
    "cudaErrorDeviceAlreadyInUse": "hipErrorContextAlreadyInUse",
    "cudaErrorDeviceUninitialized": "hipErrorInvalidContext",
    "cudaErrorECCUncorrectable": "hipErrorECCNotCorrectable",
    "cudaErrorFileNotFound": "hipErrorFileNotFound",
    "cudaErrorGraphExecUpdateFailure": "hipErrorGraphExecUpdateFailure",
    "cudaErrorHostMemoryAlreadyRegistered": "hipErrorHostMemoryAlreadyRegistered",
    "cudaErrorHostMemoryNotRegistered": "hipErrorHostMemoryNotRegistered",
    "cudaErrorIllegalAddress": "hipErrorIllegalAddress",
    "cudaErrorIllegalState": "hipErrorIllegalState",
    "cudaErrorInitializationError": "hipErrorNotInitialized",
    "cudaErrorInsufficientDriver": "hipErrorInsufficientDriver",
    "cudaErrorInvalidConfiguration": "hipErrorInvalidConfiguration",
    "cudaErrorInvalidDevice": "hipErrorInvalidDevice",
    "cudaErrorInvalidDeviceFunction": "hipErrorInvalidDeviceFunction",
    "cudaErrorInvalidDevicePointer": "hipErrorInvalidDevicePointer",
    "cudaErrorInvalidGraphicsContext": "hipErrorInvalidGraphicsContext",
    "cudaErrorInvalidKernelImage": "hipErrorInvalidImage",
    "cudaErrorInvalidMemcpyDirection": "hipErrorInvalidMemcpyDirection",
    "cudaErrorInvalidPitchValue": "hipErrorInvalidPitchValue",
    "cudaErrorInvalidPtx": "hipErrorInvalidKernelFile",
    "cudaErrorInvalidResourceHandle": "hipErrorInvalidHandle",
    "cudaErrorInvalidSource": "hipErrorInvalidSource",
    "cudaErrorInvalidSymbol": "hipErrorInvalidSymbol",
    "cudaErrorInvalidValue": "hipErrorInvalidValue",
    "cudaErrorLaunchFailure": "hipErrorLaunchFailure",
    "cudaErrorLaunchOutOfResources": "hipErrorLaunchOutOfResources",
    "cudaErrorLaunchTimeout": "hipErrorLaunchTimeOut",
    "cudaErrorMapBufferObjectFailed": "hipErrorMapFailed",
    "cudaErrorMemoryAllocation": "hipErrorOutOfMemory",
    "cudaErrorMissingConfiguration": "hipErrorMissingConfiguration",
    "cudaErrorNoDevice": "hipErrorNoDevice",
    "cudaErrorNoKernelImageForDevice": "hipErrorNoBinaryForGpu",
    "cudaErrorNotMapped": "hipErrorNotMapped",
    "cudaErrorNotMappedAsArray": "hipErrorNotMappedAsArray",
    "cudaErrorNotMappedAsPointer": "hipErrorNotMappedAsPointer",
    "cudaErrorNotReady": "hipErrorNotReady",
    "cudaErrorNotSupported": "hipErrorNotSupported",
    "cudaErrorOperatingSystem": "hipErrorOperatingSystem",
    "cudaErrorPeerAccessAlreadyEnabled": "hipErrorPeerAccessAlreadyEnabled",
    "cudaErrorPeerAccessNotEnabled": "hipErrorPeerAccessNotEnabled",
    "cudaErrorPeerAccessUnsupported": "hipErrorPeerAccessUnsupported",
    "cudaErrorPriorLaunchFailure": "hipErrorPriorLaunchFailure",
    "cudaErrorProfilerAlreadyStarted": "hipErrorProfilerAlreadyStarted",
    "cudaErrorProfilerAlreadyStopped": "hipErrorProfilerAlreadyStopped",
    "cudaErrorProfilerDisabled": "hipErrorProfilerDisabled",
    "cudaErrorProfilerNotInitialized": "hipErrorProfilerNotInitialized",
    "cudaErrorSetOnActiveProcess": "hipErrorSetOnActiveProcess",
    "cudaErrorSharedObjectInitFailed": "hipErrorSharedObjectInitFailed",
    "cudaErrorSharedObjectSymbolNotFound": "hipErrorSharedObjectSymbolNotFound",
    "cudaErrorStreamCaptureImplicit": "hipErrorStreamCaptureImplicit",
    "cudaErrorStreamCaptureInvalidated": "hipErrorStreamCaptureInvalidated",
    "cudaErrorStreamCaptureIsolation": "hipErrorStreamCaptureIsolation",
    "cudaErrorStreamCaptureMerge": "hipErrorStreamCaptureMerge",
    "cudaErrorStreamCaptureUnjoined": "hipErrorStreamCaptureUnjoined",
    "cudaErrorStreamCaptureUnmatched": "hipErrorStreamCaptureUnmatched",
    "cudaErrorStreamCaptureUnsupported": "hipErrorStreamCaptureUnsupported",
    "cudaErrorStreamCaptureWrongThread": "hipErrorStreamCaptureWrongThread",
    "cudaErrorSymbolNotFound": "hipErrorNotFound",
    "cudaErrorUnknown": "hipErrorUnknown",
    "cudaErrorUnmapBufferObjectFailed": "hipErrorUnmapFailed",
    "cudaErrorUnsupportedLimit": "hipErrorUnsupportedLimit",
    "cudaExternalMemoryHandleTypeD3D11Resource": "hipExternalMemoryHandleTypeD3D11Resource",
    "cudaExternalMemoryHandleTypeD3D11ResourceKmt": "hipExternalMemoryHandleTypeD3D11ResourceKmt",
    "cudaExternalMemoryHandleTypeD3D12Heap": "hipExternalMemoryHandleTypeD3D12Heap",
    "cudaExternalMemoryHandleTypeD3D12Resource": "hipExternalMemoryHandleTypeD3D12Resource",
    "cudaExternalMemoryHandleTypeOpaqueFd": "hipExternalMemoryHandleTypeOpaqueFd",
    "cudaExternalMemoryHandleTypeOpaqueWin32": "hipExternalMemoryHandleTypeOpaqueWin32",
    "cudaExternalMemoryHandleTypeOpaqueWin32Kmt": "hipExternalMemoryHandleTypeOpaqueWin32Kmt",
    "cudaExternalSemaphoreHandleTypeD3D12Fence": "hipExternalSemaphoreHandleTypeD3D12Fence",
    "cudaExternalSemaphoreHandleTypeOpaqueFd": "hipExternalSemaphoreHandleTypeOpaqueFd",
    "cudaExternalSemaphoreHandleTypeOpaqueWin32": "hipExternalSemaphoreHandleTypeOpaqueWin32",
    "cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt": "hipExternalSemaphoreHandleTypeOpaqueWin32Kmt",
    "cudaFilterModeLinear": "hipFilterModeLinear",
    "cudaFilterModePoint": "hipFilterModePoint",
    "cudaFuncAttributeMax": "hipFuncAttributeMax",
    "cudaFuncAttributeMaxDynamicSharedMemorySize": "hipFuncAttributeMaxDynamicSharedMemorySize",
    "cudaFuncAttributePreferredSharedMemoryCarveout": "hipFuncAttributePreferredSharedMemoryCarveout",
    "cudaFuncCachePreferEqual": "hipFuncCachePreferEqual",
    "cudaFuncCachePreferL1": "hipFuncCachePreferL1",
    "cudaFuncCachePreferNone": "hipFuncCachePreferNone",
    "cudaFuncCachePreferShared": "hipFuncCachePreferShared",
    "cudaGLDeviceListAll": "hipGLDeviceListAll",
    "cudaGLDeviceListCurrentFrame": "hipGLDeviceListCurrentFrame",
    "cudaGLDeviceListNextFrame": "hipGLDeviceListNextFrame",
    "cudaGraphExecUpdateError": "hipGraphExecUpdateError",
    "cudaGraphExecUpdateErrorFunctionChanged": "hipGraphExecUpdateErrorFunctionChanged",
    "cudaGraphExecUpdateErrorNodeTypeChanged": "hipGraphExecUpdateErrorNodeTypeChanged",
    "cudaGraphExecUpdateErrorNotSupported": "hipGraphExecUpdateErrorNotSupported",
    "cudaGraphExecUpdateErrorParametersChanged": "hipGraphExecUpdateErrorParametersChanged",
    "cudaGraphExecUpdateErrorTopologyChanged": "hipGraphExecUpdateErrorTopologyChanged",
    "cudaGraphExecUpdateErrorUnsupportedFunctionChange": "hipGraphExecUpdateErrorUnsupportedFunctionChange",
    "cudaGraphExecUpdateSuccess": "hipGraphExecUpdateSuccess",
    "cudaGraphInstantiateFlagAutoFreeOnLaunch": "hipGraphInstantiateFlagAutoFreeOnLaunch",
    "cudaGraphMemAttrReservedMemCurrent": "hipGraphMemAttrReservedMemCurrent",
    "cudaGraphMemAttrReservedMemHigh": "hipGraphMemAttrReservedMemHigh",
    "cudaGraphMemAttrUsedMemCurrent": "hipGraphMemAttrUsedMemCurrent",
    "cudaGraphMemAttrUsedMemHigh": "hipGraphMemAttrUsedMemHigh",
    "cudaGraphNodeTypeCount": "hipGraphNodeTypeCount",
    "cudaGraphNodeTypeEmpty": "hipGraphNodeTypeEmpty",
    "cudaGraphNodeTypeEventRecord": "hipGraphNodeTypeEventRecord",
    "cudaGraphNodeTypeExtSemaphoreSignal": "hipGraphNodeTypeExtSemaphoreSignal",
    "cudaGraphNodeTypeExtSemaphoreWait": "hipGraphNodeTypeExtSemaphoreWait",
    "cudaGraphNodeTypeGraph": "hipGraphNodeTypeGraph",
    "cudaGraphNodeTypeHost": "hipGraphNodeTypeHost",
    "cudaGraphNodeTypeKernel": "hipGraphNodeTypeKernel",
    "cudaGraphNodeTypeMemcpy": "hipGraphNodeTypeMemcpy",
    "cudaGraphNodeTypeMemset": "hipGraphNodeTypeMemset",
    "cudaGraphNodeTypeWaitEvent": "hipGraphNodeTypeWaitEvent",
    "cudaGraphUserObjectMove": "hipGraphUserObjectMove",
    "cudaGraphicsRegisterFlagsNone": "hipGraphicsRegisterFlagsNone",
    "cudaGraphicsRegisterFlagsReadOnly": "hipGraphicsRegisterFlagsReadOnly",
    "cudaGraphicsRegisterFlagsSurfaceLoadStore": "hipGraphicsRegisterFlagsSurfaceLoadStore",
    "cudaGraphicsRegisterFlagsTextureGather": "hipGraphicsRegisterFlagsTextureGather",
    "cudaGraphicsRegisterFlagsWriteDiscard": "hipGraphicsRegisterFlagsWriteDiscard",
    "cudaKernelNodeAttributeAccessPolicyWindow": "hipKernelNodeAttributeAccessPolicyWindow",
    "cudaKernelNodeAttributeCooperative": "hipKernelNodeAttributeCooperative",
    "cudaLimitMallocHeapSize": "hipLimitMallocHeapSize",
    "cudaLimitPrintfFifoSize": "hipLimitPrintfFifoSize",
    "cudaLimitStackSize": "hipLimitStackSize",
    "cudaMemAccessFlagsProtNone": "hipMemAccessFlagsProtNone",
    "cudaMemAccessFlagsProtRead": "hipMemAccessFlagsProtRead",
    "cudaMemAccessFlagsProtReadWrite": "hipMemAccessFlagsProtReadWrite",
    "cudaMemAdviseSetAccessedBy": "hipMemAdviseSetAccessedBy",
    "cudaMemAdviseSetPreferredLocation": "hipMemAdviseSetPreferredLocation",
    "cudaMemAdviseSetReadMostly": "hipMemAdviseSetReadMostly",
    "cudaMemAdviseUnsetAccessedBy": "hipMemAdviseUnsetAccessedBy",
    "cudaMemAdviseUnsetPreferredLocation": "hipMemAdviseUnsetPreferredLocation",
    "cudaMemAdviseUnsetReadMostly": "hipMemAdviseUnsetReadMostly",
    "cudaMemAllocationTypeInvalid": "hipMemAllocationTypeInvalid",
    "cudaMemAllocationTypeMax": "hipMemAllocationTypeMax",
    "cudaMemAllocationTypePinned": "hipMemAllocationTypePinned",
    "cudaMemHandleTypeNone": "hipMemHandleTypeNone",
    "cudaMemHandleTypePosixFileDescriptor": "hipMemHandleTypePosixFileDescriptor",
    "cudaMemHandleTypeWin32": "hipMemHandleTypeWin32",
    "cudaMemHandleTypeWin32Kmt": "hipMemHandleTypeWin32Kmt",
    "cudaMemLocationTypeDevice": "hipMemLocationTypeDevice",
    "cudaMemLocationTypeInvalid": "hipMemLocationTypeInvalid",
    "cudaMemPoolAttrReleaseThreshold": "hipMemPoolAttrReleaseThreshold",
    "cudaMemPoolAttrReservedMemCurrent": "hipMemPoolAttrReservedMemCurrent",
    "cudaMemPoolAttrReservedMemHigh": "hipMemPoolAttrReservedMemHigh",
    "cudaMemPoolAttrUsedMemCurrent": "hipMemPoolAttrUsedMemCurrent",
    "cudaMemPoolAttrUsedMemHigh": "hipMemPoolAttrUsedMemHigh",
    "cudaMemPoolReuseAllowInternalDependencies": "hipMemPoolReuseAllowInternalDependencies",
    "cudaMemPoolReuseAllowOpportunistic": "hipMemPoolReuseAllowOpportunistic",
    "cudaMemPoolReuseFollowEventDependencies": "hipMemPoolReuseFollowEventDependencies",
    "cudaMemRangeAttributeAccessedBy": "hipMemRangeAttributeAccessedBy",
    "cudaMemRangeAttributeLastPrefetchLocation": "hipMemRangeAttributeLastPrefetchLocation",
    "cudaMemRangeAttributePreferredLocation": "hipMemRangeAttributePreferredLocation",
    "cudaMemRangeAttributeReadMostly": "hipMemRangeAttributeReadMostly",
    "cudaMemcpyDefault": "hipMemcpyDefault",
    "cudaMemcpyDeviceToDevice": "hipMemcpyDeviceToDevice",
    "cudaMemcpyDeviceToHost": "hipMemcpyDeviceToHost",
    "cudaMemcpyHostToDevice": "hipMemcpyHostToDevice",
    "cudaMemcpyHostToHost": "hipMemcpyHostToHost",
    "cudaMemoryTypeDevice": "hipMemoryTypeDevice",
    "cudaMemoryTypeHost": "hipMemoryTypeHost",
    "cudaMemoryTypeManaged": "hipMemoryTypeManaged",
    "cudaReadModeElementType": "hipReadModeElementType",
    "cudaReadModeNormalizedFloat": "hipReadModeNormalizedFloat",
    "cudaResViewFormatFloat1": "hipResViewFormatFloat1",
    "cudaResViewFormatFloat2": "hipResViewFormatFloat2",
    "cudaResViewFormatFloat4": "hipResViewFormatFloat4",
    "cudaResViewFormatHalf1": "hipResViewFormatHalf1",
    "cudaResViewFormatHalf2": "hipResViewFormatHalf2",
    "cudaResViewFormatHalf4": "hipResViewFormatHalf4",
    "cudaResViewFormatNone": "hipResViewFormatNone",
    "cudaResViewFormatSignedBlockCompressed4": "hipResViewFormatSignedBlockCompressed4",
    "cudaResViewFormatSignedBlockCompressed5": "hipResViewFormatSignedBlockCompressed5",
    "cudaResViewFormatSignedBlockCompressed6H": "hipResViewFormatSignedBlockCompressed6H",
    "cudaResViewFormatSignedChar1": "hipResViewFormatSignedChar1",
    "cudaResViewFormatSignedChar2": "hipResViewFormatSignedChar2",
    "cudaResViewFormatSignedChar4": "hipResViewFormatSignedChar4",
    "cudaResViewFormatSignedInt1": "hipResViewFormatSignedInt1",
    "cudaResViewFormatSignedInt2": "hipResViewFormatSignedInt2",
    "cudaResViewFormatSignedInt4": "hipResViewFormatSignedInt4",
    "cudaResViewFormatSignedShort1": "hipResViewFormatSignedShort1",
    "cudaResViewFormatSignedShort2": "hipResViewFormatSignedShort2",
    "cudaResViewFormatSignedShort4": "hipResViewFormatSignedShort4",
    "cudaResViewFormatUnsignedBlockCompressed1": "hipResViewFormatUnsignedBlockCompressed1",
    "cudaResViewFormatUnsignedBlockCompressed2": "hipResViewFormatUnsignedBlockCompressed2",
    "cudaResViewFormatUnsignedBlockCompressed3": "hipResViewFormatUnsignedBlockCompressed3",
    "cudaResViewFormatUnsignedBlockCompressed4": "hipResViewFormatUnsignedBlockCompressed4",
    "cudaResViewFormatUnsignedBlockCompressed5": "hipResViewFormatUnsignedBlockCompressed5",
    "cudaResViewFormatUnsignedBlockCompressed6H": "hipResViewFormatUnsignedBlockCompressed6H",
    "cudaResViewFormatUnsignedBlockCompressed7": "hipResViewFormatUnsignedBlockCompressed7",
    "cudaResViewFormatUnsignedChar1": "hipResViewFormatUnsignedChar1",
    "cudaResViewFormatUnsignedChar2": "hipResViewFormatUnsignedChar2",
    "cudaResViewFormatUnsignedChar4": "hipResViewFormatUnsignedChar4",
    "cudaResViewFormatUnsignedInt1": "hipResViewFormatUnsignedInt1",
    "cudaResViewFormatUnsignedInt2": "hipResViewFormatUnsignedInt2",
    "cudaResViewFormatUnsignedInt4": "hipResViewFormatUnsignedInt4",
    "cudaResViewFormatUnsignedShort1": "hipResViewFormatUnsignedShort1",
    "cudaResViewFormatUnsignedShort2": "hipResViewFormatUnsignedShort2",
    "cudaResViewFormatUnsignedShort4": "hipResViewFormatUnsignedShort4",
    "cudaResourceTypeArray": "hipResourceTypeArray",
    "cudaResourceTypeLinear": "hipResourceTypeLinear",
    "cudaResourceTypeMipmappedArray": "hipResourceTypeMipmappedArray",
    "cudaResourceTypePitch2D": "hipResourceTypePitch2D",
    "cudaSharedMemBankSizeDefault": "hipSharedMemBankSizeDefault",
    "cudaSharedMemBankSizeEightByte": "hipSharedMemBankSizeEightByte",
    "cudaSharedMemBankSizeFourByte": "hipSharedMemBankSizeFourByte",
    "cudaStreamAddCaptureDependencies": "hipStreamAddCaptureDependencies",
    "cudaStreamCaptureModeGlobal": "hipStreamCaptureModeGlobal",
    "cudaStreamCaptureModeRelaxed": "hipStreamCaptureModeRelaxed",
    "cudaStreamCaptureModeThreadLocal": "hipStreamCaptureModeThreadLocal",
    "cudaStreamCaptureStatusActive": "hipStreamCaptureStatusActive",
    "cudaStreamCaptureStatusInvalidated": "hipStreamCaptureStatusInvalidated",
    "cudaStreamCaptureStatusNone": "hipStreamCaptureStatusNone",
    "cudaStreamSetCaptureDependencies": "hipStreamSetCaptureDependencies",
    "cudaSuccess": "hipSuccess",
    "cudaUserObjectNoDestructorSync": "hipUserObjectNoDestructorSync",
    "CUB_MAX": "CUB_MAX",
    "CUB_MIN": "CUB_MIN",
    "CUB_NAMESPACE_BEGIN": "BEGIN_HIPCUB_NAMESPACE",
    "CUB_NAMESPACE_END": "END_HIPCUB_NAMESPACE",
    "CUB_PTX_ARCH": "HIPCUB_ARCH",
    "CUB_PTX_WARP_THREADS": "HIPCUB_WARP_THREADS",
    "CUB_RUNTIME_FUNCTION": "HIPCUB_RUNTIME_FUNCTION",
    "CUB_STDERR": "HIPCUB_STDERR",
    "CUDA_ARRAY3D_CUBEMAP": "hipArrayCubemap",
    "CUDA_ARRAY3D_LAYERED": "hipArrayLayered",
    "CUDA_ARRAY3D_SURFACE_LDST": "hipArraySurfaceLoadStore",
    "CUDA_ARRAY3D_TEXTURE_GATHER": "hipArrayTextureGather",
    "CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC": "hipCooperativeLaunchMultiDeviceNoPostSync",
    "CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC": "hipCooperativeLaunchMultiDeviceNoPreSync",
    "CUDA_IPC_HANDLE_SIZE": "HIP_IPC_HANDLE_SIZE",
    "CURAND_VERSION": "HIPRAND_VERSION",
    "CU_DEVICE_CPU": "hipCpuDeviceId",
    "CU_DEVICE_INVALID": "hipInvalidDeviceId",
    "CU_IPC_HANDLE_SIZE": "HIP_IPC_HANDLE_SIZE",
    "CU_LAUNCH_PARAM_BUFFER_POINTER": "HIP_LAUNCH_PARAM_BUFFER_POINTER",
    "CU_LAUNCH_PARAM_BUFFER_SIZE": "HIP_LAUNCH_PARAM_BUFFER_SIZE",
    "CU_LAUNCH_PARAM_END": "HIP_LAUNCH_PARAM_END",
    "CU_MEMHOSTALLOC_DEVICEMAP": "hipHostMallocMapped",
    "CU_MEMHOSTALLOC_PORTABLE": "hipHostMallocPortable",
    "CU_MEMHOSTALLOC_WRITECOMBINED": "hipHostMallocWriteCombined",
    "CU_MEMHOSTREGISTER_DEVICEMAP": "hipHostRegisterMapped",
    "CU_MEMHOSTREGISTER_IOMEMORY": "hipHostRegisterIoMemory",
    "CU_MEMHOSTREGISTER_PORTABLE": "hipHostRegisterPortable",
    "CU_STREAM_PER_THREAD": "hipStreamPerThread",
    "CU_TRSA_OVERRIDE_FORMAT": "HIP_TRSA_OVERRIDE_FORMAT",
    "CU_TRSF_NORMALIZED_COORDINATES": "HIP_TRSF_NORMALIZED_COORDINATES",
    "CU_TRSF_READ_AS_INTEGER": "HIP_TRSF_READ_AS_INTEGER",
    "CU_TRSF_SRGB": "HIP_TRSF_SRGB",
    "CubDebug": "HipcubDebug",
    "REGISTER_CUDA_OPERATOR": "REGISTER_HIP_OPERATOR",
    "REGISTER_CUDA_OPERATOR_CREATOR": "REGISTER_HIP_OPERATOR_CREATOR",
    "_CubLog": "_HipcubLog",
    "__CUB_ALIGN_BYTES": "__HIPCUB_ALIGN_BYTES",
    "__CUDACC__": "__HIPCC__",
    "cudaArrayCubemap": "hipArrayCubemap",
    "cudaArrayDefault": "hipArrayDefault",
    "cudaArrayLayered": "hipArrayLayered",
    "cudaArraySurfaceLoadStore": "hipArraySurfaceLoadStore",
    "cudaArrayTextureGather": "hipArrayTextureGather",
    "cudaCooperativeLaunchMultiDeviceNoPostSync": "hipCooperativeLaunchMultiDeviceNoPostSync",
    "cudaCooperativeLaunchMultiDeviceNoPreSync": "hipCooperativeLaunchMultiDeviceNoPreSync",
    "cudaCpuDeviceId": "hipCpuDeviceId",
    "cudaDeviceBlockingSync": "hipDeviceScheduleBlockingSync",
    "cudaDeviceLmemResizeToMax": "hipDeviceLmemResizeToMax",
    "cudaDeviceMapHost": "hipDeviceMapHost",
    "cudaDeviceScheduleAuto": "hipDeviceScheduleAuto",
    "cudaDeviceScheduleBlockingSync": "hipDeviceScheduleBlockingSync",
    "cudaDeviceScheduleMask": "hipDeviceScheduleMask",
    "cudaDeviceScheduleSpin": "hipDeviceScheduleSpin",
    "cudaDeviceScheduleYield": "hipDeviceScheduleYield",
    "cudaEventBlockingSync": "hipEventBlockingSync",
    "cudaEventDefault": "hipEventDefault",
    "cudaEventDisableTiming": "hipEventDisableTiming",
    "cudaEventInterprocess": "hipEventInterprocess",
    "cudaHostAllocDefault": "hipHostMallocDefault",
    "cudaHostAllocMapped": "hipHostMallocMapped",
    "cudaHostAllocPortable": "hipHostMallocPortable",
    "cudaHostAllocWriteCombined": "hipHostMallocWriteCombined",
    "cudaHostRegisterDefault": "hipHostRegisterDefault",
    "cudaHostRegisterIoMemory": "hipHostRegisterIoMemory",
    "cudaHostRegisterMapped": "hipHostRegisterMapped",
    "cudaHostRegisterPortable": "hipHostRegisterPortable",
    "cudaInvalidDeviceId": "hipInvalidDeviceId",
    "cudaIpcMemLazyEnablePeerAccess": "hipIpcMemLazyEnablePeerAccess",
    "cudaMemAttachGlobal": "hipMemAttachGlobal",
    "cudaMemAttachHost": "hipMemAttachHost",
    "cudaMemAttachSingle": "hipMemAttachSingle",
    "cudaOccupancyDefault": "hipOccupancyDefault",
    "cudaStreamDefault": "hipStreamDefault",
    "cudaStreamNonBlocking": "hipStreamNonBlocking",
    "cudaStreamPerThread": "hipStreamPerThread",
    "cudaTextureType1D": "hipTextureType1D",
    "cudaTextureType1DLayered": "hipTextureType1DLayered",
    "cudaTextureType2D": "hipTextureType2D",
    "cudaTextureType2DLayered": "hipTextureType2DLayered",
    "cudaTextureType3D": "hipTextureType3D",
    "cudaTextureTypeCubemap": "hipTextureTypeCubemap",
    "cudaTextureTypeCubemapLayered": "hipTextureTypeCubemapLayered"
}
hip2cuda={
    "hipDrvGetErrorName": [
        "cuGetErrorName"
    ],
    "hipDrvGetErrorString": [
        "cuGetErrorString"
    ],
    "rocblas_axpy_ex": [
        "cublasAxpyEx"
    ],
    "rocblas_caxpy": [
        "cublasCaxpy",
        "cublasCaxpy_v2"
    ],
    "rocblas_ccopy": [
        "cublasCcopy",
        "cublasCcopy_v2"
    ],
    "rocblas_cdgmm": [
        "cublasCdgmm"
    ],
    "rocblas_cdotc": [
        "cublasCdotc",
        "cublasCdotc_v2"
    ],
    "rocblas_cdotu": [
        "cublasCdotu",
        "cublasCdotu_v2"
    ],
    "rocblas_cgbmv": [
        "cublasCgbmv",
        "cublasCgbmv_v2"
    ],
    "rocblas_cgeam": [
        "cublasCgeam"
    ],
    "rocblas_cgemm": [
        "cublasCgemm",
        "cublasCgemm_v2"
    ],
    "rocblas_cgemm_batched": [
        "cublasCgemmBatched"
    ],
    "rocblas_cgemm_strided_batched": [
        "cublasCgemmStridedBatched"
    ],
    "rocblas_cgemv": [
        "cublasCgemv",
        "cublasCgemv_v2"
    ],
    "rocblas_cgerc": [
        "cublasCgerc",
        "cublasCgerc_v2"
    ],
    "rocblas_cgeru": [
        "cublasCgeru",
        "cublasCgeru_v2"
    ],
    "rocblas_chbmv": [
        "cublasChbmv",
        "cublasChbmv_v2"
    ],
    "rocblas_chemm": [
        "cublasChemm",
        "cublasChemm_v2"
    ],
    "rocblas_chemv": [
        "cublasChemv",
        "cublasChemv_v2"
    ],
    "rocblas_cher": [
        "cublasCher",
        "cublasCher_v2"
    ],
    "rocblas_cher2": [
        "cublasCher2",
        "cublasCher2_v2"
    ],
    "rocblas_cher2k": [
        "cublasCher2k",
        "cublasCher2k_v2"
    ],
    "rocblas_cherk": [
        "cublasCherk",
        "cublasCherk_v2"
    ],
    "rocblas_cherkx": [
        "cublasCherkx"
    ],
    "rocblas_chpmv": [
        "cublasChpmv",
        "cublasChpmv_v2"
    ],
    "rocblas_chpr": [
        "cublasChpr",
        "cublasChpr_v2"
    ],
    "rocblas_chpr2": [
        "cublasChpr2",
        "cublasChpr2_v2"
    ],
    "rocblas_create_handle": [
        "cublasCreate",
        "cublasCreate_v2"
    ],
    "rocblas_crot": [
        "cublasCrot",
        "cublasCrot_v2"
    ],
    "rocblas_crotg": [
        "cublasCrotg",
        "cublasCrotg_v2"
    ],
    "rocblas_cscal": [
        "cublasCscal",
        "cublasCscal_v2"
    ],
    "rocblas_csrot": [
        "cublasCsrot",
        "cublasCsrot_v2"
    ],
    "rocblas_csscal": [
        "cublasCsscal",
        "cublasCsscal_v2"
    ],
    "rocblas_cswap": [
        "cublasCswap",
        "cublasCswap_v2"
    ],
    "rocblas_csymm": [
        "cublasCsymm",
        "cublasCsymm_v2"
    ],
    "rocblas_csymv": [
        "cublasCsymv",
        "cublasCsymv_v2"
    ],
    "rocblas_csyr": [
        "cublasCsyr",
        "cublasCsyr_v2"
    ],
    "rocblas_csyr2": [
        "cublasCsyr2",
        "cublasCsyr2_v2"
    ],
    "rocblas_csyr2k": [
        "cublasCsyr2k",
        "cublasCsyr2k_v2"
    ],
    "rocblas_csyrk": [
        "cublasCsyrk",
        "cublasCsyrk_v2"
    ],
    "rocblas_csyrkx": [
        "cublasCsyrkx"
    ],
    "rocblas_ctbmv": [
        "cublasCtbmv",
        "cublasCtbmv_v2"
    ],
    "rocblas_ctbsv": [
        "cublasCtbsv",
        "cublasCtbsv_v2"
    ],
    "rocblas_ctpmv": [
        "cublasCtpmv",
        "cublasCtpmv_v2"
    ],
    "rocblas_ctpsv": [
        "cublasCtpsv",
        "cublasCtpsv_v2"
    ],
    "rocblas_ctrmm_outofplace": [
        "cublasCtrmm",
        "cublasCtrmm_v2"
    ],
    "rocblas_ctrmv": [
        "cublasCtrmv",
        "cublasCtrmv_v2"
    ],
    "rocblas_ctrsm": [
        "cublasCtrsm",
        "cublasCtrsm_v2"
    ],
    "rocblas_ctrsm_batched": [
        "cublasCtrsmBatched"
    ],
    "rocblas_ctrsv": [
        "cublasCtrsv",
        "cublasCtrsv_v2"
    ],
    "rocblas_dasum": [
        "cublasDasum",
        "cublasDasum_v2"
    ],
    "rocblas_daxpy": [
        "cublasDaxpy",
        "cublasDaxpy_v2"
    ],
    "rocblas_dcopy": [
        "cublasDcopy",
        "cublasDcopy_v2"
    ],
    "rocblas_ddgmm": [
        "cublasDdgmm"
    ],
    "rocblas_ddot": [
        "cublasDdot",
        "cublasDdot_v2"
    ],
    "rocblas_destroy_handle": [
        "cublasDestroy",
        "cublasDestroy_v2"
    ],
    "rocblas_dgbmv": [
        "cublasDgbmv",
        "cublasDgbmv_v2"
    ],
    "rocblas_dgeam": [
        "cublasDgeam"
    ],
    "rocblas_dgemm": [
        "cublasDgemm",
        "cublasDgemm_v2"
    ],
    "rocblas_dgemm_batched": [
        "cublasDgemmBatched"
    ],
    "rocblas_dgemm_strided_batched": [
        "cublasDgemmStridedBatched"
    ],
    "rocblas_dgemv": [
        "cublasDgemv",
        "cublasDgemv_v2"
    ],
    "rocblas_dger": [
        "cublasDger",
        "cublasDger_v2"
    ],
    "rocblas_dnrm2": [
        "cublasDnrm2",
        "cublasDnrm2_v2"
    ],
    "rocblas_dot_ex": [
        "cublasDotEx"
    ],
    "rocblas_dotc_ex": [
        "cublasDotcEx"
    ],
    "rocblas_drot": [
        "cublasDrot",
        "cublasDrot_v2"
    ],
    "rocblas_drotg": [
        "cublasDrotg",
        "cublasDrotg_v2"
    ],
    "rocblas_drotm": [
        "cublasDrotm",
        "cublasDrotm_v2"
    ],
    "rocblas_drotmg": [
        "cublasDrotmg",
        "cublasDrotmg_v2"
    ],
    "rocblas_dsbmv": [
        "cublasDsbmv",
        "cublasDsbmv_v2"
    ],
    "rocblas_dscal": [
        "cublasDscal",
        "cublasDscal_v2"
    ],
    "rocblas_dspmv": [
        "cublasDspmv",
        "cublasDspmv_v2"
    ],
    "rocblas_dspr": [
        "cublasDspr",
        "cublasDspr_v2"
    ],
    "rocblas_dspr2": [
        "cublasDspr2",
        "cublasDspr2_v2"
    ],
    "rocblas_dswap": [
        "cublasDswap",
        "cublasDswap_v2"
    ],
    "rocblas_dsymm": [
        "cublasDsymm",
        "cublasDsymm_v2"
    ],
    "rocblas_dsymv": [
        "cublasDsymv",
        "cublasDsymv_v2"
    ],
    "rocblas_dsyr": [
        "cublasDsyr",
        "cublasDsyr_v2"
    ],
    "rocblas_dsyr2": [
        "cublasDsyr2",
        "cublasDsyr2_v2"
    ],
    "rocblas_dsyr2k": [
        "cublasDsyr2k",
        "cublasDsyr2k_v2"
    ],
    "rocblas_dsyrk": [
        "cublasDsyrk",
        "cublasDsyrk_v2"
    ],
    "rocblas_dsyrkx": [
        "cublasDsyrkx"
    ],
    "rocblas_dtbmv": [
        "cublasDtbmv",
        "cublasDtbmv_v2"
    ],
    "rocblas_dtbsv": [
        "cublasDtbsv",
        "cublasDtbsv_v2"
    ],
    "rocblas_dtpmv": [
        "cublasDtpmv",
        "cublasDtpmv_v2"
    ],
    "rocblas_dtpsv": [
        "cublasDtpsv",
        "cublasDtpsv_v2"
    ],
    "rocblas_dtrmm_outofplace": [
        "cublasDtrmm",
        "cublasDtrmm_v2"
    ],
    "rocblas_dtrmv": [
        "cublasDtrmv",
        "cublasDtrmv_v2"
    ],
    "rocblas_dtrsm": [
        "cublasDtrsm",
        "cublasDtrsm_v2"
    ],
    "rocblas_dtrsm_batched": [
        "cublasDtrsmBatched"
    ],
    "rocblas_dtrsv": [
        "cublasDtrsv",
        "cublasDtrsv_v2"
    ],
    "rocblas_dzasum": [
        "cublasDzasum",
        "cublasDzasum_v2"
    ],
    "rocblas_dznrm2": [
        "cublasDznrm2",
        "cublasDznrm2_v2"
    ],
    "rocblas_gemm_batched_ex": [
        "cublasGemmBatchedEx"
    ],
    "rocblas_gemm_ex": [
        "cublasGemmEx"
    ],
    "rocblas_gemm_strided_batched_ex": [
        "cublasGemmStridedBatchedEx"
    ],
    "rocblas_get_atomics_mode": [
        "cublasGetAtomicsMode"
    ],
    "rocblas_get_matrix": [
        "cublasGetMatrix"
    ],
    "rocblas_get_matrix_async": [
        "cublasGetMatrixAsync"
    ],
    "rocblas_get_pointer_mode": [
        "cublasGetPointerMode",
        "cublasSetPointerMode_v2"
    ],
    "rocblas_set_pointer_mode": [
        "cublasGetPointerMode_v2",
        "cublasSetPointerMode"
    ],
    "rocblas_status_to_string": [
        "cublasGetStatusString"
    ],
    "rocblas_get_stream": [
        "cublasGetStream",
        "cublasGetStream_v2"
    ],
    "rocblas_get_vector": [
        "cublasGetVector"
    ],
    "rocblas_get_vector_async": [
        "cublasGetVectorAsync"
    ],
    "rocblas_hgemm": [
        "cublasHgemm"
    ],
    "rocblas_hgemm_batched": [
        "cublasHgemmBatched"
    ],
    "rocblas_hgemm_strided_batched": [
        "cublasHgemmStridedBatched"
    ],
    "rocblas_icamax": [
        "cublasIcamax",
        "cublasIcamax_v2"
    ],
    "rocblas_icamin": [
        "cublasIcamin",
        "cublasIcamin_v2"
    ],
    "rocblas_idamax": [
        "cublasIdamax",
        "cublasIdamax_v2"
    ],
    "rocblas_idamin": [
        "cublasIdamin",
        "cublasIdamin_v2"
    ],
    "rocblas_initialize": [
        "cublasInit"
    ],
    "rocblas_isamax": [
        "cublasIsamax",
        "cublasIsamax_v2"
    ],
    "rocblas_isamin": [
        "cublasIsamin",
        "cublasIsamin_v2"
    ],
    "rocblas_izamax": [
        "cublasIzamax",
        "cublasIzamax_v2"
    ],
    "rocblas_izamin": [
        "cublasIzamin",
        "cublasIzamin_v2"
    ],
    "rocblas_nrm2_ex": [
        "cublasNrm2Ex"
    ],
    "rocblas_rot_ex": [
        "cublasRotEx"
    ],
    "rocblas_sasum": [
        "cublasSasum",
        "cublasSasum_v2"
    ],
    "rocblas_saxpy": [
        "cublasSaxpy",
        "cublasSaxpy_v2"
    ],
    "rocblas_scal_ex": [
        "cublasScalEx"
    ],
    "rocblas_scasum": [
        "cublasScasum",
        "cublasScasum_v2"
    ],
    "rocblas_scnrm2": [
        "cublasScnrm2",
        "cublasScnrm2_v2"
    ],
    "rocblas_scopy": [
        "cublasScopy",
        "cublasScopy_v2"
    ],
    "rocblas_sdgmm": [
        "cublasSdgmm"
    ],
    "rocblas_sdot": [
        "cublasSdot",
        "cublasSdot_v2"
    ],
    "rocblas_set_atomics_mode": [
        "cublasSetAtomicsMode"
    ],
    "rocblas_set_matrix": [
        "cublasSetMatrix"
    ],
    "rocblas_set_matrix_async": [
        "cublasSetMatrixAsync"
    ],
    "rocblas_set_stream": [
        "cublasSetStream",
        "cublasSetStream_v2"
    ],
    "rocblas_set_vector": [
        "cublasSetVector"
    ],
    "rocblas_set_vector_async": [
        "cublasSetVectorAsync"
    ],
    "rocblas_sgbmv": [
        "cublasSgbmv",
        "cublasSgbmv_v2"
    ],
    "rocblas_sgeam": [
        "cublasSgeam"
    ],
    "rocblas_sgemm": [
        "cublasSgemm",
        "cublasSgemm_v2"
    ],
    "rocblas_sgemm_batched": [
        "cublasSgemmBatched"
    ],
    "rocblas_sgemm_strided_batched": [
        "cublasSgemmStridedBatched"
    ],
    "rocblas_sgemv": [
        "cublasSgemv",
        "cublasSgemv_v2"
    ],
    "rocblas_sger": [
        "cublasSger",
        "cublasSger_v2"
    ],
    "rocblas_snrm2": [
        "cublasSnrm2",
        "cublasSnrm2_v2"
    ],
    "rocblas_srot": [
        "cublasSrot",
        "cublasSrot_v2"
    ],
    "rocblas_srotg": [
        "cublasSrotg",
        "cublasSrotg_v2"
    ],
    "rocblas_srotm": [
        "cublasSrotm",
        "cublasSrotm_v2"
    ],
    "rocblas_srotmg": [
        "cublasSrotmg",
        "cublasSrotmg_v2"
    ],
    "rocblas_ssbmv": [
        "cublasSsbmv",
        "cublasSsbmv_v2"
    ],
    "rocblas_sscal": [
        "cublasSscal",
        "cublasSscal_v2"
    ],
    "rocblas_sspmv": [
        "cublasSspmv",
        "cublasSspmv_v2"
    ],
    "rocblas_sspr": [
        "cublasSspr",
        "cublasSspr_v2"
    ],
    "rocblas_sspr2": [
        "cublasSspr2",
        "cublasSspr2_v2"
    ],
    "rocblas_sswap": [
        "cublasSswap",
        "cublasSswap_v2"
    ],
    "rocblas_ssymm": [
        "cublasSsymm",
        "cublasSsymm_v2"
    ],
    "rocblas_ssymv": [
        "cublasSsymv",
        "cublasSsymv_v2"
    ],
    "rocblas_ssyr": [
        "cublasSsyr",
        "cublasSsyr_v2"
    ],
    "rocblas_ssyr2": [
        "cublasSsyr2",
        "cublasSsyr2_v2"
    ],
    "rocblas_ssyr2k": [
        "cublasSsyr2k",
        "cublasSsyr2k_v2"
    ],
    "rocblas_ssyrk": [
        "cublasSsyrk",
        "cublasSsyrk_v2"
    ],
    "rocblas_ssyrkx": [
        "cublasSsyrkx"
    ],
    "rocblas_stbmv": [
        "cublasStbmv",
        "cublasStbmv_v2"
    ],
    "rocblas_stbsv": [
        "cublasStbsv",
        "cublasStbsv_v2"
    ],
    "rocblas_stpmv": [
        "cublasStpmv",
        "cublasStpmv_v2"
    ],
    "rocblas_stpsv": [
        "cublasStpsv",
        "cublasStpsv_v2"
    ],
    "rocblas_strmm_outofplace": [
        "cublasStrmm",
        "cublasStrmm_v2"
    ],
    "rocblas_strmv": [
        "cublasStrmv",
        "cublasStrmv_v2"
    ],
    "rocblas_strsm": [
        "cublasStrsm",
        "cublasStrsm_v2"
    ],
    "rocblas_strsm_batched": [
        "cublasStrsmBatched"
    ],
    "rocblas_strsv": [
        "cublasStrsv",
        "cublasStrsv_v2"
    ],
    "rocblas_zaxpy": [
        "cublasZaxpy",
        "cublasZaxpy_v2"
    ],
    "rocblas_zcopy": [
        "cublasZcopy",
        "cublasZcopy_v2"
    ],
    "rocblas_zdgmm": [
        "cublasZdgmm"
    ],
    "rocblas_zdotc": [
        "cublasZdotc",
        "cublasZdotc_v2"
    ],
    "rocblas_zdotu": [
        "cublasZdotu",
        "cublasZdotu_v2"
    ],
    "rocblas_zdrot": [
        "cublasZdrot",
        "cublasZdrot_v2"
    ],
    "rocblas_zdscal": [
        "cublasZdscal",
        "cublasZdscal_v2"
    ],
    "rocblas_zgbmv": [
        "cublasZgbmv",
        "cublasZgbmv_v2"
    ],
    "rocblas_zgeam": [
        "cublasZgeam"
    ],
    "rocblas_zgemm": [
        "cublasZgemm",
        "cublasZgemm_v2"
    ],
    "rocblas_zgemm_batched": [
        "cublasZgemmBatched"
    ],
    "rocblas_zgemm_strided_batched": [
        "cublasZgemmStridedBatched"
    ],
    "rocblas_zgemv": [
        "cublasZgemv",
        "cublasZgemv_v2"
    ],
    "rocblas_zgerc": [
        "cublasZgerc",
        "cublasZgerc_v2"
    ],
    "rocblas_zgeru": [
        "cublasZgeru",
        "cublasZgeru_v2"
    ],
    "rocblas_zhbmv": [
        "cublasZhbmv",
        "cublasZhbmv_v2"
    ],
    "rocblas_zhemm": [
        "cublasZhemm",
        "cublasZhemm_v2"
    ],
    "rocblas_zhemv": [
        "cublasZhemv",
        "cublasZhemv_v2"
    ],
    "rocblas_zher": [
        "cublasZher",
        "cublasZher_v2"
    ],
    "rocblas_zher2": [
        "cublasZher2",
        "cublasZher2_v2"
    ],
    "rocblas_zher2k": [
        "cublasZher2k",
        "cublasZher2k_v2"
    ],
    "rocblas_zherk": [
        "cublasZherk",
        "cublasZherk_v2"
    ],
    "rocblas_zherkx": [
        "cublasZherkx"
    ],
    "rocblas_zhpmv": [
        "cublasZhpmv",
        "cublasZhpmv_v2"
    ],
    "rocblas_zhpr": [
        "cublasZhpr",
        "cublasZhpr_v2"
    ],
    "rocblas_zhpr2": [
        "cublasZhpr2",
        "cublasZhpr2_v2"
    ],
    "rocblas_zrot": [
        "cublasZrot",
        "cublasZrot_v2"
    ],
    "rocblas_zrotg": [
        "cublasZrotg",
        "cublasZrotg_v2"
    ],
    "rocblas_zscal": [
        "cublasZscal",
        "cublasZscal_v2"
    ],
    "rocblas_zswap": [
        "cublasZswap",
        "cublasZswap_v2"
    ],
    "rocblas_zsymm": [
        "cublasZsymm",
        "cublasZsymm_v2"
    ],
    "rocblas_zsymv": [
        "cublasZsymv",
        "cublasZsymv_v2"
    ],
    "rocblas_zsyr": [
        "cublasZsyr",
        "cublasZsyr_v2"
    ],
    "rocblas_zsyr2": [
        "cublasZsyr2",
        "cublasZsyr2_v2"
    ],
    "rocblas_zsyr2k": [
        "cublasZsyr2k",
        "cublasZsyr2k_v2"
    ],
    "rocblas_zsyrk": [
        "cublasZsyrk",
        "cublasZsyrk_v2"
    ],
    "rocblas_zsyrkx": [
        "cublasZsyrkx"
    ],
    "rocblas_ztbmv": [
        "cublasZtbmv",
        "cublasZtbmv_v2"
    ],
    "rocblas_ztbsv": [
        "cublasZtbsv",
        "cublasZtbsv_v2"
    ],
    "rocblas_ztpmv": [
        "cublasZtpmv",
        "cublasZtpmv_v2"
    ],
    "rocblas_ztpsv": [
        "cublasZtpsv",
        "cublasZtpsv_v2"
    ],
    "rocblas_ztrmm_outofplace": [
        "cublasZtrmm",
        "cublasZtrmm_v2"
    ],
    "rocblas_ztrmv": [
        "cublasZtrmv",
        "cublasZtrmv_v2"
    ],
    "rocblas_ztrsm": [
        "cublasZtrsm",
        "cublasZtrsm_v2"
    ],
    "rocblas_ztrsm_batched": [
        "cublasZtrsmBatched"
    ],
    "rocblas_ztrsv": [
        "cublasZtrsv",
        "cublasZtrsv_v2"
    ],
    "rocblas_atomics_mode": [
        "cublasAtomicsMode_t"
    ],
    "_rocblas_handle": [
        "cublasContext"
    ],
    "rocblas_datatype": [
        "cublasDataType_t",
        "cudaDataType"
    ],
    "rocblas_diagonal": [
        "cublasDiagType_t"
    ],
    "rocblas_fill": [
        "cublasFillMode_t"
    ],
    "rocblas_gemm_algo": [
        "cublasGemmAlgo_t"
    ],
    "rocblas_handle": [
        "cublasHandle_t"
    ],
    "rocblas_operation": [
        "cublasOperation_t"
    ],
    "rocblas_pointer_mode": [
        "cublasPointerMode_t"
    ],
    "rocblas_side": [
        "cublasSideMode_t"
    ],
    "rocblas_status": [
        "cublasStatus",
        "cublasStatus_t"
    ],
    "rocblas_datatype_": [
        "cudaDataType_t"
    ],
    "rocblas_atomics_allowed": [
        "CUBLAS_ATOMICS_ALLOWED"
    ],
    "rocblas_atomics_not_allowed": [
        "CUBLAS_ATOMICS_NOT_ALLOWED"
    ],
    "rocblas_diagonal_non_unit": [
        "CUBLAS_DIAG_NON_UNIT"
    ],
    "rocblas_diagonal_unit": [
        "CUBLAS_DIAG_UNIT"
    ],
    "rocblas_fill_full": [
        "CUBLAS_FILL_MODE_FULL"
    ],
    "rocblas_fill_lower": [
        "CUBLAS_FILL_MODE_LOWER"
    ],
    "rocblas_fill_upper": [
        "CUBLAS_FILL_MODE_UPPER"
    ],
    "rocblas_gemm_algo_standard": [
        "CUBLAS_GEMM_DEFAULT",
        "CUBLAS_GEMM_DFALT"
    ],
    "rocblas_operation_conjugate_transpose": [
        "CUBLAS_OP_C",
        "CUBLAS_OP_HERMITAN"
    ],
    "rocblas_operation_none": [
        "CUBLAS_OP_N"
    ],
    "rocblas_operation_transpose": [
        "CUBLAS_OP_T"
    ],
    "rocblas_pointer_mode_device": [
        "CUBLAS_POINTER_MODE_DEVICE"
    ],
    "rocblas_pointer_mode_host": [
        "CUBLAS_POINTER_MODE_HOST"
    ],
    "rocblas_side_left": [
        "CUBLAS_SIDE_LEFT"
    ],
    "rocblas_side_right": [
        "CUBLAS_SIDE_RIGHT"
    ],
    "rocblas_status_not_implemented": [
        "CUBLAS_STATUS_ALLOC_FAILED"
    ],
    "rocblas_status_size_query_mismatch": [
        "CUBLAS_STATUS_ARCH_MISMATCH"
    ],
    "rocblas_status_memory_error": [
        "CUBLAS_STATUS_EXECUTION_FAILED"
    ],
    "rocblas_status_internal_error": [
        "CUBLAS_STATUS_INTERNAL_ERROR"
    ],
    "rocblas_status_invalid_pointer": [
        "CUBLAS_STATUS_INVALID_VALUE"
    ],
    "rocblas_status_invalid_size": [
        "CUBLAS_STATUS_MAPPING_ERROR"
    ],
    "rocblas_status_invalid_handle": [
        "CUBLAS_STATUS_NOT_INITIALIZED"
    ],
    "rocblas_status_perf_degraded": [
        "CUBLAS_STATUS_NOT_SUPPORTED"
    ],
    "rocblas_status_success": [
        "CUBLAS_STATUS_SUCCESS"
    ],
    "rocblas_datatype_bf16_c": [
        "CUDA_C_16BF"
    ],
    "rocblas_datatype_f16_c": [
        "CUDA_C_16F"
    ],
    "rocblas_datatype_f32_c": [
        "CUDA_C_32F"
    ],
    "rocblas_datatype_i32_c": [
        "CUDA_C_32I"
    ],
    "rocblas_datatype_u32_c": [
        "CUDA_C_32U"
    ],
    "rocblas_datatype_f64_c": [
        "CUDA_C_64F"
    ],
    "rocblas_datatype_i8_c": [
        "CUDA_C_8I"
    ],
    "rocblas_datatype_u8_c": [
        "CUDA_C_8U"
    ],
    "rocblas_datatype_bf16_r": [
        "CUDA_R_16BF"
    ],
    "rocblas_datatype_f16_r": [
        "CUDA_R_16F"
    ],
    "rocblas_datatype_f32_r": [
        "CUDA_R_32F"
    ],
    "rocblas_datatype_i32_r": [
        "CUDA_R_32I"
    ],
    "rocblas_datatype_u32_r": [
        "CUDA_R_32U"
    ],
    "rocblas_datatype_f64_r": [
        "CUDA_R_64F"
    ],
    "rocblas_datatype_i8_r": [
        "CUDA_R_8I"
    ],
    "rocblas_datatype_u8_r": [
        "CUDA_R_8U"
    ],
    "hipGetErrorName": [
        "cudaGetErrorName"
    ],
    "hipGetErrorString": [
        "cudaGetErrorString"
    ],
    "hipGetLastError": [
        "cudaGetLastError"
    ],
    "hipPeekAtLastError": [
        "cudaPeekAtLastError"
    ],
    "hipInit": [
        "cuInit"
    ],
    "hipDriverGetVersion": [
        "cuDriverGetVersion",
        "cudaDriverGetVersion"
    ],
    "hipRuntimeGetVersion": [
        "cudaRuntimeGetVersion"
    ],
    "hipDeviceComputeCapability": [
        "cuDeviceComputeCapability"
    ],
    "hipDeviceGet": [
        "cuDeviceGet"
    ],
    "hipDeviceGetAttribute": [
        "cuDeviceGetAttribute",
        "cudaDeviceGetAttribute"
    ],
    "hipGetDeviceCount": [
        "cuDeviceGetCount",
        "cudaGetDeviceCount"
    ],
    "hipDeviceGetDefaultMemPool": [
        "cuDeviceGetDefaultMemPool",
        "cudaDeviceGetDefaultMemPool"
    ],
    "hipDeviceGetMemPool": [
        "cuDeviceGetMemPool",
        "cudaDeviceGetMemPool"
    ],
    "hipDeviceGetName": [
        "cuDeviceGetName"
    ],
    "hipDeviceGetUuid": [
        "cuDeviceGetUuid",
        "cuDeviceGetUuid_v2"
    ],
    "hipDeviceSetMemPool": [
        "cuDeviceSetMemPool",
        "cudaDeviceSetMemPool"
    ],
    "hipDeviceTotalMem": [
        "cuDeviceTotalMem",
        "cuDeviceTotalMem_v2"
    ],
    "hipChooseDevice": [
        "cudaChooseDevice"
    ],
    "hipDeviceGetByPCIBusId": [
        "cudaDeviceGetByPCIBusId",
        "cuDeviceGetByPCIBusId"
    ],
    "hipDeviceGetCacheConfig": [
        "cudaDeviceGetCacheConfig",
        "cudaThreadGetCacheConfig"
    ],
    "hipDeviceGetLimit": [
        "cudaDeviceGetLimit",
        "cuCtxGetLimit"
    ],
    "hipDeviceGetP2PAttribute": [
        "cudaDeviceGetP2PAttribute",
        "cuDeviceGetP2PAttribute"
    ],
    "hipDeviceGetPCIBusId": [
        "cudaDeviceGetPCIBusId",
        "cuDeviceGetPCIBusId"
    ],
    "hipDeviceGetSharedMemConfig": [
        "cudaDeviceGetSharedMemConfig"
    ],
    "hipDeviceGetStreamPriorityRange": [
        "cudaDeviceGetStreamPriorityRange",
        "cuCtxGetStreamPriorityRange"
    ],
    "hipDeviceReset": [
        "cudaDeviceReset",
        "cudaThreadExit"
    ],
    "hipDeviceSetCacheConfig": [
        "cudaDeviceSetCacheConfig",
        "cudaThreadSetCacheConfig"
    ],
    "hipDeviceSetLimit": [
        "cudaDeviceSetLimit",
        "cuCtxSetLimit"
    ],
    "hipDeviceSetSharedMemConfig": [
        "cudaDeviceSetSharedMemConfig"
    ],
    "hipDeviceSynchronize": [
        "cudaDeviceSynchronize",
        "cudaThreadSynchronize"
    ],
    "hipFuncSetCacheConfig": [
        "cudaFuncSetCacheConfig"
    ],
    "hipGetDevice": [
        "cudaGetDevice"
    ],
    "hipGetDeviceFlags": [
        "cudaGetDeviceFlags"
    ],
    "hipGetDeviceProperties": [
        "cudaGetDeviceProperties"
    ],
    "hipIpcCloseMemHandle": [
        "cudaIpcCloseMemHandle",
        "cuIpcCloseMemHandle"
    ],
    "hipIpcGetEventHandle": [
        "cudaIpcGetEventHandle",
        "cuIpcGetEventHandle"
    ],
    "hipIpcGetMemHandle": [
        "cudaIpcGetMemHandle",
        "cuIpcGetMemHandle"
    ],
    "hipIpcOpenEventHandle": [
        "cudaIpcOpenEventHandle",
        "cuIpcOpenEventHandle"
    ],
    "hipIpcOpenMemHandle": [
        "cudaIpcOpenMemHandle",
        "cuIpcOpenMemHandle"
    ],
    "hipSetDevice": [
        "cudaSetDevice"
    ],
    "hipSetDeviceFlags": [
        "cudaSetDeviceFlags"
    ],
    "hipCtxCreate": [
        "cuCtxCreate",
        "cuCtxCreate_v2"
    ],
    "hipCtxDestroy": [
        "cuCtxDestroy",
        "cuCtxDestroy_v2"
    ],
    "hipCtxGetApiVersion": [
        "cuCtxGetApiVersion"
    ],
    "hipCtxGetCacheConfig": [
        "cuCtxGetCacheConfig"
    ],
    "hipCtxGetCurrent": [
        "cuCtxGetCurrent"
    ],
    "hipCtxGetDevice": [
        "cuCtxGetDevice"
    ],
    "hipCtxGetFlags": [
        "cuCtxGetFlags"
    ],
    "hipCtxGetSharedMemConfig": [
        "cuCtxGetSharedMemConfig"
    ],
    "hipCtxPopCurrent": [
        "cuCtxPopCurrent",
        "cuCtxPopCurrent_v2"
    ],
    "hipCtxPushCurrent": [
        "cuCtxPushCurrent",
        "cuCtxPushCurrent_v2"
    ],
    "hipCtxSetCacheConfig": [
        "cuCtxSetCacheConfig"
    ],
    "hipCtxSetCurrent": [
        "cuCtxSetCurrent"
    ],
    "hipCtxSetSharedMemConfig": [
        "cuCtxSetSharedMemConfig"
    ],
    "hipCtxSynchronize": [
        "cuCtxSynchronize"
    ],
    "hipDevicePrimaryCtxGetState": [
        "cuDevicePrimaryCtxGetState"
    ],
    "hipDevicePrimaryCtxRelease": [
        "cuDevicePrimaryCtxRelease",
        "cuDevicePrimaryCtxRelease_v2"
    ],
    "hipDevicePrimaryCtxReset": [
        "cuDevicePrimaryCtxReset",
        "cuDevicePrimaryCtxReset_v2"
    ],
    "hipDevicePrimaryCtxRetain": [
        "cuDevicePrimaryCtxRetain"
    ],
    "hipDevicePrimaryCtxSetFlags": [
        "cuDevicePrimaryCtxSetFlags",
        "cuDevicePrimaryCtxSetFlags_v2"
    ],
    "hiprtcLinkAddData": [
        "cuLinkAddData",
        "cuLinkAddData_v2"
    ],
    "hiprtcLinkAddFile": [
        "cuLinkAddFile",
        "cuLinkAddFile_v2"
    ],
    "hiprtcLinkComplete": [
        "cuLinkComplete"
    ],
    "hiprtcLinkCreate": [
        "cuLinkCreate",
        "cuLinkCreate_v2"
    ],
    "hiprtcLinkDestroy": [
        "cuLinkDestroy"
    ],
    "hipModuleGetFunction": [
        "cuModuleGetFunction"
    ],
    "hipModuleGetGlobal": [
        "cuModuleGetGlobal",
        "cuModuleGetGlobal_v2"
    ],
    "hipModuleGetTexRef": [
        "cuModuleGetTexRef"
    ],
    "hipModuleLoad": [
        "cuModuleLoad"
    ],
    "hipModuleLoadData": [
        "cuModuleLoadData"
    ],
    "hipModuleLoadDataEx": [
        "cuModuleLoadDataEx"
    ],
    "hipModuleUnload": [
        "cuModuleUnload"
    ],
    "hipArray3DCreate": [
        "cuArray3DCreate",
        "cuArray3DCreate_v2"
    ],
    "hipArrayCreate": [
        "cuArrayCreate",
        "cuArrayCreate_v2"
    ],
    "hipArrayDestroy": [
        "cuArrayDestroy"
    ],
    "hipMalloc": [
        "cuMemAlloc",
        "cuMemAlloc_v2",
        "cudaMalloc"
    ],
    "hipMemAllocHost": [
        "cuMemAllocHost",
        "cuMemAllocHost_v2"
    ],
    "hipMallocManaged": [
        "cuMemAllocManaged",
        "cudaMallocManaged"
    ],
    "hipMemAllocPitch": [
        "cuMemAllocPitch",
        "cuMemAllocPitch_v2"
    ],
    "hipFree": [
        "cuMemFree",
        "cuMemFree_v2",
        "cudaFree"
    ],
    "hipHostFree": [
        "cuMemFreeHost",
        "cudaFreeHost"
    ],
    "hipMemGetAddressRange": [
        "cuMemGetAddressRange",
        "cuMemGetAddressRange_v2"
    ],
    "hipMemGetInfo": [
        "cuMemGetInfo",
        "cuMemGetInfo_v2",
        "cudaMemGetInfo"
    ],
    "hipHostAlloc": [
        "cuMemHostAlloc",
        "cudaHostAlloc"
    ],
    "hipHostGetDevicePointer": [
        "cuMemHostGetDevicePointer",
        "cuMemHostGetDevicePointer_v2",
        "cudaHostGetDevicePointer"
    ],
    "hipHostGetFlags": [
        "cuMemHostGetFlags",
        "cudaHostGetFlags"
    ],
    "hipHostRegister": [
        "cuMemHostRegister",
        "cuMemHostRegister_v2",
        "cudaHostRegister"
    ],
    "hipHostUnregister": [
        "cuMemHostUnregister",
        "cudaHostUnregister"
    ],
    "hipMemcpyParam2D": [
        "cuMemcpy2D",
        "cuMemcpy2D_v2"
    ],
    "hipMemcpyParam2DAsync": [
        "cuMemcpy2DAsync",
        "cuMemcpy2DAsync_v2"
    ],
    "hipDrvMemcpy2DUnaligned": [
        "cuMemcpy2DUnaligned",
        "cuMemcpy2DUnaligned_v2"
    ],
    "hipDrvMemcpy3D": [
        "cuMemcpy3D",
        "cuMemcpy3D_v2"
    ],
    "hipDrvMemcpy3DAsync": [
        "cuMemcpy3DAsync",
        "cuMemcpy3DAsync_v2"
    ],
    "hipMemcpyAtoH": [
        "cuMemcpyAtoH",
        "cuMemcpyAtoH_v2"
    ],
    "hipMemcpyDtoD": [
        "cuMemcpyDtoD",
        "cuMemcpyDtoD_v2"
    ],
    "hipMemcpyDtoDAsync": [
        "cuMemcpyDtoDAsync",
        "cuMemcpyDtoDAsync_v2"
    ],
    "hipMemcpyDtoH": [
        "cuMemcpyDtoH",
        "cuMemcpyDtoH_v2"
    ],
    "hipMemcpyDtoHAsync": [
        "cuMemcpyDtoHAsync",
        "cuMemcpyDtoHAsync_v2"
    ],
    "hipMemcpyHtoA": [
        "cuMemcpyHtoA",
        "cuMemcpyHtoA_v2"
    ],
    "hipMemcpyHtoD": [
        "cuMemcpyHtoD",
        "cuMemcpyHtoD_v2"
    ],
    "hipMemcpyHtoDAsync": [
        "cuMemcpyHtoDAsync",
        "cuMemcpyHtoDAsync_v2"
    ],
    "hipMemsetD16": [
        "cuMemsetD16",
        "cuMemsetD16_v2"
    ],
    "hipMemsetD16Async": [
        "cuMemsetD16Async"
    ],
    "hipMemsetD32": [
        "cuMemsetD32",
        "cuMemsetD32_v2"
    ],
    "hipMemsetD32Async": [
        "cuMemsetD32Async"
    ],
    "hipMemsetD8": [
        "cuMemsetD8",
        "cuMemsetD8_v2"
    ],
    "hipMemsetD8Async": [
        "cuMemsetD8Async"
    ],
    "hipMipmappedArrayCreate": [
        "cuMipmappedArrayCreate"
    ],
    "hipMipmappedArrayDestroy": [
        "cuMipmappedArrayDestroy"
    ],
    "hipMipmappedArrayGetLevel": [
        "cuMipmappedArrayGetLevel"
    ],
    "hipFreeArray": [
        "cudaFreeArray"
    ],
    "hipFreeAsync": [
        "cudaFreeAsync",
        "cuMemFreeAsync"
    ],
    "hipFreeMipmappedArray": [
        "cudaFreeMipmappedArray"
    ],
    "hipGetMipmappedArrayLevel": [
        "cudaGetMipmappedArrayLevel"
    ],
    "hipGetSymbolAddress": [
        "cudaGetSymbolAddress"
    ],
    "hipGetSymbolSize": [
        "cudaGetSymbolSize"
    ],
    "hipMalloc3D": [
        "cudaMalloc3D"
    ],
    "hipMalloc3DArray": [
        "cudaMalloc3DArray"
    ],
    "hipMallocArray": [
        "cudaMallocArray"
    ],
    "hipMallocAsync": [
        "cudaMallocAsync",
        "cuMemAllocAsync"
    ],
    "hipMallocFromPoolAsync": [
        "cudaMallocFromPoolAsync",
        "cuMemAllocFromPoolAsync"
    ],
    "hipHostMalloc": [
        "cudaMallocHost"
    ],
    "hipMallocMipmappedArray": [
        "cudaMallocMipmappedArray"
    ],
    "hipMallocPitch": [
        "cudaMallocPitch"
    ],
    "hipMemAdvise": [
        "cudaMemAdvise",
        "cuMemAdvise"
    ],
    "hipMemPoolCreate": [
        "cudaMemPoolCreate",
        "cuMemPoolCreate"
    ],
    "hipMemPoolDestroy": [
        "cudaMemPoolDestroy",
        "cuMemPoolDestroy"
    ],
    "hipMemPoolExportPointer": [
        "cudaMemPoolExportPointer",
        "cuMemPoolExportPointer"
    ],
    "hipMemPoolExportToShareableHandle": [
        "cudaMemPoolExportToShareableHandle",
        "cuMemPoolExportToShareableHandle"
    ],
    "hipMemPoolGetAccess": [
        "cudaMemPoolGetAccess",
        "cuMemPoolGetAccess"
    ],
    "hipMemPoolGetAttribute": [
        "cudaMemPoolGetAttribute",
        "cuMemPoolGetAttribute"
    ],
    "hipMemPoolImportFromShareableHandle": [
        "cudaMemPoolImportFromShareableHandle",
        "cuMemPoolImportFromShareableHandle"
    ],
    "hipMemPoolImportPointer": [
        "cudaMemPoolImportPointer",
        "cuMemPoolImportPointer"
    ],
    "hipMemPoolSetAccess": [
        "cudaMemPoolSetAccess",
        "cuMemPoolSetAccess"
    ],
    "hipMemPoolSetAttribute": [
        "cudaMemPoolSetAttribute",
        "cuMemPoolSetAttribute"
    ],
    "hipMemPoolTrimTo": [
        "cudaMemPoolTrimTo",
        "cuMemPoolTrimTo"
    ],
    "hipMemPrefetchAsync": [
        "cudaMemPrefetchAsync",
        "cuMemPrefetchAsync"
    ],
    "hipMemRangeGetAttribute": [
        "cudaMemRangeGetAttribute",
        "cuMemRangeGetAttribute"
    ],
    "hipMemRangeGetAttributes": [
        "cudaMemRangeGetAttributes",
        "cuMemRangeGetAttributes"
    ],
    "hipMemcpy": [
        "cudaMemcpy"
    ],
    "hipMemcpy2D": [
        "cudaMemcpy2D"
    ],
    "hipMemcpy2DAsync": [
        "cudaMemcpy2DAsync"
    ],
    "hipMemcpy2DFromArray": [
        "cudaMemcpy2DFromArray"
    ],
    "hipMemcpy2DFromArrayAsync": [
        "cudaMemcpy2DFromArrayAsync"
    ],
    "hipMemcpy2DToArray": [
        "cudaMemcpy2DToArray"
    ],
    "hipMemcpy2DToArrayAsync": [
        "cudaMemcpy2DToArrayAsync"
    ],
    "hipMemcpy3D": [
        "cudaMemcpy3D"
    ],
    "hipMemcpy3DAsync": [
        "cudaMemcpy3DAsync"
    ],
    "hipMemcpyAsync": [
        "cudaMemcpyAsync"
    ],
    "hipMemcpyFromArray": [
        "cudaMemcpyFromArray"
    ],
    "hipMemcpyFromSymbol": [
        "cudaMemcpyFromSymbol"
    ],
    "hipMemcpyFromSymbolAsync": [
        "cudaMemcpyFromSymbolAsync"
    ],
    "hipMemcpyPeer": [
        "cudaMemcpyPeer"
    ],
    "hipMemcpyPeerAsync": [
        "cudaMemcpyPeerAsync"
    ],
    "hipMemcpyToArray": [
        "cudaMemcpyToArray"
    ],
    "hipMemcpyToSymbol": [
        "cudaMemcpyToSymbol"
    ],
    "hipMemcpyToSymbolAsync": [
        "cudaMemcpyToSymbolAsync"
    ],
    "hipMemset": [
        "cudaMemset"
    ],
    "hipMemset2D": [
        "cudaMemset2D"
    ],
    "hipMemset2DAsync": [
        "cudaMemset2DAsync"
    ],
    "hipMemset3D": [
        "cudaMemset3D"
    ],
    "hipMemset3DAsync": [
        "cudaMemset3DAsync"
    ],
    "hipMemsetAsync": [
        "cudaMemsetAsync"
    ],
    "make_hipExtent": [
        "make_cudaExtent"
    ],
    "make_hipPitchedPtr": [
        "make_cudaPitchedPtr"
    ],
    "make_hipPos": [
        "make_cudaPos"
    ],
    "hipMemAddressFree": [
        "cuMemAddressFree"
    ],
    "hipMemAddressReserve": [
        "cuMemAddressReserve"
    ],
    "hipMemCreate": [
        "cuMemCreate"
    ],
    "hipMemExportToShareableHandle": [
        "cuMemExportToShareableHandle"
    ],
    "hipMemGetAccess": [
        "cuMemGetAccess"
    ],
    "hipMemGetAllocationGranularity": [
        "cuMemGetAllocationGranularity"
    ],
    "hipMemGetAllocationPropertiesFromHandle": [
        "cuMemGetAllocationPropertiesFromHandle"
    ],
    "hipMemImportFromShareableHandle": [
        "cuMemImportFromShareableHandle"
    ],
    "hipMemMap": [
        "cuMemMap"
    ],
    "hipMemMapArrayAsync": [
        "cuMemMapArrayAsync"
    ],
    "hipMemRelease": [
        "cuMemRelease"
    ],
    "hipMemRetainAllocationHandle": [
        "cuMemRetainAllocationHandle"
    ],
    "hipMemSetAccess": [
        "cuMemSetAccess"
    ],
    "hipMemUnmap": [
        "cuMemUnmap"
    ],
    "hipPointerGetAttribute": [
        "cuPointerGetAttribute"
    ],
    "hipDrvPointerGetAttributes": [
        "cuPointerGetAttributes"
    ],
    "hipPointerGetAttributes": [
        "cudaPointerGetAttributes"
    ],
    "hipStreamAddCallback": [
        "cuStreamAddCallback",
        "cudaStreamAddCallback"
    ],
    "hipStreamAttachMemAsync": [
        "cuStreamAttachMemAsync",
        "cudaStreamAttachMemAsync"
    ],
    "hipStreamBeginCapture": [
        "cuStreamBeginCapture",
        "cuStreamBeginCapture_v2",
        "cudaStreamBeginCapture"
    ],
    "hipStreamCreateWithFlags": [
        "cuStreamCreate",
        "cudaStreamCreateWithFlags"
    ],
    "hipStreamCreateWithPriority": [
        "cuStreamCreateWithPriority",
        "cudaStreamCreateWithPriority"
    ],
    "hipStreamDestroy": [
        "cuStreamDestroy",
        "cuStreamDestroy_v2",
        "cudaStreamDestroy"
    ],
    "hipStreamEndCapture": [
        "cuStreamEndCapture",
        "cudaStreamEndCapture"
    ],
    "hipStreamGetCaptureInfo": [
        "cuStreamGetCaptureInfo",
        "cudaStreamGetCaptureInfo"
    ],
    "hipStreamGetCaptureInfo_v2": [
        "cuStreamGetCaptureInfo_v2"
    ],
    "hipStreamGetFlags": [
        "cuStreamGetFlags",
        "cudaStreamGetFlags"
    ],
    "hipStreamGetPriority": [
        "cuStreamGetPriority",
        "cudaStreamGetPriority"
    ],
    "hipStreamIsCapturing": [
        "cuStreamIsCapturing",
        "cudaStreamIsCapturing"
    ],
    "hipStreamQuery": [
        "cuStreamQuery",
        "cudaStreamQuery"
    ],
    "hipStreamSynchronize": [
        "cuStreamSynchronize",
        "cudaStreamSynchronize"
    ],
    "hipStreamUpdateCaptureDependencies": [
        "cuStreamUpdateCaptureDependencies"
    ],
    "hipStreamWaitEvent": [
        "cuStreamWaitEvent",
        "cudaStreamWaitEvent"
    ],
    "hipThreadExchangeStreamCaptureMode": [
        "cuThreadExchangeStreamCaptureMode",
        "cudaThreadExchangeStreamCaptureMode"
    ],
    "hipStreamCreate": [
        "cudaStreamCreate"
    ],
    "hipEventCreateWithFlags": [
        "cuEventCreate",
        "cudaEventCreateWithFlags"
    ],
    "hipEventDestroy": [
        "cuEventDestroy",
        "cuEventDestroy_v2",
        "cudaEventDestroy"
    ],
    "hipEventElapsedTime": [
        "cuEventElapsedTime",
        "cudaEventElapsedTime"
    ],
    "hipEventQuery": [
        "cuEventQuery",
        "cudaEventQuery"
    ],
    "hipEventRecord": [
        "cuEventRecord",
        "cudaEventRecord"
    ],
    "hipEventSynchronize": [
        "cuEventSynchronize",
        "cudaEventSynchronize"
    ],
    "hipEventCreate": [
        "cudaEventCreate"
    ],
    "hipDestroyExternalMemory": [
        "cuDestroyExternalMemory",
        "cudaDestroyExternalMemory"
    ],
    "hipDestroyExternalSemaphore": [
        "cuDestroyExternalSemaphore",
        "cudaDestroyExternalSemaphore"
    ],
    "hipExternalMemoryGetMappedBuffer": [
        "cuExternalMemoryGetMappedBuffer",
        "cudaExternalMemoryGetMappedBuffer"
    ],
    "hipImportExternalMemory": [
        "cuImportExternalMemory",
        "cudaImportExternalMemory"
    ],
    "hipImportExternalSemaphore": [
        "cuImportExternalSemaphore",
        "cudaImportExternalSemaphore"
    ],
    "hipSignalExternalSemaphoresAsync": [
        "cuSignalExternalSemaphoresAsync",
        "cudaSignalExternalSemaphoresAsync"
    ],
    "hipWaitExternalSemaphoresAsync": [
        "cuWaitExternalSemaphoresAsync",
        "cudaWaitExternalSemaphoresAsync"
    ],
    "hipStreamWaitValue32": [
        "cuStreamWaitValue32",
        "cuStreamWaitValue32_v2"
    ],
    "hipStreamWaitValue64": [
        "cuStreamWaitValue64",
        "cuStreamWaitValue64_v2"
    ],
    "hipStreamWriteValue32": [
        "cuStreamWriteValue32",
        "cuStreamWriteValue32_v2"
    ],
    "hipStreamWriteValue64": [
        "cuStreamWriteValue64",
        "cuStreamWriteValue64_v2"
    ],
    "hipFuncGetAttribute": [
        "cuFuncGetAttribute"
    ],
    "hipLaunchHostFunc": [
        "cuLaunchHostFunc",
        "cudaLaunchHostFunc"
    ],
    "hipModuleLaunchKernel": [
        "cuLaunchKernel"
    ],
    "hipConfigureCall": [
        "cudaConfigureCall"
    ],
    "hipFuncGetAttributes": [
        "cudaFuncGetAttributes"
    ],
    "hipFuncSetAttribute": [
        "cudaFuncSetAttribute"
    ],
    "hipFuncSetSharedMemConfig": [
        "cudaFuncSetSharedMemConfig"
    ],
    "hipLaunchByPtr": [
        "cudaLaunch"
    ],
    "hipLaunchCooperativeKernel": [
        "cudaLaunchCooperativeKernel"
    ],
    "hipLaunchCooperativeKernelMultiDevice": [
        "cudaLaunchCooperativeKernelMultiDevice"
    ],
    "hipLaunchKernel": [
        "cudaLaunchKernel"
    ],
    "hipSetupArgument": [
        "cudaSetupArgument"
    ],
    "hipDeviceGetGraphMemAttribute": [
        "cuDeviceGetGraphMemAttribute",
        "cudaDeviceGetGraphMemAttribute"
    ],
    "hipDeviceGraphMemTrim": [
        "cuDeviceGraphMemTrim",
        "cudaDeviceGraphMemTrim"
    ],
    "hipDeviceSetGraphMemAttribute": [
        "cuDeviceSetGraphMemAttribute",
        "cudaDeviceSetGraphMemAttribute"
    ],
    "hipGraphAddChildGraphNode": [
        "cuGraphAddChildGraphNode",
        "cudaGraphAddChildGraphNode"
    ],
    "hipGraphAddDependencies": [
        "cuGraphAddDependencies",
        "cudaGraphAddDependencies"
    ],
    "hipGraphAddEmptyNode": [
        "cuGraphAddEmptyNode",
        "cudaGraphAddEmptyNode"
    ],
    "hipGraphAddEventRecordNode": [
        "cuGraphAddEventRecordNode",
        "cudaGraphAddEventRecordNode"
    ],
    "hipGraphAddEventWaitNode": [
        "cuGraphAddEventWaitNode",
        "cudaGraphAddEventWaitNode"
    ],
    "hipGraphAddHostNode": [
        "cuGraphAddHostNode",
        "cudaGraphAddHostNode"
    ],
    "hipGraphAddKernelNode": [
        "cuGraphAddKernelNode",
        "cudaGraphAddKernelNode"
    ],
    "hipGraphChildGraphNodeGetGraph": [
        "cuGraphChildGraphNodeGetGraph",
        "cudaGraphChildGraphNodeGetGraph"
    ],
    "hipGraphClone": [
        "cuGraphClone",
        "cudaGraphClone"
    ],
    "hipGraphCreate": [
        "cuGraphCreate",
        "cudaGraphCreate"
    ],
    "hipGraphDestroy": [
        "cuGraphDestroy",
        "cudaGraphDestroy"
    ],
    "hipGraphDestroyNode": [
        "cuGraphDestroyNode",
        "cudaGraphDestroyNode"
    ],
    "hipGraphEventRecordNodeGetEvent": [
        "cuGraphEventRecordNodeGetEvent",
        "cudaGraphEventRecordNodeGetEvent"
    ],
    "hipGraphEventRecordNodeSetEvent": [
        "cuGraphEventRecordNodeSetEvent",
        "cudaGraphEventRecordNodeSetEvent"
    ],
    "hipGraphEventWaitNodeGetEvent": [
        "cuGraphEventWaitNodeGetEvent",
        "cudaGraphEventWaitNodeGetEvent"
    ],
    "hipGraphEventWaitNodeSetEvent": [
        "cuGraphEventWaitNodeSetEvent",
        "cudaGraphEventWaitNodeSetEvent"
    ],
    "hipGraphExecChildGraphNodeSetParams": [
        "cuGraphExecChildGraphNodeSetParams",
        "cudaGraphExecChildGraphNodeSetParams"
    ],
    "hipGraphExecDestroy": [
        "cuGraphExecDestroy",
        "cudaGraphExecDestroy"
    ],
    "hipGraphExecEventRecordNodeSetEvent": [
        "cuGraphExecEventRecordNodeSetEvent",
        "cudaGraphExecEventRecordNodeSetEvent"
    ],
    "hipGraphExecEventWaitNodeSetEvent": [
        "cuGraphExecEventWaitNodeSetEvent",
        "cudaGraphExecEventWaitNodeSetEvent"
    ],
    "hipGraphExecHostNodeSetParams": [
        "cuGraphExecHostNodeSetParams",
        "cudaGraphExecHostNodeSetParams"
    ],
    "hipGraphExecKernelNodeSetParams": [
        "cuGraphExecKernelNodeSetParams",
        "cudaGraphExecKernelNodeSetParams"
    ],
    "hipGraphExecUpdate": [
        "cuGraphExecUpdate",
        "cudaGraphExecUpdate"
    ],
    "hipGraphGetEdges": [
        "cuGraphGetEdges",
        "cudaGraphGetEdges"
    ],
    "hipGraphGetNodes": [
        "cuGraphGetNodes",
        "cudaGraphGetNodes"
    ],
    "hipGraphGetRootNodes": [
        "cuGraphGetRootNodes",
        "cudaGraphGetRootNodes"
    ],
    "hipGraphHostNodeGetParams": [
        "cuGraphHostNodeGetParams",
        "cudaGraphHostNodeGetParams"
    ],
    "hipGraphHostNodeSetParams": [
        "cuGraphHostNodeSetParams",
        "cudaGraphHostNodeSetParams"
    ],
    "hipGraphInstantiate": [
        "cuGraphInstantiate",
        "cuGraphInstantiate_v2",
        "cudaGraphInstantiate"
    ],
    "hipGraphInstantiateWithFlags": [
        "cuGraphInstantiateWithFlags",
        "cudaGraphInstantiateWithFlags"
    ],
    "hipGraphKernelNodeGetAttribute": [
        "cuGraphKernelNodeGetAttribute",
        "cudaGraphKernelNodeGetAttribute"
    ],
    "hipGraphKernelNodeGetParams": [
        "cuGraphKernelNodeGetParams",
        "cudaGraphKernelNodeGetParams"
    ],
    "hipGraphKernelNodeSetAttribute": [
        "cuGraphKernelNodeSetAttribute",
        "cudaGraphKernelNodeSetAttribute"
    ],
    "hipGraphKernelNodeSetParams": [
        "cuGraphKernelNodeSetParams",
        "cudaGraphKernelNodeSetParams"
    ],
    "hipGraphLaunch": [
        "cuGraphLaunch",
        "cudaGraphLaunch"
    ],
    "hipGraphMemcpyNodeGetParams": [
        "cuGraphMemcpyNodeGetParams",
        "cudaGraphMemcpyNodeGetParams"
    ],
    "hipGraphMemcpyNodeSetParams": [
        "cuGraphMemcpyNodeSetParams",
        "cudaGraphMemcpyNodeSetParams"
    ],
    "hipGraphMemsetNodeGetParams": [
        "cuGraphMemsetNodeGetParams",
        "cudaGraphMemsetNodeGetParams"
    ],
    "hipGraphMemsetNodeSetParams": [
        "cuGraphMemsetNodeSetParams",
        "cudaGraphMemsetNodeSetParams"
    ],
    "hipGraphNodeFindInClone": [
        "cuGraphNodeFindInClone",
        "cudaGraphNodeFindInClone"
    ],
    "hipGraphNodeGetDependencies": [
        "cuGraphNodeGetDependencies",
        "cudaGraphNodeGetDependencies"
    ],
    "hipGraphNodeGetDependentNodes": [
        "cuGraphNodeGetDependentNodes",
        "cudaGraphNodeGetDependentNodes"
    ],
    "hipGraphNodeGetType": [
        "cuGraphNodeGetType",
        "cudaGraphNodeGetType"
    ],
    "hipGraphReleaseUserObject": [
        "cuGraphReleaseUserObject",
        "cudaGraphReleaseUserObject"
    ],
    "hipGraphRemoveDependencies": [
        "cuGraphRemoveDependencies",
        "cudaGraphRemoveDependencies"
    ],
    "hipGraphRetainUserObject": [
        "cuGraphRetainUserObject",
        "cudaGraphRetainUserObject"
    ],
    "hipGraphUpload": [
        "cuGraphUpload",
        "cudaGraphUpload"
    ],
    "hipUserObjectCreate": [
        "cuUserObjectCreate",
        "cudaUserObjectCreate"
    ],
    "hipUserObjectRelease": [
        "cuUserObjectRelease",
        "cudaUserObjectRelease"
    ],
    "hipUserObjectRetain": [
        "cuUserObjectRetain",
        "cudaUserObjectRetain"
    ],
    "hipGraphAddMemcpyNode": [
        "cudaGraphAddMemcpyNode"
    ],
    "hipGraphAddMemcpyNode1D": [
        "cudaGraphAddMemcpyNode1D"
    ],
    "hipGraphAddMemcpyNodeFromSymbol": [
        "cudaGraphAddMemcpyNodeFromSymbol"
    ],
    "hipGraphAddMemcpyNodeToSymbol": [
        "cudaGraphAddMemcpyNodeToSymbol"
    ],
    "hipGraphAddMemsetNode": [
        "cudaGraphAddMemsetNode"
    ],
    "hipGraphExecMemcpyNodeSetParams": [
        "cudaGraphExecMemcpyNodeSetParams"
    ],
    "hipGraphExecMemcpyNodeSetParams1D": [
        "cudaGraphExecMemcpyNodeSetParams1D"
    ],
    "hipGraphExecMemcpyNodeSetParamsFromSymbol": [
        "cudaGraphExecMemcpyNodeSetParamsFromSymbol"
    ],
    "hipGraphExecMemcpyNodeSetParamsToSymbol": [
        "cudaGraphExecMemcpyNodeSetParamsToSymbol"
    ],
    "hipGraphExecMemsetNodeSetParams": [
        "cudaGraphExecMemsetNodeSetParams"
    ],
    "hipGraphMemcpyNodeSetParams1D": [
        "cudaGraphMemcpyNodeSetParams1D"
    ],
    "hipGraphMemcpyNodeSetParamsFromSymbol": [
        "cudaGraphMemcpyNodeSetParamsFromSymbol"
    ],
    "hipGraphMemcpyNodeSetParamsToSymbol": [
        "cudaGraphMemcpyNodeSetParamsToSymbol"
    ],
    "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor": [
        "cuOccupancyMaxActiveBlocksPerMultiprocessor"
    ],
    "hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags": [
        "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"
    ],
    "hipModuleOccupancyMaxPotentialBlockSize": [
        "cuOccupancyMaxPotentialBlockSize"
    ],
    "hipModuleOccupancyMaxPotentialBlockSizeWithFlags": [
        "cuOccupancyMaxPotentialBlockSizeWithFlags"
    ],
    "hipOccupancyMaxActiveBlocksPerMultiprocessor": [
        "cudaOccupancyMaxActiveBlocksPerMultiprocessor"
    ],
    "hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags": [
        "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"
    ],
    "hipOccupancyMaxPotentialBlockSize": [
        "cudaOccupancyMaxPotentialBlockSize"
    ],
    "hipOccupancyMaxPotentialBlockSizeWithFlags": [
        "cudaOccupancyMaxPotentialBlockSizeWithFlags"
    ],
    "hipTexObjectCreate": [
        "cuTexObjectCreate"
    ],
    "hipTexObjectDestroy": [
        "cuTexObjectDestroy"
    ],
    "hipTexObjectGetResourceDesc": [
        "cuTexObjectGetResourceDesc"
    ],
    "hipTexObjectGetResourceViewDesc": [
        "cuTexObjectGetResourceViewDesc"
    ],
    "hipTexObjectGetTextureDesc": [
        "cuTexObjectGetTextureDesc"
    ],
    "hipTexRefGetAddress": [
        "cuTexRefGetAddress",
        "cuTexRefGetAddress_v2"
    ],
    "hipTexRefGetAddressMode": [
        "cuTexRefGetAddressMode"
    ],
    "hipTexRefGetArray": [
        "cuTexRefGetArray"
    ],
    "hipTexRefGetFilterMode": [
        "cuTexRefGetFilterMode"
    ],
    "hipTexRefGetFlags": [
        "cuTexRefGetFlags"
    ],
    "hipTexRefGetFormat": [
        "cuTexRefGetFormat"
    ],
    "hipTexRefGetMaxAnisotropy": [
        "cuTexRefGetMaxAnisotropy"
    ],
    "hipTexRefGetMipmapFilterMode": [
        "cuTexRefGetMipmapFilterMode"
    ],
    "hipTexRefGetMipmapLevelBias": [
        "cuTexRefGetMipmapLevelBias"
    ],
    "hipTexRefGetMipmapLevelClamp": [
        "cuTexRefGetMipmapLevelClamp"
    ],
    "hipTexRefGetMipMappedArray": [
        "cuTexRefGetMipmappedArray"
    ],
    "hipTexRefSetAddress": [
        "cuTexRefSetAddress",
        "cuTexRefSetAddress_v2"
    ],
    "hipTexRefSetAddress2D": [
        "cuTexRefSetAddress2D",
        "cuTexRefSetAddress2D_v2",
        "cuTexRefSetAddress2D_v3"
    ],
    "hipTexRefSetAddressMode": [
        "cuTexRefSetAddressMode"
    ],
    "hipTexRefSetArray": [
        "cuTexRefSetArray"
    ],
    "hipTexRefSetBorderColor": [
        "cuTexRefSetBorderColor"
    ],
    "hipTexRefSetFilterMode": [
        "cuTexRefSetFilterMode"
    ],
    "hipTexRefSetFlags": [
        "cuTexRefSetFlags"
    ],
    "hipTexRefSetFormat": [
        "cuTexRefSetFormat"
    ],
    "hipTexRefSetMaxAnisotropy": [
        "cuTexRefSetMaxAnisotropy"
    ],
    "hipTexRefSetMipmapFilterMode": [
        "cuTexRefSetMipmapFilterMode"
    ],
    "hipTexRefSetMipmapLevelBias": [
        "cuTexRefSetMipmapLevelBias"
    ],
    "hipTexRefSetMipmapLevelClamp": [
        "cuTexRefSetMipmapLevelClamp"
    ],
    "hipTexRefSetMipmappedArray": [
        "cuTexRefSetMipmappedArray"
    ],
    "hipBindTexture": [
        "cudaBindTexture"
    ],
    "hipBindTexture2D": [
        "cudaBindTexture2D"
    ],
    "hipBindTextureToArray": [
        "cudaBindTextureToArray"
    ],
    "hipBindTextureToMipmappedArray": [
        "cudaBindTextureToMipmappedArray"
    ],
    "hipCreateChannelDesc": [
        "cudaCreateChannelDesc"
    ],
    "hipCreateTextureObject": [
        "cudaCreateTextureObject"
    ],
    "hipDestroyTextureObject": [
        "cudaDestroyTextureObject"
    ],
    "hipGetChannelDesc": [
        "cudaGetChannelDesc"
    ],
    "hipGetTextureAlignmentOffset": [
        "cudaGetTextureAlignmentOffset"
    ],
    "hipGetTextureObjectResourceDesc": [
        "cudaGetTextureObjectResourceDesc"
    ],
    "hipGetTextureObjectResourceViewDesc": [
        "cudaGetTextureObjectResourceViewDesc"
    ],
    "hipGetTextureObjectTextureDesc": [
        "cudaGetTextureObjectTextureDesc"
    ],
    "hipGetTextureReference": [
        "cudaGetTextureReference"
    ],
    "hipUnbindTexture": [
        "cudaUnbindTexture"
    ],
    "hipCreateSurfaceObject": [
        "cudaCreateSurfaceObject"
    ],
    "hipDestroySurfaceObject": [
        "cudaDestroySurfaceObject"
    ],
    "hipCtxDisablePeerAccess": [
        "cuCtxDisablePeerAccess"
    ],
    "hipCtxEnablePeerAccess": [
        "cuCtxEnablePeerAccess"
    ],
    "hipDeviceCanAccessPeer": [
        "cuDeviceCanAccessPeer",
        "cudaDeviceCanAccessPeer"
    ],
    "hipDeviceDisablePeerAccess": [
        "cudaDeviceDisablePeerAccess"
    ],
    "hipDeviceEnablePeerAccess": [
        "cudaDeviceEnablePeerAccess"
    ],
    "hipGraphicsMapResources": [
        "cuGraphicsMapResources",
        "cudaGraphicsMapResources"
    ],
    "hipGraphicsResourceGetMappedPointer": [
        "cuGraphicsResourceGetMappedPointer",
        "cuGraphicsResourceGetMappedPointer_v2",
        "cudaGraphicsResourceGetMappedPointer"
    ],
    "hipGraphicsSubResourceGetMappedArray": [
        "cuGraphicsSubResourceGetMappedArray",
        "cudaGraphicsSubResourceGetMappedArray"
    ],
    "hipGraphicsUnmapResources": [
        "cuGraphicsUnmapResources",
        "cudaGraphicsUnmapResources"
    ],
    "hipGraphicsUnregisterResource": [
        "cuGraphicsUnregisterResource",
        "cudaGraphicsUnregisterResource"
    ],
    "hipProfilerStart": [
        "cuProfilerStart",
        "cudaProfilerStart"
    ],
    "hipProfilerStop": [
        "cuProfilerStop",
        "cudaProfilerStop"
    ],
    "hipGLGetDevices": [
        "cuGLGetDevices",
        "cudaGLGetDevices"
    ],
    "hipGraphicsGLRegisterBuffer": [
        "cuGraphicsGLRegisterBuffer",
        "cudaGraphicsGLRegisterBuffer"
    ],
    "hipGraphicsGLRegisterImage": [
        "cuGraphicsGLRegisterImage",
        "cudaGraphicsGLRegisterImage"
    ],
    "hipCabs": [
        "cuCabs"
    ],
    "hipCabsf": [
        "cuCabsf"
    ],
    "hipCadd": [
        "cuCadd"
    ],
    "hipCaddf": [
        "cuCaddf"
    ],
    "hipCdiv": [
        "cuCdiv"
    ],
    "hipCdivf": [
        "cuCdivf"
    ],
    "hipCfma": [
        "cuCfma"
    ],
    "hipCfmaf": [
        "cuCfmaf"
    ],
    "hipCimag": [
        "cuCimag"
    ],
    "hipCimagf": [
        "cuCimagf"
    ],
    "hipCmul": [
        "cuCmul"
    ],
    "hipCmulf": [
        "cuCmulf"
    ],
    "hipComplexDoubleToFloat": [
        "cuComplexDoubleToFloat"
    ],
    "hipComplexFloatToDouble": [
        "cuComplexFloatToDouble"
    ],
    "hipConj": [
        "cuConj"
    ],
    "hipConjf": [
        "cuConjf"
    ],
    "hipCreal": [
        "cuCreal"
    ],
    "hipCrealf": [
        "cuCrealf"
    ],
    "hipCsub": [
        "cuCsub"
    ],
    "hipCsubf": [
        "cuCsubf"
    ],
    "make_hipComplex": [
        "make_cuComplex"
    ],
    "make_hipDoubleComplex": [
        "make_cuDoubleComplex"
    ],
    "make_hipFloatComplex": [
        "make_cuFloatComplex"
    ],
    "hipblasAxpyEx": [
        "cublasAxpyEx"
    ],
    "hipblasCaxpy": [
        "cublasCaxpy",
        "cublasCaxpy_v2"
    ],
    "hipblasCcopy": [
        "cublasCcopy",
        "cublasCcopy_v2"
    ],
    "hipblasCdgmm": [
        "cublasCdgmm"
    ],
    "hipblasCdotc": [
        "cublasCdotc",
        "cublasCdotc_v2"
    ],
    "hipblasCdotu": [
        "cublasCdotu",
        "cublasCdotu_v2"
    ],
    "hipblasCgbmv": [
        "cublasCgbmv",
        "cublasCgbmv_v2"
    ],
    "hipblasCgeam": [
        "cublasCgeam"
    ],
    "hipblasCgemm": [
        "cublasCgemm",
        "cublasCgemm_v2"
    ],
    "hipblasCgemmBatched": [
        "cublasCgemmBatched"
    ],
    "hipblasCgemmStridedBatched": [
        "cublasCgemmStridedBatched"
    ],
    "hipblasCgemv": [
        "cublasCgemv",
        "cublasCgemv_v2"
    ],
    "hipblasCgeqrfBatched": [
        "cublasCgeqrfBatched"
    ],
    "hipblasCgerc": [
        "cublasCgerc",
        "cublasCgerc_v2"
    ],
    "hipblasCgeru": [
        "cublasCgeru",
        "cublasCgeru_v2"
    ],
    "hipblasCgetrfBatched": [
        "cublasCgetrfBatched"
    ],
    "hipblasCgetriBatched": [
        "cublasCgetriBatched"
    ],
    "hipblasCgetrsBatched": [
        "cublasCgetrsBatched"
    ],
    "hipblasChbmv": [
        "cublasChbmv",
        "cublasChbmv_v2"
    ],
    "hipblasChemm": [
        "cublasChemm",
        "cublasChemm_v2"
    ],
    "hipblasChemv": [
        "cublasChemv",
        "cublasChemv_v2"
    ],
    "hipblasCher": [
        "cublasCher",
        "cublasCher_v2"
    ],
    "hipblasCher2": [
        "cublasCher2",
        "cublasCher2_v2"
    ],
    "hipblasCher2k": [
        "cublasCher2k",
        "cublasCher2k_v2"
    ],
    "hipblasCherk": [
        "cublasCherk",
        "cublasCherk_v2"
    ],
    "hipblasCherkx": [
        "cublasCherkx"
    ],
    "hipblasChpmv": [
        "cublasChpmv",
        "cublasChpmv_v2"
    ],
    "hipblasChpr": [
        "cublasChpr",
        "cublasChpr_v2"
    ],
    "hipblasChpr2": [
        "cublasChpr2",
        "cublasChpr2_v2"
    ],
    "hipblasCreate": [
        "cublasCreate",
        "cublasCreate_v2"
    ],
    "hipblasCrot": [
        "cublasCrot",
        "cublasCrot_v2"
    ],
    "hipblasCrotg": [
        "cublasCrotg",
        "cublasCrotg_v2"
    ],
    "hipblasCscal": [
        "cublasCscal",
        "cublasCscal_v2"
    ],
    "hipblasCsrot": [
        "cublasCsrot",
        "cublasCsrot_v2"
    ],
    "hipblasCsscal": [
        "cublasCsscal",
        "cublasCsscal_v2"
    ],
    "hipblasCswap": [
        "cublasCswap",
        "cublasCswap_v2"
    ],
    "hipblasCsymm": [
        "cublasCsymm",
        "cublasCsymm_v2"
    ],
    "hipblasCsymv": [
        "cublasCsymv",
        "cublasCsymv_v2"
    ],
    "hipblasCsyr": [
        "cublasCsyr",
        "cublasCsyr_v2"
    ],
    "hipblasCsyr2": [
        "cublasCsyr2",
        "cublasCsyr2_v2"
    ],
    "hipblasCsyr2k": [
        "cublasCsyr2k",
        "cublasCsyr2k_v2"
    ],
    "hipblasCsyrk": [
        "cublasCsyrk",
        "cublasCsyrk_v2"
    ],
    "hipblasCsyrkx": [
        "cublasCsyrkx"
    ],
    "hipblasCtbmv": [
        "cublasCtbmv",
        "cublasCtbmv_v2"
    ],
    "hipblasCtbsv": [
        "cublasCtbsv",
        "cublasCtbsv_v2"
    ],
    "hipblasCtpmv": [
        "cublasCtpmv",
        "cublasCtpmv_v2"
    ],
    "hipblasCtpsv": [
        "cublasCtpsv",
        "cublasCtpsv_v2"
    ],
    "hipblasCtrmv": [
        "cublasCtrmv",
        "cublasCtrmv_v2"
    ],
    "hipblasCtrsm": [
        "cublasCtrsm",
        "cublasCtrsm_v2"
    ],
    "hipblasCtrsmBatched": [
        "cublasCtrsmBatched"
    ],
    "hipblasCtrsv": [
        "cublasCtrsv",
        "cublasCtrsv_v2"
    ],
    "hipblasDasum": [
        "cublasDasum",
        "cublasDasum_v2"
    ],
    "hipblasDaxpy": [
        "cublasDaxpy",
        "cublasDaxpy_v2"
    ],
    "hipblasDcopy": [
        "cublasDcopy",
        "cublasDcopy_v2"
    ],
    "hipblasDdgmm": [
        "cublasDdgmm"
    ],
    "hipblasDdot": [
        "cublasDdot",
        "cublasDdot_v2"
    ],
    "hipblasDestroy": [
        "cublasDestroy",
        "cublasDestroy_v2"
    ],
    "hipblasDgbmv": [
        "cublasDgbmv",
        "cublasDgbmv_v2"
    ],
    "hipblasDgeam": [
        "cublasDgeam"
    ],
    "hipblasDgemm": [
        "cublasDgemm",
        "cublasDgemm_v2"
    ],
    "hipblasDgemmBatched": [
        "cublasDgemmBatched"
    ],
    "hipblasDgemmStridedBatched": [
        "cublasDgemmStridedBatched"
    ],
    "hipblasDgemv": [
        "cublasDgemv",
        "cublasDgemv_v2"
    ],
    "hipblasDgeqrfBatched": [
        "cublasDgeqrfBatched"
    ],
    "hipblasDger": [
        "cublasDger",
        "cublasDger_v2"
    ],
    "hipblasDgetrfBatched": [
        "cublasDgetrfBatched"
    ],
    "hipblasDgetriBatched": [
        "cublasDgetriBatched"
    ],
    "hipblasDgetrsBatched": [
        "cublasDgetrsBatched"
    ],
    "hipblasDnrm2": [
        "cublasDnrm2",
        "cublasDnrm2_v2"
    ],
    "hipblasDotEx": [
        "cublasDotEx"
    ],
    "hipblasDotcEx": [
        "cublasDotcEx"
    ],
    "hipblasDrot": [
        "cublasDrot",
        "cublasDrot_v2"
    ],
    "hipblasDrotg": [
        "cublasDrotg",
        "cublasDrotg_v2"
    ],
    "hipblasDrotm": [
        "cublasDrotm",
        "cublasDrotm_v2"
    ],
    "hipblasDrotmg": [
        "cublasDrotmg",
        "cublasDrotmg_v2"
    ],
    "hipblasDsbmv": [
        "cublasDsbmv",
        "cublasDsbmv_v2"
    ],
    "hipblasDscal": [
        "cublasDscal",
        "cublasDscal_v2"
    ],
    "hipblasDspmv": [
        "cublasDspmv",
        "cublasDspmv_v2"
    ],
    "hipblasDspr": [
        "cublasDspr",
        "cublasDspr_v2"
    ],
    "hipblasDspr2": [
        "cublasDspr2",
        "cublasDspr2_v2"
    ],
    "hipblasDswap": [
        "cublasDswap",
        "cublasDswap_v2"
    ],
    "hipblasDsymm": [
        "cublasDsymm",
        "cublasDsymm_v2"
    ],
    "hipblasDsymv": [
        "cublasDsymv",
        "cublasDsymv_v2"
    ],
    "hipblasDsyr": [
        "cublasDsyr",
        "cublasDsyr_v2"
    ],
    "hipblasDsyr2": [
        "cublasDsyr2",
        "cublasDsyr2_v2"
    ],
    "hipblasDsyr2k": [
        "cublasDsyr2k",
        "cublasDsyr2k_v2"
    ],
    "hipblasDsyrk": [
        "cublasDsyrk",
        "cublasDsyrk_v2"
    ],
    "hipblasDsyrkx": [
        "cublasDsyrkx"
    ],
    "hipblasDtbmv": [
        "cublasDtbmv",
        "cublasDtbmv_v2"
    ],
    "hipblasDtbsv": [
        "cublasDtbsv",
        "cublasDtbsv_v2"
    ],
    "hipblasDtpmv": [
        "cublasDtpmv",
        "cublasDtpmv_v2"
    ],
    "hipblasDtpsv": [
        "cublasDtpsv",
        "cublasDtpsv_v2"
    ],
    "hipblasDtrmv": [
        "cublasDtrmv",
        "cublasDtrmv_v2"
    ],
    "hipblasDtrsm": [
        "cublasDtrsm",
        "cublasDtrsm_v2"
    ],
    "hipblasDtrsmBatched": [
        "cublasDtrsmBatched"
    ],
    "hipblasDtrsv": [
        "cublasDtrsv",
        "cublasDtrsv_v2"
    ],
    "hipblasDzasum": [
        "cublasDzasum",
        "cublasDzasum_v2"
    ],
    "hipblasDznrm2": [
        "cublasDznrm2",
        "cublasDznrm2_v2"
    ],
    "hipblasGemmBatchedEx": [
        "cublasGemmBatchedEx"
    ],
    "hipblasGemmEx": [
        "cublasGemmEx"
    ],
    "hipblasGemmStridedBatchedEx": [
        "cublasGemmStridedBatchedEx"
    ],
    "hipblasGetAtomicsMode": [
        "cublasGetAtomicsMode"
    ],
    "hipblasGetMatrix": [
        "cublasGetMatrix"
    ],
    "hipblasGetMatrixAsync": [
        "cublasGetMatrixAsync"
    ],
    "hipblasGetPointerMode": [
        "cublasGetPointerMode",
        "cublasGetPointerMode_v2"
    ],
    "hipblasGetStream": [
        "cublasGetStream",
        "cublasGetStream_v2"
    ],
    "hipblasGetVector": [
        "cublasGetVector"
    ],
    "hipblasGetVectorAsync": [
        "cublasGetVectorAsync"
    ],
    "hipblasHgemm": [
        "cublasHgemm"
    ],
    "hipblasHgemmBatched": [
        "cublasHgemmBatched"
    ],
    "hipblasHgemmStridedBatched": [
        "cublasHgemmStridedBatched"
    ],
    "hipblasIcamax": [
        "cublasIcamax",
        "cublasIcamax_v2"
    ],
    "hipblasIcamin": [
        "cublasIcamin",
        "cublasIcamin_v2"
    ],
    "hipblasIdamax": [
        "cublasIdamax",
        "cublasIdamax_v2"
    ],
    "hipblasIdamin": [
        "cublasIdamin",
        "cublasIdamin_v2"
    ],
    "hipblasIsamax": [
        "cublasIsamax",
        "cublasIsamax_v2"
    ],
    "hipblasIsamin": [
        "cublasIsamin",
        "cublasIsamin_v2"
    ],
    "hipblasIzamax": [
        "cublasIzamax",
        "cublasIzamax_v2"
    ],
    "hipblasIzamin": [
        "cublasIzamin",
        "cublasIzamin_v2"
    ],
    "hipblasNrm2Ex": [
        "cublasNrm2Ex"
    ],
    "hipblasRotEx": [
        "cublasRotEx"
    ],
    "hipblasSasum": [
        "cublasSasum",
        "cublasSasum_v2"
    ],
    "hipblasSaxpy": [
        "cublasSaxpy",
        "cublasSaxpy_v2"
    ],
    "hipblasScalEx": [
        "cublasScalEx"
    ],
    "hipblasScasum": [
        "cublasScasum",
        "cublasScasum_v2"
    ],
    "hipblasScnrm2": [
        "cublasScnrm2",
        "cublasScnrm2_v2"
    ],
    "hipblasScopy": [
        "cublasScopy",
        "cublasScopy_v2"
    ],
    "hipblasSdgmm": [
        "cublasSdgmm"
    ],
    "hipblasSdot": [
        "cublasSdot",
        "cublasSdot_v2"
    ],
    "hipblasSetAtomicsMode": [
        "cublasSetAtomicsMode"
    ],
    "hipblasSetMatrix": [
        "cublasSetMatrix"
    ],
    "hipblasSetMatrixAsync": [
        "cublasSetMatrixAsync"
    ],
    "hipblasSetPointerMode": [
        "cublasSetPointerMode",
        "cublasSetPointerMode_v2"
    ],
    "hipblasSetStream": [
        "cublasSetStream",
        "cublasSetStream_v2"
    ],
    "hipblasSetVector": [
        "cublasSetVector"
    ],
    "hipblasSetVectorAsync": [
        "cublasSetVectorAsync"
    ],
    "hipblasSgbmv": [
        "cublasSgbmv",
        "cublasSgbmv_v2"
    ],
    "hipblasSgeam": [
        "cublasSgeam"
    ],
    "hipblasSgemm": [
        "cublasSgemm",
        "cublasSgemm_v2"
    ],
    "hipblasSgemmBatched": [
        "cublasSgemmBatched"
    ],
    "hipblasSgemmStridedBatched": [
        "cublasSgemmStridedBatched"
    ],
    "hipblasSgemv": [
        "cublasSgemv",
        "cublasSgemv_v2"
    ],
    "hipblasSgeqrfBatched": [
        "cublasSgeqrfBatched"
    ],
    "hipblasSger": [
        "cublasSger",
        "cublasSger_v2"
    ],
    "hipblasSgetrfBatched": [
        "cublasSgetrfBatched"
    ],
    "hipblasSgetriBatched": [
        "cublasSgetriBatched"
    ],
    "hipblasSgetrsBatched": [
        "cublasSgetrsBatched"
    ],
    "hipblasSnrm2": [
        "cublasSnrm2",
        "cublasSnrm2_v2"
    ],
    "hipblasSrot": [
        "cublasSrot",
        "cublasSrot_v2"
    ],
    "hipblasSrotg": [
        "cublasSrotg",
        "cublasSrotg_v2"
    ],
    "hipblasSrotm": [
        "cublasSrotm",
        "cublasSrotm_v2"
    ],
    "hipblasSrotmg": [
        "cublasSrotmg",
        "cublasSrotmg_v2"
    ],
    "hipblasSsbmv": [
        "cublasSsbmv",
        "cublasSsbmv_v2"
    ],
    "hipblasSscal": [
        "cublasSscal",
        "cublasSscal_v2"
    ],
    "hipblasSspmv": [
        "cublasSspmv",
        "cublasSspmv_v2"
    ],
    "hipblasSspr": [
        "cublasSspr",
        "cublasSspr_v2"
    ],
    "hipblasSspr2": [
        "cublasSspr2",
        "cublasSspr2_v2"
    ],
    "hipblasSswap": [
        "cublasSswap",
        "cublasSswap_v2"
    ],
    "hipblasSsymm": [
        "cublasSsymm",
        "cublasSsymm_v2"
    ],
    "hipblasSsymv": [
        "cublasSsymv",
        "cublasSsymv_v2"
    ],
    "hipblasSsyr": [
        "cublasSsyr",
        "cublasSsyr_v2"
    ],
    "hipblasSsyr2": [
        "cublasSsyr2",
        "cublasSsyr2_v2"
    ],
    "hipblasSsyr2k": [
        "cublasSsyr2k",
        "cublasSsyr2k_v2"
    ],
    "hipblasSsyrk": [
        "cublasSsyrk",
        "cublasSsyrk_v2"
    ],
    "hipblasSsyrkx": [
        "cublasSsyrkx"
    ],
    "hipblasStbmv": [
        "cublasStbmv",
        "cublasStbmv_v2"
    ],
    "hipblasStbsv": [
        "cublasStbsv",
        "cublasStbsv_v2"
    ],
    "hipblasStpmv": [
        "cublasStpmv",
        "cublasStpmv_v2"
    ],
    "hipblasStpsv": [
        "cublasStpsv",
        "cublasStpsv_v2"
    ],
    "hipblasStrmv": [
        "cublasStrmv",
        "cublasStrmv_v2"
    ],
    "hipblasStrsm": [
        "cublasStrsm",
        "cublasStrsm_v2"
    ],
    "hipblasStrsmBatched": [
        "cublasStrsmBatched"
    ],
    "hipblasStrsv": [
        "cublasStrsv",
        "cublasStrsv_v2"
    ],
    "hipblasZaxpy": [
        "cublasZaxpy",
        "cublasZaxpy_v2"
    ],
    "hipblasZcopy": [
        "cublasZcopy",
        "cublasZcopy_v2"
    ],
    "hipblasZdgmm": [
        "cublasZdgmm"
    ],
    "hipblasZdotc": [
        "cublasZdotc",
        "cublasZdotc_v2"
    ],
    "hipblasZdotu": [
        "cublasZdotu",
        "cublasZdotu_v2"
    ],
    "hipblasZdrot": [
        "cublasZdrot",
        "cublasZdrot_v2"
    ],
    "hipblasZdscal": [
        "cublasZdscal",
        "cublasZdscal_v2"
    ],
    "hipblasZgbmv": [
        "cublasZgbmv",
        "cublasZgbmv_v2"
    ],
    "hipblasZgeam": [
        "cublasZgeam"
    ],
    "hipblasZgemm": [
        "cublasZgemm",
        "cublasZgemm_v2"
    ],
    "hipblasZgemmBatched": [
        "cublasZgemmBatched"
    ],
    "hipblasZgemmStridedBatched": [
        "cublasZgemmStridedBatched"
    ],
    "hipblasZgemv": [
        "cublasZgemv",
        "cublasZgemv_v2"
    ],
    "hipblasZgeqrfBatched": [
        "cublasZgeqrfBatched"
    ],
    "hipblasZgerc": [
        "cublasZgerc",
        "cublasZgerc_v2"
    ],
    "hipblasZgeru": [
        "cublasZgeru",
        "cublasZgeru_v2"
    ],
    "hipblasZgetrfBatched": [
        "cublasZgetrfBatched"
    ],
    "hipblasZgetriBatched": [
        "cublasZgetriBatched"
    ],
    "hipblasZgetrsBatched": [
        "cublasZgetrsBatched"
    ],
    "hipblasZhbmv": [
        "cublasZhbmv",
        "cublasZhbmv_v2"
    ],
    "hipblasZhemm": [
        "cublasZhemm",
        "cublasZhemm_v2"
    ],
    "hipblasZhemv": [
        "cublasZhemv",
        "cublasZhemv_v2"
    ],
    "hipblasZher": [
        "cublasZher",
        "cublasZher_v2"
    ],
    "hipblasZher2": [
        "cublasZher2",
        "cublasZher2_v2"
    ],
    "hipblasZher2k": [
        "cublasZher2k",
        "cublasZher2k_v2"
    ],
    "hipblasZherk": [
        "cublasZherk",
        "cublasZherk_v2"
    ],
    "hipblasZherkx": [
        "cublasZherkx"
    ],
    "hipblasZhpmv": [
        "cublasZhpmv",
        "cublasZhpmv_v2"
    ],
    "hipblasZhpr": [
        "cublasZhpr",
        "cublasZhpr_v2"
    ],
    "hipblasZhpr2": [
        "cublasZhpr2",
        "cublasZhpr2_v2"
    ],
    "hipblasZrot": [
        "cublasZrot",
        "cublasZrot_v2"
    ],
    "hipblasZrotg": [
        "cublasZrotg",
        "cublasZrotg_v2"
    ],
    "hipblasZscal": [
        "cublasZscal",
        "cublasZscal_v2"
    ],
    "hipblasZswap": [
        "cublasZswap",
        "cublasZswap_v2"
    ],
    "hipblasZsymm": [
        "cublasZsymm",
        "cublasZsymm_v2"
    ],
    "hipblasZsymv": [
        "cublasZsymv",
        "cublasZsymv_v2"
    ],
    "hipblasZsyr": [
        "cublasZsyr",
        "cublasZsyr_v2"
    ],
    "hipblasZsyr2": [
        "cublasZsyr2",
        "cublasZsyr2_v2"
    ],
    "hipblasZsyr2k": [
        "cublasZsyr2k",
        "cublasZsyr2k_v2"
    ],
    "hipblasZsyrk": [
        "cublasZsyrk",
        "cublasZsyrk_v2"
    ],
    "hipblasZsyrkx": [
        "cublasZsyrkx"
    ],
    "hipblasZtbmv": [
        "cublasZtbmv",
        "cublasZtbmv_v2"
    ],
    "hipblasZtbsv": [
        "cublasZtbsv",
        "cublasZtbsv_v2"
    ],
    "hipblasZtpmv": [
        "cublasZtpmv",
        "cublasZtpmv_v2"
    ],
    "hipblasZtpsv": [
        "cublasZtpsv",
        "cublasZtpsv_v2"
    ],
    "hipblasZtrmv": [
        "cublasZtrmv",
        "cublasZtrmv_v2"
    ],
    "hipblasZtrsm": [
        "cublasZtrsm",
        "cublasZtrsm_v2"
    ],
    "hipblasZtrsmBatched": [
        "cublasZtrsmBatched"
    ],
    "hipblasZtrsv": [
        "cublasZtrsv",
        "cublasZtrsv_v2"
    ],
    "hip_stream": [
        "cuda_stream"
    ],
    "hipdnnActivationBackward": [
        "cudnnActivationBackward"
    ],
    "hipdnnActivationForward": [
        "cudnnActivationForward"
    ],
    "hipdnnAddTensor": [
        "cudnnAddTensor"
    ],
    "hipdnnBatchNormalizationBackward": [
        "cudnnBatchNormalizationBackward"
    ],
    "hipdnnBatchNormalizationForwardInference": [
        "cudnnBatchNormalizationForwardInference"
    ],
    "hipdnnBatchNormalizationForwardTraining": [
        "cudnnBatchNormalizationForwardTraining"
    ],
    "hipdnnConvolutionBackwardBias": [
        "cudnnConvolutionBackwardBias"
    ],
    "hipdnnConvolutionBackwardData": [
        "cudnnConvolutionBackwardData"
    ],
    "hipdnnConvolutionBackwardFilter": [
        "cudnnConvolutionBackwardFilter"
    ],
    "hipdnnConvolutionForward": [
        "cudnnConvolutionForward"
    ],
    "hipdnnCreate": [
        "cudnnCreate"
    ],
    "hipdnnCreateActivationDescriptor": [
        "cudnnCreateActivationDescriptor"
    ],
    "hipdnnCreateConvolutionDescriptor": [
        "cudnnCreateConvolutionDescriptor"
    ],
    "hipdnnCreateDropoutDescriptor": [
        "cudnnCreateDropoutDescriptor"
    ],
    "hipdnnCreateFilterDescriptor": [
        "cudnnCreateFilterDescriptor"
    ],
    "hipdnnCreateLRNDescriptor": [
        "cudnnCreateLRNDescriptor"
    ],
    "hipdnnCreateOpTensorDescriptor": [
        "cudnnCreateOpTensorDescriptor"
    ],
    "hipdnnCreatePersistentRNNPlan": [
        "cudnnCreatePersistentRNNPlan"
    ],
    "hipdnnCreatePoolingDescriptor": [
        "cudnnCreatePoolingDescriptor"
    ],
    "hipdnnCreateRNNDescriptor": [
        "cudnnCreateRNNDescriptor"
    ],
    "hipdnnCreateReduceTensorDescriptor": [
        "cudnnCreateReduceTensorDescriptor"
    ],
    "hipdnnCreateTensorDescriptor": [
        "cudnnCreateTensorDescriptor"
    ],
    "hipdnnDeriveBNTensorDescriptor": [
        "cudnnDeriveBNTensorDescriptor"
    ],
    "hipdnnDestroy": [
        "cudnnDestroy"
    ],
    "hipdnnDestroyActivationDescriptor": [
        "cudnnDestroyActivationDescriptor"
    ],
    "hipdnnDestroyConvolutionDescriptor": [
        "cudnnDestroyConvolutionDescriptor"
    ],
    "hipdnnDestroyDropoutDescriptor": [
        "cudnnDestroyDropoutDescriptor"
    ],
    "hipdnnDestroyFilterDescriptor": [
        "cudnnDestroyFilterDescriptor"
    ],
    "hipdnnDestroyLRNDescriptor": [
        "cudnnDestroyLRNDescriptor"
    ],
    "hipdnnDestroyOpTensorDescriptor": [
        "cudnnDestroyOpTensorDescriptor"
    ],
    "hipdnnDestroyPersistentRNNPlan": [
        "cudnnDestroyPersistentRNNPlan"
    ],
    "hipdnnDestroyPoolingDescriptor": [
        "cudnnDestroyPoolingDescriptor"
    ],
    "hipdnnDestroyRNNDescriptor": [
        "cudnnDestroyRNNDescriptor"
    ],
    "hipdnnDestroyReduceTensorDescriptor": [
        "cudnnDestroyReduceTensorDescriptor"
    ],
    "hipdnnDestroyTensorDescriptor": [
        "cudnnDestroyTensorDescriptor"
    ],
    "hipdnnDropoutGetStatesSize": [
        "cudnnDropoutGetStatesSize"
    ],
    "hipdnnFindConvolutionBackwardDataAlgorithm": [
        "cudnnFindConvolutionBackwardDataAlgorithm"
    ],
    "hipdnnFindConvolutionBackwardDataAlgorithmEx": [
        "cudnnFindConvolutionBackwardDataAlgorithmEx"
    ],
    "hipdnnFindConvolutionBackwardFilterAlgorithm": [
        "cudnnFindConvolutionBackwardFilterAlgorithm"
    ],
    "hipdnnFindConvolutionBackwardFilterAlgorithmEx": [
        "cudnnFindConvolutionBackwardFilterAlgorithmEx"
    ],
    "hipdnnFindConvolutionForwardAlgorithm": [
        "cudnnFindConvolutionForwardAlgorithm"
    ],
    "hipdnnFindConvolutionForwardAlgorithmEx": [
        "cudnnFindConvolutionForwardAlgorithmEx"
    ],
    "hipdnnGetActivationDescriptor": [
        "cudnnGetActivationDescriptor"
    ],
    "hipdnnGetConvolution2dDescriptor": [
        "cudnnGetConvolution2dDescriptor"
    ],
    "hipdnnGetConvolution2dForwardOutputDim": [
        "cudnnGetConvolution2dForwardOutputDim"
    ],
    "hipdnnGetConvolutionBackwardDataAlgorithm": [
        "cudnnGetConvolutionBackwardDataAlgorithm"
    ],
    "hipdnnGetConvolutionBackwardDataWorkspaceSize": [
        "cudnnGetConvolutionBackwardDataWorkspaceSize"
    ],
    "hipdnnGetConvolutionBackwardFilterAlgorithm": [
        "cudnnGetConvolutionBackwardFilterAlgorithm"
    ],
    "hipdnnGetConvolutionBackwardFilterWorkspaceSize": [
        "cudnnGetConvolutionBackwardFilterWorkspaceSize"
    ],
    "hipdnnGetConvolutionForwardAlgorithm": [
        "cudnnGetConvolutionForwardAlgorithm"
    ],
    "hipdnnGetConvolutionForwardWorkspaceSize": [
        "cudnnGetConvolutionForwardWorkspaceSize"
    ],
    "hipdnnGetErrorString": [
        "cudnnGetErrorString"
    ],
    "hipdnnGetFilter4dDescriptor": [
        "cudnnGetFilter4dDescriptor"
    ],
    "hipdnnGetFilterNdDescriptor": [
        "cudnnGetFilterNdDescriptor"
    ],
    "hipdnnGetLRNDescriptor": [
        "cudnnGetLRNDescriptor"
    ],
    "hipdnnGetOpTensorDescriptor": [
        "cudnnGetOpTensorDescriptor"
    ],
    "hipdnnGetPooling2dDescriptor": [
        "cudnnGetPooling2dDescriptor"
    ],
    "hipdnnGetPooling2dForwardOutputDim": [
        "cudnnGetPooling2dForwardOutputDim"
    ],
    "hipdnnGetRNNDescriptor": [
        "cudnnGetRNNDescriptor"
    ],
    "hipdnnGetRNNLinLayerBiasParams": [
        "cudnnGetRNNLinLayerBiasParams"
    ],
    "hipdnnGetRNNLinLayerMatrixParams": [
        "cudnnGetRNNLinLayerMatrixParams"
    ],
    "hipdnnGetRNNParamsSize": [
        "cudnnGetRNNParamsSize"
    ],
    "hipdnnGetRNNTrainingReserveSize": [
        "cudnnGetRNNTrainingReserveSize"
    ],
    "hipdnnGetRNNWorkspaceSize": [
        "cudnnGetRNNWorkspaceSize"
    ],
    "hipdnnGetReduceTensorDescriptor": [
        "cudnnGetReduceTensorDescriptor"
    ],
    "hipdnnGetReductionWorkspaceSize": [
        "cudnnGetReductionWorkspaceSize"
    ],
    "hipdnnGetStream": [
        "cudnnGetStream"
    ],
    "hipdnnGetTensor4dDescriptor": [
        "cudnnGetTensor4dDescriptor"
    ],
    "hipdnnGetTensorNdDescriptor": [
        "cudnnGetTensorNdDescriptor"
    ],
    "hipdnnGetVersion": [
        "cudnnGetVersion"
    ],
    "hipdnnLRNCrossChannelBackward": [
        "cudnnLRNCrossChannelBackward"
    ],
    "hipdnnLRNCrossChannelForward": [
        "cudnnLRNCrossChannelForward"
    ],
    "hipdnnOpTensor": [
        "cudnnOpTensor"
    ],
    "hipdnnPoolingBackward": [
        "cudnnPoolingBackward"
    ],
    "hipdnnPoolingForward": [
        "cudnnPoolingForward"
    ],
    "hipdnnRNNBackwardData": [
        "cudnnRNNBackwardData"
    ],
    "hipdnnRNNBackwardWeights": [
        "cudnnRNNBackwardWeights"
    ],
    "hipdnnRNNForwardInference": [
        "cudnnRNNForwardInference"
    ],
    "hipdnnRNNForwardTraining": [
        "cudnnRNNForwardTraining"
    ],
    "hipdnnReduceTensor": [
        "cudnnReduceTensor"
    ],
    "hipdnnScaleTensor": [
        "cudnnScaleTensor"
    ],
    "hipdnnSetActivationDescriptor": [
        "cudnnSetActivationDescriptor"
    ],
    "hipdnnSetConvolution2dDescriptor": [
        "cudnnSetConvolution2dDescriptor"
    ],
    "hipdnnSetConvolutionGroupCount": [
        "cudnnSetConvolutionGroupCount"
    ],
    "hipdnnSetConvolutionMathType": [
        "cudnnSetConvolutionMathType"
    ],
    "hipdnnSetConvolutionNdDescriptor": [
        "cudnnSetConvolutionNdDescriptor"
    ],
    "hipdnnSetDropoutDescriptor": [
        "cudnnSetDropoutDescriptor"
    ],
    "hipdnnSetFilter4dDescriptor": [
        "cudnnSetFilter4dDescriptor"
    ],
    "hipdnnSetFilterNdDescriptor": [
        "cudnnSetFilterNdDescriptor"
    ],
    "hipdnnSetLRNDescriptor": [
        "cudnnSetLRNDescriptor"
    ],
    "hipdnnSetOpTensorDescriptor": [
        "cudnnSetOpTensorDescriptor"
    ],
    "hipdnnSetPersistentRNNPlan": [
        "cudnnSetPersistentRNNPlan"
    ],
    "hipdnnSetPooling2dDescriptor": [
        "cudnnSetPooling2dDescriptor"
    ],
    "hipdnnSetPoolingNdDescriptor": [
        "cudnnSetPoolingNdDescriptor"
    ],
    "hipdnnSetRNNDescriptor": [
        "cudnnSetRNNDescriptor"
    ],
    "hipdnnSetRNNDescriptor_v5": [
        "cudnnSetRNNDescriptor_v5"
    ],
    "hipdnnSetRNNDescriptor_v6": [
        "cudnnSetRNNDescriptor_v6"
    ],
    "hipdnnSetReduceTensorDescriptor": [
        "cudnnSetReduceTensorDescriptor"
    ],
    "hipdnnSetStream": [
        "cudnnSetStream"
    ],
    "hipdnnSetTensor": [
        "cudnnSetTensor"
    ],
    "hipdnnSetTensor4dDescriptor": [
        "cudnnSetTensor4dDescriptor"
    ],
    "hipdnnSetTensor4dDescriptorEx": [
        "cudnnSetTensor4dDescriptorEx"
    ],
    "hipdnnSetTensorNdDescriptor": [
        "cudnnSetTensorNdDescriptor"
    ],
    "hipdnnSoftmaxBackward": [
        "cudnnSoftmaxBackward"
    ],
    "hipdnnSoftmaxForward": [
        "cudnnSoftmaxForward"
    ],
    "hipfftCallbackLoadC": [
        "cufftCallbackLoadC"
    ],
    "hipfftCallbackLoadD": [
        "cufftCallbackLoadD"
    ],
    "hipfftCallbackLoadR": [
        "cufftCallbackLoadR"
    ],
    "hipfftCallbackLoadZ": [
        "cufftCallbackLoadZ"
    ],
    "hipfftCallbackStoreC": [
        "cufftCallbackStoreC"
    ],
    "hipfftCallbackStoreD": [
        "cufftCallbackStoreD"
    ],
    "hipfftCallbackStoreR": [
        "cufftCallbackStoreR"
    ],
    "hipfftCallbackStoreZ": [
        "cufftCallbackStoreZ"
    ],
    "hipfftCreate": [
        "cufftCreate"
    ],
    "hipfftDestroy": [
        "cufftDestroy"
    ],
    "hipfftEstimate1d": [
        "cufftEstimate1d"
    ],
    "hipfftEstimate2d": [
        "cufftEstimate2d"
    ],
    "hipfftEstimate3d": [
        "cufftEstimate3d"
    ],
    "hipfftEstimateMany": [
        "cufftEstimateMany"
    ],
    "hipfftExecC2C": [
        "cufftExecC2C"
    ],
    "hipfftExecC2R": [
        "cufftExecC2R"
    ],
    "hipfftExecD2Z": [
        "cufftExecD2Z"
    ],
    "hipfftExecR2C": [
        "cufftExecR2C"
    ],
    "hipfftExecZ2D": [
        "cufftExecZ2D"
    ],
    "hipfftExecZ2Z": [
        "cufftExecZ2Z"
    ],
    "hipfftGetProperty": [
        "cufftGetProperty"
    ],
    "hipfftGetSize": [
        "cufftGetSize"
    ],
    "hipfftGetSize1d": [
        "cufftGetSize1d"
    ],
    "hipfftGetSize2d": [
        "cufftGetSize2d"
    ],
    "hipfftGetSize3d": [
        "cufftGetSize3d"
    ],
    "hipfftGetSizeMany": [
        "cufftGetSizeMany"
    ],
    "hipfftGetSizeMany64": [
        "cufftGetSizeMany64"
    ],
    "hipfftGetVersion": [
        "cufftGetVersion"
    ],
    "hipfftMakePlan1d": [
        "cufftMakePlan1d"
    ],
    "hipfftMakePlan2d": [
        "cufftMakePlan2d"
    ],
    "hipfftMakePlan3d": [
        "cufftMakePlan3d"
    ],
    "hipfftMakePlanMany": [
        "cufftMakePlanMany"
    ],
    "hipfftMakePlanMany64": [
        "cufftMakePlanMany64"
    ],
    "hipfftPlan1d": [
        "cufftPlan1d"
    ],
    "hipfftPlan2d": [
        "cufftPlan2d"
    ],
    "hipfftPlan3d": [
        "cufftPlan3d"
    ],
    "hipfftPlanMany": [
        "cufftPlanMany"
    ],
    "hipfftSetAutoAllocation": [
        "cufftSetAutoAllocation"
    ],
    "hipfftSetStream": [
        "cufftSetStream"
    ],
    "hipfftSetWorkArea": [
        "cufftSetWorkArea"
    ],
    "hipfftXtClearCallback": [
        "cufftXtClearCallback"
    ],
    "hipfftXtSetCallback": [
        "cufftXtSetCallback"
    ],
    "hipfftXtSetCallbackSharedSize": [
        "cufftXtSetCallbackSharedSize"
    ],
    "hiprandCreateGenerator": [
        "curandCreateGenerator"
    ],
    "hiprandCreateGeneratorHost": [
        "curandCreateGeneratorHost"
    ],
    "hiprandCreatePoissonDistribution": [
        "curandCreatePoissonDistribution"
    ],
    "hiprandDestroyDistribution": [
        "curandDestroyDistribution"
    ],
    "hiprandDestroyGenerator": [
        "curandDestroyGenerator"
    ],
    "hiprandGenerate": [
        "curandGenerate"
    ],
    "hiprandGenerateLogNormal": [
        "curandGenerateLogNormal"
    ],
    "hiprandGenerateLogNormalDouble": [
        "curandGenerateLogNormalDouble"
    ],
    "hiprandGenerateNormal": [
        "curandGenerateNormal"
    ],
    "hiprandGenerateNormalDouble": [
        "curandGenerateNormalDouble"
    ],
    "hiprandGeneratePoisson": [
        "curandGeneratePoisson"
    ],
    "hiprandGenerateSeeds": [
        "curandGenerateSeeds"
    ],
    "hiprandGenerateUniform": [
        "curandGenerateUniform"
    ],
    "hiprandGenerateUniformDouble": [
        "curandGenerateUniformDouble"
    ],
    "hiprandGetVersion": [
        "curandGetVersion"
    ],
    "hiprandMakeMTGP32Constants": [
        "curandMakeMTGP32Constants"
    ],
    "hiprandMakeMTGP32KernelState": [
        "curandMakeMTGP32KernelState"
    ],
    "hiprandSetGeneratorOffset": [
        "curandSetGeneratorOffset"
    ],
    "hiprandSetPseudoRandomGeneratorSeed": [
        "curandSetPseudoRandomGeneratorSeed"
    ],
    "hiprandSetQuasiRandomGeneratorDimensions": [
        "curandSetQuasiRandomGeneratorDimensions"
    ],
    "hiprandSetStream": [
        "curandSetStream"
    ],
    "hipsparseAxpby": [
        "cusparseAxpby"
    ],
    "hipsparseBlockedEllGet": [
        "cusparseBlockedEllGet"
    ],
    "hipsparseCaxpyi": [
        "cusparseCaxpyi"
    ],
    "hipsparseCbsr2csr": [
        "cusparseCbsr2csr"
    ],
    "hipsparseCbsric02": [
        "cusparseCbsric02"
    ],
    "hipsparseCbsric02_analysis": [
        "cusparseCbsric02_analysis"
    ],
    "hipsparseCbsric02_bufferSize": [
        "cusparseCbsric02_bufferSize"
    ],
    "hipsparseCbsrilu02": [
        "cusparseCbsrilu02"
    ],
    "hipsparseCbsrilu02_analysis": [
        "cusparseCbsrilu02_analysis"
    ],
    "hipsparseCbsrilu02_bufferSize": [
        "cusparseCbsrilu02_bufferSize"
    ],
    "hipsparseCbsrilu02_numericBoost": [
        "cusparseCbsrilu02_numericBoost"
    ],
    "hipsparseCbsrmm": [
        "cusparseCbsrmm"
    ],
    "hipsparseCbsrmv": [
        "cusparseCbsrmv"
    ],
    "hipsparseCbsrsm2_analysis": [
        "cusparseCbsrsm2_analysis"
    ],
    "hipsparseCbsrsm2_bufferSize": [
        "cusparseCbsrsm2_bufferSize"
    ],
    "hipsparseCbsrsm2_solve": [
        "cusparseCbsrsm2_solve"
    ],
    "hipsparseCbsrsv2_analysis": [
        "cusparseCbsrsv2_analysis"
    ],
    "hipsparseCbsrsv2_bufferSize": [
        "cusparseCbsrsv2_bufferSize"
    ],
    "hipsparseCbsrsv2_bufferSizeExt": [
        "cusparseCbsrsv2_bufferSizeExt"
    ],
    "hipsparseCbsrsv2_solve": [
        "cusparseCbsrsv2_solve"
    ],
    "hipsparseCbsrxmv": [
        "cusparseCbsrxmv"
    ],
    "hipsparseCcsc2dense": [
        "cusparseCcsc2dense"
    ],
    "hipsparseCcsr2bsr": [
        "cusparseCcsr2bsr"
    ],
    "hipsparseCcsr2csc": [
        "cusparseCcsr2csc"
    ],
    "hipsparseCcsr2csr_compress": [
        "cusparseCcsr2csr_compress"
    ],
    "hipsparseCcsr2csru": [
        "cusparseCcsr2csru"
    ],
    "hipsparseCcsr2dense": [
        "cusparseCcsr2dense"
    ],
    "hipsparseCcsr2gebsr": [
        "cusparseCcsr2gebsr"
    ],
    "hipsparseCcsr2gebsr_bufferSize": [
        "cusparseCcsr2gebsr_bufferSize"
    ],
    "hipsparseCcsr2hyb": [
        "cusparseCcsr2hyb"
    ],
    "hipsparseCcsrcolor": [
        "cusparseCcsrcolor"
    ],
    "hipsparseCcsrgeam": [
        "cusparseCcsrgeam"
    ],
    "hipsparseCcsrgeam2": [
        "cusparseCcsrgeam2"
    ],
    "hipsparseCcsrgeam2_bufferSizeExt": [
        "cusparseCcsrgeam2_bufferSizeExt"
    ],
    "hipsparseCcsrgemm": [
        "cusparseCcsrgemm"
    ],
    "hipsparseCcsrgemm2": [
        "cusparseCcsrgemm2"
    ],
    "hipsparseCcsrgemm2_bufferSizeExt": [
        "cusparseCcsrgemm2_bufferSizeExt"
    ],
    "hipsparseCcsric02": [
        "cusparseCcsric02"
    ],
    "hipsparseCcsric02_analysis": [
        "cusparseCcsric02_analysis"
    ],
    "hipsparseCcsric02_bufferSize": [
        "cusparseCcsric02_bufferSize"
    ],
    "hipsparseCcsric02_bufferSizeExt": [
        "cusparseCcsric02_bufferSizeExt"
    ],
    "hipsparseCcsrilu02": [
        "cusparseCcsrilu02"
    ],
    "hipsparseCcsrilu02_analysis": [
        "cusparseCcsrilu02_analysis"
    ],
    "hipsparseCcsrilu02_bufferSize": [
        "cusparseCcsrilu02_bufferSize"
    ],
    "hipsparseCcsrilu02_bufferSizeExt": [
        "cusparseCcsrilu02_bufferSizeExt"
    ],
    "hipsparseCcsrilu02_numericBoost": [
        "cusparseCcsrilu02_numericBoost"
    ],
    "hipsparseCcsrmm": [
        "cusparseCcsrmm"
    ],
    "hipsparseCcsrmm2": [
        "cusparseCcsrmm2"
    ],
    "hipsparseCcsrmv": [
        "cusparseCcsrmv"
    ],
    "hipsparseCcsrsm2_analysis": [
        "cusparseCcsrsm2_analysis"
    ],
    "hipsparseCcsrsm2_bufferSizeExt": [
        "cusparseCcsrsm2_bufferSizeExt"
    ],
    "hipsparseCcsrsm2_solve": [
        "cusparseCcsrsm2_solve"
    ],
    "hipsparseCcsrsv2_analysis": [
        "cusparseCcsrsv2_analysis"
    ],
    "hipsparseCcsrsv2_bufferSize": [
        "cusparseCcsrsv2_bufferSize"
    ],
    "hipsparseCcsrsv2_bufferSizeExt": [
        "cusparseCcsrsv2_bufferSizeExt"
    ],
    "hipsparseCcsrsv2_solve": [
        "cusparseCcsrsv2_solve"
    ],
    "hipsparseCcsru2csr": [
        "cusparseCcsru2csr"
    ],
    "hipsparseCcsru2csr_bufferSizeExt": [
        "cusparseCcsru2csr_bufferSizeExt"
    ],
    "hipsparseCdense2csc": [
        "cusparseCdense2csc"
    ],
    "hipsparseCdense2csr": [
        "cusparseCdense2csr"
    ],
    "hipsparseCdotci": [
        "cusparseCdotci"
    ],
    "hipsparseCdoti": [
        "cusparseCdoti"
    ],
    "hipsparseCgebsr2csr": [
        "cusparseCgebsr2csr"
    ],
    "hipsparseCgebsr2gebsc": [
        "cusparseCgebsr2gebsc"
    ],
    "hipsparseCgebsr2gebsc_bufferSize": [
        "cusparseCgebsr2gebsc_bufferSize"
    ],
    "hipsparseCgebsr2gebsr": [
        "cusparseCgebsr2gebsr"
    ],
    "hipsparseCgebsr2gebsr_bufferSize": [
        "cusparseCgebsr2gebsr_bufferSize"
    ],
    "hipsparseCgemmi": [
        "cusparseCgemmi"
    ],
    "hipsparseCgemvi": [
        "cusparseCgemvi"
    ],
    "hipsparseCgemvi_bufferSize": [
        "cusparseCgemvi_bufferSize"
    ],
    "hipsparseCgpsvInterleavedBatch": [
        "cusparseCgpsvInterleavedBatch"
    ],
    "hipsparseCgpsvInterleavedBatch_bufferSizeExt": [
        "cusparseCgpsvInterleavedBatch_bufferSizeExt"
    ],
    "hipsparseCgthr": [
        "cusparseCgthr"
    ],
    "hipsparseCgthrz": [
        "cusparseCgthrz"
    ],
    "hipsparseCgtsv2": [
        "cusparseCgtsv2"
    ],
    "hipsparseCgtsv2StridedBatch": [
        "cusparseCgtsv2StridedBatch"
    ],
    "hipsparseCgtsv2StridedBatch_bufferSizeExt": [
        "cusparseCgtsv2StridedBatch_bufferSizeExt"
    ],
    "hipsparseCgtsv2_bufferSizeExt": [
        "cusparseCgtsv2_bufferSizeExt"
    ],
    "hipsparseCgtsv2_nopivot": [
        "cusparseCgtsv2_nopivot"
    ],
    "hipsparseCgtsv2_nopivot_bufferSizeExt": [
        "cusparseCgtsv2_nopivot_bufferSizeExt"
    ],
    "hipsparseCgtsvInterleavedBatch": [
        "cusparseCgtsvInterleavedBatch"
    ],
    "hipsparseCgtsvInterleavedBatch_bufferSizeExt": [
        "cusparseCgtsvInterleavedBatch_bufferSizeExt"
    ],
    "hipsparseChyb2csr": [
        "cusparseChyb2csr"
    ],
    "hipsparseChybmv": [
        "cusparseChybmv"
    ],
    "hipsparseCnnz": [
        "cusparseCnnz"
    ],
    "hipsparseCnnz_compress": [
        "cusparseCnnz_compress"
    ],
    "hipsparseCooAoSGet": [
        "cusparseCooAoSGet"
    ],
    "hipsparseCooGet": [
        "cusparseCooGet"
    ],
    "hipsparseCooSetPointers": [
        "cusparseCooSetPointers"
    ],
    "hipsparseCooSetStridedBatch": [
        "cusparseCooSetStridedBatch"
    ],
    "hipsparseCreate": [
        "cusparseCreate"
    ],
    "hipsparseCreateBlockedEll": [
        "cusparseCreateBlockedEll"
    ],
    "hipsparseCreateBsric02Info": [
        "cusparseCreateBsric02Info"
    ],
    "hipsparseCreateBsrilu02Info": [
        "cusparseCreateBsrilu02Info"
    ],
    "hipsparseCreateBsrsm2Info": [
        "cusparseCreateBsrsm2Info"
    ],
    "hipsparseCreateBsrsv2Info": [
        "cusparseCreateBsrsv2Info"
    ],
    "hipsparseCreateColorInfo": [
        "cusparseCreateColorInfo"
    ],
    "hipsparseCreateCoo": [
        "cusparseCreateCoo"
    ],
    "hipsparseCreateCooAoS": [
        "cusparseCreateCooAoS"
    ],
    "hipsparseCreateCsc": [
        "cusparseCreateCsc"
    ],
    "hipsparseCreateCsr": [
        "cusparseCreateCsr"
    ],
    "hipsparseCreateCsrgemm2Info": [
        "cusparseCreateCsrgemm2Info"
    ],
    "hipsparseCreateCsric02Info": [
        "cusparseCreateCsric02Info"
    ],
    "hipsparseCreateCsrilu02Info": [
        "cusparseCreateCsrilu02Info"
    ],
    "hipsparseCreateCsrsm2Info": [
        "cusparseCreateCsrsm2Info"
    ],
    "hipsparseCreateCsrsv2Info": [
        "cusparseCreateCsrsv2Info"
    ],
    "hipsparseCreateCsru2csrInfo": [
        "cusparseCreateCsru2csrInfo"
    ],
    "hipsparseCreateDnMat": [
        "cusparseCreateDnMat"
    ],
    "hipsparseCreateDnVec": [
        "cusparseCreateDnVec"
    ],
    "hipsparseCreateHybMat": [
        "cusparseCreateHybMat"
    ],
    "hipsparseCreateIdentityPermutation": [
        "cusparseCreateIdentityPermutation"
    ],
    "hipsparseCreateMatDescr": [
        "cusparseCreateMatDescr"
    ],
    "hipsparseCreatePruneInfo": [
        "cusparseCreatePruneInfo"
    ],
    "hipsparseCreateSpVec": [
        "cusparseCreateSpVec"
    ],
    "hipsparseCscSetPointers": [
        "cusparseCscSetPointers"
    ],
    "hipsparseCsctr": [
        "cusparseCsctr"
    ],
    "hipsparseCsrGet": [
        "cusparseCsrGet"
    ],
    "hipsparseCsrSetPointers": [
        "cusparseCsrSetPointers"
    ],
    "hipsparseCsrSetStridedBatch": [
        "cusparseCsrSetStridedBatch"
    ],
    "hipsparseDaxpyi": [
        "cusparseDaxpyi"
    ],
    "hipsparseDbsr2csr": [
        "cusparseDbsr2csr"
    ],
    "hipsparseDbsric02": [
        "cusparseDbsric02"
    ],
    "hipsparseDbsric02_analysis": [
        "cusparseDbsric02_analysis"
    ],
    "hipsparseDbsric02_bufferSize": [
        "cusparseDbsric02_bufferSize"
    ],
    "hipsparseDbsrilu02": [
        "cusparseDbsrilu02"
    ],
    "hipsparseDbsrilu02_analysis": [
        "cusparseDbsrilu02_analysis"
    ],
    "hipsparseDbsrilu02_bufferSize": [
        "cusparseDbsrilu02_bufferSize"
    ],
    "hipsparseDbsrilu02_numericBoost": [
        "cusparseDbsrilu02_numericBoost"
    ],
    "hipsparseDbsrmm": [
        "cusparseDbsrmm"
    ],
    "hipsparseDbsrmv": [
        "cusparseDbsrmv"
    ],
    "hipsparseDbsrsm2_analysis": [
        "cusparseDbsrsm2_analysis"
    ],
    "hipsparseDbsrsm2_bufferSize": [
        "cusparseDbsrsm2_bufferSize"
    ],
    "hipsparseDbsrsm2_solve": [
        "cusparseDbsrsm2_solve"
    ],
    "hipsparseDbsrsv2_analysis": [
        "cusparseDbsrsv2_analysis"
    ],
    "hipsparseDbsrsv2_bufferSize": [
        "cusparseDbsrsv2_bufferSize"
    ],
    "hipsparseDbsrsv2_bufferSizeExt": [
        "cusparseDbsrsv2_bufferSizeExt"
    ],
    "hipsparseDbsrsv2_solve": [
        "cusparseDbsrsv2_solve"
    ],
    "hipsparseDbsrxmv": [
        "cusparseDbsrxmv"
    ],
    "hipsparseDcsc2dense": [
        "cusparseDcsc2dense"
    ],
    "hipsparseDcsr2bsr": [
        "cusparseDcsr2bsr"
    ],
    "hipsparseDcsr2csc": [
        "cusparseDcsr2csc"
    ],
    "hipsparseDcsr2csr_compress": [
        "cusparseDcsr2csr_compress"
    ],
    "hipsparseDcsr2csru": [
        "cusparseDcsr2csru"
    ],
    "hipsparseDcsr2dense": [
        "cusparseDcsr2dense"
    ],
    "hipsparseDcsr2gebsr": [
        "cusparseDcsr2gebsr"
    ],
    "hipsparseDcsr2gebsr_bufferSize": [
        "cusparseDcsr2gebsr_bufferSize"
    ],
    "hipsparseDcsr2hyb": [
        "cusparseDcsr2hyb"
    ],
    "hipsparseDcsrcolor": [
        "cusparseDcsrcolor"
    ],
    "hipsparseDcsrgeam": [
        "cusparseDcsrgeam"
    ],
    "hipsparseDcsrgeam2": [
        "cusparseDcsrgeam2"
    ],
    "hipsparseDcsrgeam2_bufferSizeExt": [
        "cusparseDcsrgeam2_bufferSizeExt"
    ],
    "hipsparseDcsrgemm": [
        "cusparseDcsrgemm"
    ],
    "hipsparseDcsrgemm2": [
        "cusparseDcsrgemm2"
    ],
    "hipsparseDcsrgemm2_bufferSizeExt": [
        "cusparseDcsrgemm2_bufferSizeExt"
    ],
    "hipsparseDcsric02": [
        "cusparseDcsric02"
    ],
    "hipsparseDcsric02_analysis": [
        "cusparseDcsric02_analysis"
    ],
    "hipsparseDcsric02_bufferSize": [
        "cusparseDcsric02_bufferSize"
    ],
    "hipsparseDcsric02_bufferSizeExt": [
        "cusparseDcsric02_bufferSizeExt"
    ],
    "hipsparseDcsrilu02": [
        "cusparseDcsrilu02"
    ],
    "hipsparseDcsrilu02_analysis": [
        "cusparseDcsrilu02_analysis"
    ],
    "hipsparseDcsrilu02_bufferSize": [
        "cusparseDcsrilu02_bufferSize"
    ],
    "hipsparseDcsrilu02_bufferSizeExt": [
        "cusparseDcsrilu02_bufferSizeExt"
    ],
    "hipsparseDcsrilu02_numericBoost": [
        "cusparseDcsrilu02_numericBoost"
    ],
    "hipsparseDcsrmm": [
        "cusparseDcsrmm"
    ],
    "hipsparseDcsrmm2": [
        "cusparseDcsrmm2"
    ],
    "hipsparseDcsrmv": [
        "cusparseDcsrmv"
    ],
    "hipsparseDcsrsm2_analysis": [
        "cusparseDcsrsm2_analysis"
    ],
    "hipsparseDcsrsm2_bufferSizeExt": [
        "cusparseDcsrsm2_bufferSizeExt"
    ],
    "hipsparseDcsrsm2_solve": [
        "cusparseDcsrsm2_solve"
    ],
    "hipsparseDcsrsv2_analysis": [
        "cusparseDcsrsv2_analysis"
    ],
    "hipsparseDcsrsv2_bufferSize": [
        "cusparseDcsrsv2_bufferSize"
    ],
    "hipsparseDcsrsv2_bufferSizeExt": [
        "cusparseDcsrsv2_bufferSizeExt"
    ],
    "hipsparseDcsrsv2_solve": [
        "cusparseDcsrsv2_solve"
    ],
    "hipsparseDcsru2csr": [
        "cusparseDcsru2csr"
    ],
    "hipsparseDcsru2csr_bufferSizeExt": [
        "cusparseDcsru2csr_bufferSizeExt"
    ],
    "hipsparseDdense2csc": [
        "cusparseDdense2csc"
    ],
    "hipsparseDdense2csr": [
        "cusparseDdense2csr"
    ],
    "hipsparseDdoti": [
        "cusparseDdoti"
    ],
    "hipsparseDenseToSparse_analysis": [
        "cusparseDenseToSparse_analysis"
    ],
    "hipsparseDenseToSparse_bufferSize": [
        "cusparseDenseToSparse_bufferSize"
    ],
    "hipsparseDenseToSparse_convert": [
        "cusparseDenseToSparse_convert"
    ],
    "hipsparseDestroy": [
        "cusparseDestroy"
    ],
    "hipsparseDestroyBsric02Info": [
        "cusparseDestroyBsric02Info"
    ],
    "hipsparseDestroyBsrilu02Info": [
        "cusparseDestroyBsrilu02Info"
    ],
    "hipsparseDestroyBsrsm2Info": [
        "cusparseDestroyBsrsm2Info"
    ],
    "hipsparseDestroyBsrsv2Info": [
        "cusparseDestroyBsrsv2Info"
    ],
    "hipsparseDestroyColorInfo": [
        "cusparseDestroyColorInfo"
    ],
    "hipsparseDestroyCsrgemm2Info": [
        "cusparseDestroyCsrgemm2Info"
    ],
    "hipsparseDestroyCsric02Info": [
        "cusparseDestroyCsric02Info"
    ],
    "hipsparseDestroyCsrilu02Info": [
        "cusparseDestroyCsrilu02Info"
    ],
    "hipsparseDestroyCsrsm2Info": [
        "cusparseDestroyCsrsm2Info"
    ],
    "hipsparseDestroyCsrsv2Info": [
        "cusparseDestroyCsrsv2Info"
    ],
    "hipsparseDestroyCsru2csrInfo": [
        "cusparseDestroyCsru2csrInfo"
    ],
    "hipsparseDestroyDnMat": [
        "cusparseDestroyDnMat"
    ],
    "hipsparseDestroyDnVec": [
        "cusparseDestroyDnVec"
    ],
    "hipsparseDestroyHybMat": [
        "cusparseDestroyHybMat"
    ],
    "hipsparseDestroyMatDescr": [
        "cusparseDestroyMatDescr"
    ],
    "hipsparseDestroyPruneInfo": [
        "cusparseDestroyPruneInfo"
    ],
    "hipsparseDestroySpMat": [
        "cusparseDestroySpMat"
    ],
    "hipsparseDestroySpVec": [
        "cusparseDestroySpVec"
    ],
    "hipsparseDgebsr2csr": [
        "cusparseDgebsr2csr"
    ],
    "hipsparseDgebsr2gebsc": [
        "cusparseDgebsr2gebsc"
    ],
    "hipsparseDgebsr2gebsc_bufferSize": [
        "cusparseDgebsr2gebsc_bufferSize"
    ],
    "hipsparseDgebsr2gebsr": [
        "cusparseDgebsr2gebsr"
    ],
    "hipsparseDgebsr2gebsr_bufferSize": [
        "cusparseDgebsr2gebsr_bufferSize"
    ],
    "hipsparseDgemmi": [
        "cusparseDgemmi"
    ],
    "hipsparseDgemvi": [
        "cusparseDgemvi"
    ],
    "hipsparseDgemvi_bufferSize": [
        "cusparseDgemvi_bufferSize"
    ],
    "hipsparseDgpsvInterleavedBatch": [
        "cusparseDgpsvInterleavedBatch"
    ],
    "hipsparseDgpsvInterleavedBatch_bufferSizeExt": [
        "cusparseDgpsvInterleavedBatch_bufferSizeExt"
    ],
    "hipsparseDgthr": [
        "cusparseDgthr"
    ],
    "hipsparseDgthrz": [
        "cusparseDgthrz"
    ],
    "hipsparseDgtsv2": [
        "cusparseDgtsv2"
    ],
    "hipsparseDgtsv2StridedBatch": [
        "cusparseDgtsv2StridedBatch"
    ],
    "hipsparseDgtsv2StridedBatch_bufferSizeExt": [
        "cusparseDgtsv2StridedBatch_bufferSizeExt"
    ],
    "hipsparseDgtsv2_bufferSizeExt": [
        "cusparseDgtsv2_bufferSizeExt"
    ],
    "hipsparseDgtsv2_nopivot": [
        "cusparseDgtsv2_nopivot"
    ],
    "hipsparseDgtsv2_nopivot_bufferSizeExt": [
        "cusparseDgtsv2_nopivot_bufferSizeExt"
    ],
    "hipsparseDgtsvInterleavedBatch": [
        "cusparseDgtsvInterleavedBatch"
    ],
    "hipsparseDgtsvInterleavedBatch_bufferSizeExt": [
        "cusparseDgtsvInterleavedBatch_bufferSizeExt"
    ],
    "hipsparseDhyb2csr": [
        "cusparseDhyb2csr"
    ],
    "hipsparseDhybmv": [
        "cusparseDhybmv"
    ],
    "hipsparseDnMatGet": [
        "cusparseDnMatGet"
    ],
    "hipsparseDnMatGetStridedBatch": [
        "cusparseDnMatGetStridedBatch"
    ],
    "hipsparseDnMatGetValues": [
        "cusparseDnMatGetValues"
    ],
    "hipsparseDnMatSetStridedBatch": [
        "cusparseDnMatSetStridedBatch"
    ],
    "hipsparseDnMatSetValues": [
        "cusparseDnMatSetValues"
    ],
    "hipsparseDnVecGet": [
        "cusparseDnVecGet"
    ],
    "hipsparseDnVecGetValues": [
        "cusparseDnVecGetValues"
    ],
    "hipsparseDnVecSetValues": [
        "cusparseDnVecSetValues"
    ],
    "hipsparseDnnz": [
        "cusparseDnnz"
    ],
    "hipsparseDnnz_compress": [
        "cusparseDnnz_compress"
    ],
    "hipsparseDpruneCsr2csr": [
        "cusparseDpruneCsr2csr"
    ],
    "hipsparseDpruneCsr2csrByPercentage": [
        "cusparseDpruneCsr2csrByPercentage"
    ],
    "hipsparseDpruneCsr2csrByPercentage_bufferSizeExt": [
        "cusparseDpruneCsr2csrByPercentage_bufferSizeExt"
    ],
    "hipsparseDpruneCsr2csrNnz": [
        "cusparseDpruneCsr2csrNnz"
    ],
    "hipsparseDpruneCsr2csrNnzByPercentage": [
        "cusparseDpruneCsr2csrNnzByPercentage"
    ],
    "hipsparseDpruneCsr2csr_bufferSizeExt": [
        "cusparseDpruneCsr2csr_bufferSizeExt"
    ],
    "hipsparseDpruneDense2csr": [
        "cusparseDpruneDense2csr"
    ],
    "hipsparseDpruneDense2csrByPercentage": [
        "cusparseDpruneDense2csrByPercentage"
    ],
    "hipsparseDpruneDense2csrByPercentage_bufferSizeExt": [
        "cusparseDpruneDense2csrByPercentage_bufferSizeExt"
    ],
    "hipsparseDpruneDense2csrNnz": [
        "cusparseDpruneDense2csrNnz"
    ],
    "hipsparseDpruneDense2csrNnzByPercentage": [
        "cusparseDpruneDense2csrNnzByPercentage"
    ],
    "hipsparseDpruneDense2csr_bufferSizeExt": [
        "cusparseDpruneDense2csr_bufferSizeExt"
    ],
    "hipsparseDroti": [
        "cusparseDroti"
    ],
    "hipsparseDsctr": [
        "cusparseDsctr"
    ],
    "hipsparseGather": [
        "cusparseGather"
    ],
    "hipsparseGetMatDiagType": [
        "cusparseGetMatDiagType"
    ],
    "hipsparseGetMatFillMode": [
        "cusparseGetMatFillMode"
    ],
    "hipsparseGetMatIndexBase": [
        "cusparseGetMatIndexBase"
    ],
    "hipsparseGetMatType": [
        "cusparseGetMatType"
    ],
    "hipsparseGetPointerMode": [
        "cusparseGetPointerMode"
    ],
    "hipsparseGetStream": [
        "cusparseGetStream"
    ],
    "hipsparseGetVersion": [
        "cusparseGetVersion"
    ],
    "hipsparseRot": [
        "cusparseRot"
    ],
    "hipsparseSDDMM": [
        "cusparseSDDMM"
    ],
    "hipsparseSDDMM_bufferSize": [
        "cusparseSDDMM_bufferSize"
    ],
    "hipsparseSDDMM_preprocess": [
        "cusparseSDDMM_preprocess"
    ],
    "hipsparseSaxpyi": [
        "cusparseSaxpyi"
    ],
    "hipsparseSbsr2csr": [
        "cusparseSbsr2csr"
    ],
    "hipsparseSbsric02": [
        "cusparseSbsric02"
    ],
    "hipsparseSbsric02_analysis": [
        "cusparseSbsric02_analysis"
    ],
    "hipsparseSbsric02_bufferSize": [
        "cusparseSbsric02_bufferSize"
    ],
    "hipsparseSbsrilu02": [
        "cusparseSbsrilu02"
    ],
    "hipsparseSbsrilu02_analysis": [
        "cusparseSbsrilu02_analysis"
    ],
    "hipsparseSbsrilu02_bufferSize": [
        "cusparseSbsrilu02_bufferSize"
    ],
    "hipsparseSbsrilu02_numericBoost": [
        "cusparseSbsrilu02_numericBoost"
    ],
    "hipsparseSbsrmm": [
        "cusparseSbsrmm"
    ],
    "hipsparseSbsrmv": [
        "cusparseSbsrmv"
    ],
    "hipsparseSbsrsm2_analysis": [
        "cusparseSbsrsm2_analysis"
    ],
    "hipsparseSbsrsm2_bufferSize": [
        "cusparseSbsrsm2_bufferSize"
    ],
    "hipsparseSbsrsm2_solve": [
        "cusparseSbsrsm2_solve"
    ],
    "hipsparseSbsrsv2_analysis": [
        "cusparseSbsrsv2_analysis"
    ],
    "hipsparseSbsrsv2_bufferSize": [
        "cusparseSbsrsv2_bufferSize"
    ],
    "hipsparseSbsrsv2_bufferSizeExt": [
        "cusparseSbsrsv2_bufferSizeExt"
    ],
    "hipsparseSbsrsv2_solve": [
        "cusparseSbsrsv2_solve"
    ],
    "hipsparseSbsrxmv": [
        "cusparseSbsrxmv"
    ],
    "hipsparseScatter": [
        "cusparseScatter"
    ],
    "hipsparseScsc2dense": [
        "cusparseScsc2dense"
    ],
    "hipsparseScsr2bsr": [
        "cusparseScsr2bsr"
    ],
    "hipsparseScsr2csc": [
        "cusparseScsr2csc"
    ],
    "hipsparseScsr2csr_compress": [
        "cusparseScsr2csr_compress"
    ],
    "hipsparseScsr2csru": [
        "cusparseScsr2csru"
    ],
    "hipsparseScsr2dense": [
        "cusparseScsr2dense"
    ],
    "hipsparseScsr2gebsr": [
        "cusparseScsr2gebsr"
    ],
    "hipsparseScsr2gebsr_bufferSize": [
        "cusparseScsr2gebsr_bufferSize"
    ],
    "hipsparseScsr2hyb": [
        "cusparseScsr2hyb"
    ],
    "hipsparseScsrcolor": [
        "cusparseScsrcolor"
    ],
    "hipsparseScsrgeam": [
        "cusparseScsrgeam"
    ],
    "hipsparseScsrgeam2": [
        "cusparseScsrgeam2"
    ],
    "hipsparseScsrgeam2_bufferSizeExt": [
        "cusparseScsrgeam2_bufferSizeExt"
    ],
    "hipsparseScsrgemm": [
        "cusparseScsrgemm"
    ],
    "hipsparseScsrgemm2": [
        "cusparseScsrgemm2"
    ],
    "hipsparseScsrgemm2_bufferSizeExt": [
        "cusparseScsrgemm2_bufferSizeExt"
    ],
    "hipsparseScsric02": [
        "cusparseScsric02"
    ],
    "hipsparseScsric02_analysis": [
        "cusparseScsric02_analysis"
    ],
    "hipsparseScsric02_bufferSize": [
        "cusparseScsric02_bufferSize"
    ],
    "hipsparseScsric02_bufferSizeExt": [
        "cusparseScsric02_bufferSizeExt"
    ],
    "hipsparseScsrilu02": [
        "cusparseScsrilu02"
    ],
    "hipsparseScsrilu02_analysis": [
        "cusparseScsrilu02_analysis"
    ],
    "hipsparseScsrilu02_bufferSize": [
        "cusparseScsrilu02_bufferSize"
    ],
    "hipsparseScsrilu02_bufferSizeExt": [
        "cusparseScsrilu02_bufferSizeExt"
    ],
    "hipsparseScsrilu02_numericBoost": [
        "cusparseScsrilu02_numericBoost"
    ],
    "hipsparseScsrmm": [
        "cusparseScsrmm"
    ],
    "hipsparseScsrmm2": [
        "cusparseScsrmm2"
    ],
    "hipsparseScsrmv": [
        "cusparseScsrmv"
    ],
    "hipsparseScsrsm2_analysis": [
        "cusparseScsrsm2_analysis"
    ],
    "hipsparseScsrsm2_bufferSizeExt": [
        "cusparseScsrsm2_bufferSizeExt"
    ],
    "hipsparseScsrsm2_solve": [
        "cusparseScsrsm2_solve"
    ],
    "hipsparseScsrsv2_analysis": [
        "cusparseScsrsv2_analysis"
    ],
    "hipsparseScsrsv2_bufferSize": [
        "cusparseScsrsv2_bufferSize"
    ],
    "hipsparseScsrsv2_bufferSizeExt": [
        "cusparseScsrsv2_bufferSizeExt"
    ],
    "hipsparseScsrsv2_solve": [
        "cusparseScsrsv2_solve"
    ],
    "hipsparseScsru2csr": [
        "cusparseScsru2csr"
    ],
    "hipsparseScsru2csr_bufferSizeExt": [
        "cusparseScsru2csr_bufferSizeExt"
    ],
    "hipsparseSdense2csc": [
        "cusparseSdense2csc"
    ],
    "hipsparseSdense2csr": [
        "cusparseSdense2csr"
    ],
    "hipsparseSdoti": [
        "cusparseSdoti"
    ],
    "hipsparseSetMatDiagType": [
        "cusparseSetMatDiagType"
    ],
    "hipsparseSetMatFillMode": [
        "cusparseSetMatFillMode"
    ],
    "hipsparseSetMatIndexBase": [
        "cusparseSetMatIndexBase"
    ],
    "hipsparseSetMatType": [
        "cusparseSetMatType"
    ],
    "hipsparseSetPointerMode": [
        "cusparseSetPointerMode"
    ],
    "hipsparseSetStream": [
        "cusparseSetStream"
    ],
    "hipsparseSgebsr2csr": [
        "cusparseSgebsr2csr"
    ],
    "hipsparseSgebsr2gebsc": [
        "cusparseSgebsr2gebsc"
    ],
    "hipsparseSgebsr2gebsc_bufferSize": [
        "cusparseSgebsr2gebsc_bufferSize"
    ],
    "hipsparseSgebsr2gebsr": [
        "cusparseSgebsr2gebsr"
    ],
    "hipsparseSgebsr2gebsr_bufferSize": [
        "cusparseSgebsr2gebsr_bufferSize"
    ],
    "hipsparseSgemmi": [
        "cusparseSgemmi"
    ],
    "hipsparseSgemvi": [
        "cusparseSgemvi"
    ],
    "hipsparseSgemvi_bufferSize": [
        "cusparseSgemvi_bufferSize"
    ],
    "hipsparseSgpsvInterleavedBatch": [
        "cusparseSgpsvInterleavedBatch"
    ],
    "hipsparseSgpsvInterleavedBatch_bufferSizeExt": [
        "cusparseSgpsvInterleavedBatch_bufferSizeExt"
    ],
    "hipsparseSgthr": [
        "cusparseSgthr"
    ],
    "hipsparseSgthrz": [
        "cusparseSgthrz"
    ],
    "hipsparseSgtsv2": [
        "cusparseSgtsv2"
    ],
    "hipsparseSgtsv2StridedBatch": [
        "cusparseSgtsv2StridedBatch"
    ],
    "hipsparseSgtsv2StridedBatch_bufferSizeExt": [
        "cusparseSgtsv2StridedBatch_bufferSizeExt"
    ],
    "hipsparseSgtsv2_bufferSizeExt": [
        "cusparseSgtsv2_bufferSizeExt"
    ],
    "hipsparseSgtsv2_nopivot": [
        "cusparseSgtsv2_nopivot"
    ],
    "hipsparseSgtsv2_nopivot_bufferSizeExt": [
        "cusparseSgtsv2_nopivot_bufferSizeExt"
    ],
    "hipsparseSgtsvInterleavedBatch": [
        "cusparseSgtsvInterleavedBatch"
    ],
    "hipsparseSgtsvInterleavedBatch_bufferSizeExt": [
        "cusparseSgtsvInterleavedBatch_bufferSizeExt"
    ],
    "hipsparseShyb2csr": [
        "cusparseShyb2csr"
    ],
    "hipsparseShybmv": [
        "cusparseShybmv"
    ],
    "hipsparseSnnz": [
        "cusparseSnnz"
    ],
    "hipsparseSnnz_compress": [
        "cusparseSnnz_compress"
    ],
    "hipsparseSpGEMM_compute": [
        "cusparseSpGEMM_compute"
    ],
    "hipsparseSpGEMM_copy": [
        "cusparseSpGEMM_copy"
    ],
    "hipsparseSpGEMM_createDescr": [
        "cusparseSpGEMM_createDescr"
    ],
    "hipsparseSpGEMM_destroyDescr": [
        "cusparseSpGEMM_destroyDescr"
    ],
    "hipsparseSpGEMM_workEstimation": [
        "cusparseSpGEMM_workEstimation"
    ],
    "hipsparseSpMM": [
        "cusparseSpMM"
    ],
    "hipsparseSpMM_bufferSize": [
        "cusparseSpMM_bufferSize"
    ],
    "hipsparseSpMM_preprocess": [
        "cusparseSpMM_preprocess"
    ],
    "hipsparseSpMV": [
        "cusparseSpMV"
    ],
    "hipsparseSpMV_bufferSize": [
        "cusparseSpMV_bufferSize"
    ],
    "hipsparseSpMatGetAttribute": [
        "cusparseSpMatGetAttribute"
    ],
    "hipsparseSpMatGetFormat": [
        "cusparseSpMatGetFormat"
    ],
    "hipsparseSpMatGetIndexBase": [
        "cusparseSpMatGetIndexBase"
    ],
    "hipsparseSpMatGetSize": [
        "cusparseSpMatGetSize"
    ],
    "hipsparseSpMatGetStridedBatch": [
        "cusparseSpMatGetStridedBatch"
    ],
    "hipsparseSpMatGetValues": [
        "cusparseSpMatGetValues"
    ],
    "hipsparseSpMatSetAttribute": [
        "cusparseSpMatSetAttribute"
    ],
    "hipsparseSpMatSetStridedBatch": [
        "cusparseSpMatSetStridedBatch"
    ],
    "hipsparseSpMatSetValues": [
        "cusparseSpMatSetValues"
    ],
    "hipsparseSpSM_analysis": [
        "cusparseSpSM_analysis"
    ],
    "hipsparseSpSM_bufferSize": [
        "cusparseSpSM_bufferSize"
    ],
    "hipsparseSpSM_createDescr": [
        "cusparseSpSM_createDescr"
    ],
    "hipsparseSpSM_destroyDescr": [
        "cusparseSpSM_destroyDescr"
    ],
    "hipsparseSpSM_solve": [
        "cusparseSpSM_solve"
    ],
    "hipsparseSpSV_analysis": [
        "cusparseSpSV_analysis"
    ],
    "hipsparseSpSV_bufferSize": [
        "cusparseSpSV_bufferSize"
    ],
    "hipsparseSpSV_createDescr": [
        "cusparseSpSV_createDescr"
    ],
    "hipsparseSpSV_destroyDescr": [
        "cusparseSpSV_destroyDescr"
    ],
    "hipsparseSpSV_solve": [
        "cusparseSpSV_solve"
    ],
    "hipsparseSpVV": [
        "cusparseSpVV"
    ],
    "hipsparseSpVV_bufferSize": [
        "cusparseSpVV_bufferSize"
    ],
    "hipsparseSpVecGet": [
        "cusparseSpVecGet"
    ],
    "hipsparseSpVecGetIndexBase": [
        "cusparseSpVecGetIndexBase"
    ],
    "hipsparseSpVecGetValues": [
        "cusparseSpVecGetValues"
    ],
    "hipsparseSpVecSetValues": [
        "cusparseSpVecSetValues"
    ],
    "hipsparseSparseToDense": [
        "cusparseSparseToDense"
    ],
    "hipsparseSparseToDense_bufferSize": [
        "cusparseSparseToDense_bufferSize"
    ],
    "hipsparseSpruneCsr2csr": [
        "cusparseSpruneCsr2csr"
    ],
    "hipsparseSpruneCsr2csrByPercentage": [
        "cusparseSpruneCsr2csrByPercentage"
    ],
    "hipsparseSpruneCsr2csrByPercentage_bufferSizeExt": [
        "cusparseSpruneCsr2csrByPercentage_bufferSizeExt"
    ],
    "hipsparseSpruneCsr2csrNnz": [
        "cusparseSpruneCsr2csrNnz"
    ],
    "hipsparseSpruneCsr2csrNnzByPercentage": [
        "cusparseSpruneCsr2csrNnzByPercentage"
    ],
    "hipsparseSpruneCsr2csr_bufferSizeExt": [
        "cusparseSpruneCsr2csr_bufferSizeExt"
    ],
    "hipsparseSpruneDense2csr": [
        "cusparseSpruneDense2csr"
    ],
    "hipsparseSpruneDense2csrByPercentage": [
        "cusparseSpruneDense2csrByPercentage"
    ],
    "hipsparseSpruneDense2csrByPercentage_bufferSizeExt": [
        "cusparseSpruneDense2csrByPercentage_bufferSizeExt"
    ],
    "hipsparseSpruneDense2csrNnz": [
        "cusparseSpruneDense2csrNnz"
    ],
    "hipsparseSpruneDense2csrNnzByPercentage": [
        "cusparseSpruneDense2csrNnzByPercentage"
    ],
    "hipsparseSpruneDense2csr_bufferSizeExt": [
        "cusparseSpruneDense2csr_bufferSizeExt"
    ],
    "hipsparseSroti": [
        "cusparseSroti"
    ],
    "hipsparseSsctr": [
        "cusparseSsctr"
    ],
    "hipsparseXbsric02_zeroPivot": [
        "cusparseXbsric02_zeroPivot"
    ],
    "hipsparseXbsrilu02_zeroPivot": [
        "cusparseXbsrilu02_zeroPivot"
    ],
    "hipsparseXbsrsm2_zeroPivot": [
        "cusparseXbsrsm2_zeroPivot"
    ],
    "hipsparseXbsrsv2_zeroPivot": [
        "cusparseXbsrsv2_zeroPivot"
    ],
    "hipsparseXcoo2csr": [
        "cusparseXcoo2csr"
    ],
    "hipsparseXcoosortByColumn": [
        "cusparseXcoosortByColumn"
    ],
    "hipsparseXcoosortByRow": [
        "cusparseXcoosortByRow"
    ],
    "hipsparseXcoosort_bufferSizeExt": [
        "cusparseXcoosort_bufferSizeExt"
    ],
    "hipsparseXcscsort": [
        "cusparseXcscsort"
    ],
    "hipsparseXcscsort_bufferSizeExt": [
        "cusparseXcscsort_bufferSizeExt"
    ],
    "hipsparseXcsr2bsrNnz": [
        "cusparseXcsr2bsrNnz"
    ],
    "hipsparseXcsr2coo": [
        "cusparseXcsr2coo"
    ],
    "hipsparseXcsr2gebsrNnz": [
        "cusparseXcsr2gebsrNnz"
    ],
    "hipsparseXcsrgeam2Nnz": [
        "cusparseXcsrgeam2Nnz"
    ],
    "hipsparseXcsrgeamNnz": [
        "cusparseXcsrgeamNnz"
    ],
    "hipsparseXcsrgemm2Nnz": [
        "cusparseXcsrgemm2Nnz"
    ],
    "hipsparseXcsrgemmNnz": [
        "cusparseXcsrgemmNnz"
    ],
    "hipsparseXcsric02_zeroPivot": [
        "cusparseXcsric02_zeroPivot"
    ],
    "hipsparseXcsrilu02_zeroPivot": [
        "cusparseXcsrilu02_zeroPivot"
    ],
    "hipsparseXcsrsm2_zeroPivot": [
        "cusparseXcsrsm2_zeroPivot"
    ],
    "hipsparseXcsrsort": [
        "cusparseXcsrsort"
    ],
    "hipsparseXcsrsort_bufferSizeExt": [
        "cusparseXcsrsort_bufferSizeExt"
    ],
    "hipsparseXcsrsv2_zeroPivot": [
        "cusparseXcsrsv2_zeroPivot"
    ],
    "hipsparseXgebsr2gebsrNnz": [
        "cusparseXgebsr2gebsrNnz"
    ],
    "hipsparseZaxpyi": [
        "cusparseZaxpyi"
    ],
    "hipsparseZbsr2csr": [
        "cusparseZbsr2csr"
    ],
    "hipsparseZbsric02": [
        "cusparseZbsric02"
    ],
    "hipsparseZbsric02_analysis": [
        "cusparseZbsric02_analysis"
    ],
    "hipsparseZbsric02_bufferSize": [
        "cusparseZbsric02_bufferSize"
    ],
    "hipsparseZbsrilu02": [
        "cusparseZbsrilu02"
    ],
    "hipsparseZbsrilu02_analysis": [
        "cusparseZbsrilu02_analysis"
    ],
    "hipsparseZbsrilu02_bufferSize": [
        "cusparseZbsrilu02_bufferSize"
    ],
    "hipsparseZbsrilu02_numericBoost": [
        "cusparseZbsrilu02_numericBoost"
    ],
    "hipsparseZbsrmm": [
        "cusparseZbsrmm"
    ],
    "hipsparseZbsrmv": [
        "cusparseZbsrmv"
    ],
    "hipsparseZbsrsm2_analysis": [
        "cusparseZbsrsm2_analysis"
    ],
    "hipsparseZbsrsm2_bufferSize": [
        "cusparseZbsrsm2_bufferSize"
    ],
    "hipsparseZbsrsm2_solve": [
        "cusparseZbsrsm2_solve"
    ],
    "hipsparseZbsrsv2_analysis": [
        "cusparseZbsrsv2_analysis"
    ],
    "hipsparseZbsrsv2_bufferSize": [
        "cusparseZbsrsv2_bufferSize"
    ],
    "hipsparseZbsrsv2_bufferSizeExt": [
        "cusparseZbsrsv2_bufferSizeExt"
    ],
    "hipsparseZbsrsv2_solve": [
        "cusparseZbsrsv2_solve"
    ],
    "hipsparseZbsrxmv": [
        "cusparseZbsrxmv"
    ],
    "hipsparseZcsc2dense": [
        "cusparseZcsc2dense"
    ],
    "hipsparseZcsr2bsr": [
        "cusparseZcsr2bsr"
    ],
    "hipsparseZcsr2csc": [
        "cusparseZcsr2csc"
    ],
    "hipsparseZcsr2csr_compress": [
        "cusparseZcsr2csr_compress"
    ],
    "hipsparseZcsr2csru": [
        "cusparseZcsr2csru"
    ],
    "hipsparseZcsr2dense": [
        "cusparseZcsr2dense"
    ],
    "hipsparseZcsr2gebsr": [
        "cusparseZcsr2gebsr"
    ],
    "hipsparseZcsr2gebsr_bufferSize": [
        "cusparseZcsr2gebsr_bufferSize"
    ],
    "hipsparseZcsr2hyb": [
        "cusparseZcsr2hyb"
    ],
    "hipsparseZcsrcolor": [
        "cusparseZcsrcolor"
    ],
    "hipsparseZcsrgeam": [
        "cusparseZcsrgeam"
    ],
    "hipsparseZcsrgeam2": [
        "cusparseZcsrgeam2"
    ],
    "hipsparseZcsrgeam2_bufferSizeExt": [
        "cusparseZcsrgeam2_bufferSizeExt"
    ],
    "hipsparseZcsrgemm": [
        "cusparseZcsrgemm"
    ],
    "hipsparseZcsrgemm2": [
        "cusparseZcsrgemm2"
    ],
    "hipsparseZcsrgemm2_bufferSizeExt": [
        "cusparseZcsrgemm2_bufferSizeExt"
    ],
    "hipsparseZcsric02": [
        "cusparseZcsric02"
    ],
    "hipsparseZcsric02_analysis": [
        "cusparseZcsric02_analysis"
    ],
    "hipsparseZcsric02_bufferSize": [
        "cusparseZcsric02_bufferSize"
    ],
    "hipsparseZcsric02_bufferSizeExt": [
        "cusparseZcsric02_bufferSizeExt"
    ],
    "hipsparseZcsrilu02": [
        "cusparseZcsrilu02"
    ],
    "hipsparseZcsrilu02_analysis": [
        "cusparseZcsrilu02_analysis"
    ],
    "hipsparseZcsrilu02_bufferSize": [
        "cusparseZcsrilu02_bufferSize"
    ],
    "hipsparseZcsrilu02_bufferSizeExt": [
        "cusparseZcsrilu02_bufferSizeExt"
    ],
    "hipsparseZcsrilu02_numericBoost": [
        "cusparseZcsrilu02_numericBoost"
    ],
    "hipsparseZcsrmm": [
        "cusparseZcsrmm"
    ],
    "hipsparseZcsrmm2": [
        "cusparseZcsrmm2"
    ],
    "hipsparseZcsrmv": [
        "cusparseZcsrmv"
    ],
    "hipsparseZcsrsm2_analysis": [
        "cusparseZcsrsm2_analysis"
    ],
    "hipsparseZcsrsm2_bufferSizeExt": [
        "cusparseZcsrsm2_bufferSizeExt"
    ],
    "hipsparseZcsrsm2_solve": [
        "cusparseZcsrsm2_solve"
    ],
    "hipsparseZcsrsv2_analysis": [
        "cusparseZcsrsv2_analysis"
    ],
    "hipsparseZcsrsv2_bufferSize": [
        "cusparseZcsrsv2_bufferSize"
    ],
    "hipsparseZcsrsv2_bufferSizeExt": [
        "cusparseZcsrsv2_bufferSizeExt"
    ],
    "hipsparseZcsrsv2_solve": [
        "cusparseZcsrsv2_solve"
    ],
    "hipsparseZcsru2csr": [
        "cusparseZcsru2csr"
    ],
    "hipsparseZcsru2csr_bufferSizeExt": [
        "cusparseZcsru2csr_bufferSizeExt"
    ],
    "hipsparseZdense2csc": [
        "cusparseZdense2csc"
    ],
    "hipsparseZdense2csr": [
        "cusparseZdense2csr"
    ],
    "hipsparseZdotci": [
        "cusparseZdotci"
    ],
    "hipsparseZdoti": [
        "cusparseZdoti"
    ],
    "hipsparseZgebsr2csr": [
        "cusparseZgebsr2csr"
    ],
    "hipsparseZgebsr2gebsc": [
        "cusparseZgebsr2gebsc"
    ],
    "hipsparseZgebsr2gebsc_bufferSize": [
        "cusparseZgebsr2gebsc_bufferSize"
    ],
    "hipsparseZgebsr2gebsr": [
        "cusparseZgebsr2gebsr"
    ],
    "hipsparseZgebsr2gebsr_bufferSize": [
        "cusparseZgebsr2gebsr_bufferSize"
    ],
    "hipsparseZgemmi": [
        "cusparseZgemmi"
    ],
    "hipsparseZgemvi": [
        "cusparseZgemvi"
    ],
    "hipsparseZgemvi_bufferSize": [
        "cusparseZgemvi_bufferSize"
    ],
    "hipsparseZgpsvInterleavedBatch": [
        "cusparseZgpsvInterleavedBatch"
    ],
    "hipsparseZgpsvInterleavedBatch_bufferSizeExt": [
        "cusparseZgpsvInterleavedBatch_bufferSizeExt"
    ],
    "hipsparseZgthr": [
        "cusparseZgthr"
    ],
    "hipsparseZgthrz": [
        "cusparseZgthrz"
    ],
    "hipsparseZgtsv2": [
        "cusparseZgtsv2"
    ],
    "hipsparseZgtsv2StridedBatch": [
        "cusparseZgtsv2StridedBatch"
    ],
    "hipsparseZgtsv2StridedBatch_bufferSizeExt": [
        "cusparseZgtsv2StridedBatch_bufferSizeExt"
    ],
    "hipsparseZgtsv2_bufferSizeExt": [
        "cusparseZgtsv2_bufferSizeExt"
    ],
    "hipsparseZgtsv2_nopivot": [
        "cusparseZgtsv2_nopivot"
    ],
    "hipsparseZgtsv2_nopivot_bufferSizeExt": [
        "cusparseZgtsv2_nopivot_bufferSizeExt"
    ],
    "hipsparseZgtsvInterleavedBatch": [
        "cusparseZgtsvInterleavedBatch"
    ],
    "hipsparseZgtsvInterleavedBatch_bufferSizeExt": [
        "cusparseZgtsvInterleavedBatch_bufferSizeExt"
    ],
    "hipsparseZhyb2csr": [
        "cusparseZhyb2csr"
    ],
    "hipsparseZhybmv": [
        "cusparseZhybmv"
    ],
    "hipsparseZnnz": [
        "cusparseZnnz"
    ],
    "hipsparseZnnz_compress": [
        "cusparseZnnz_compress"
    ],
    "hipsparseZsctr": [
        "cusparseZsctr"
    ],
    "hiprtcAddNameExpression": [
        "nvrtcAddNameExpression"
    ],
    "hiprtcCompileProgram": [
        "nvrtcCompileProgram"
    ],
    "hiprtcCreateProgram": [
        "nvrtcCreateProgram"
    ],
    "hiprtcDestroyProgram": [
        "nvrtcDestroyProgram"
    ],
    "hiprtcGetBitcode": [
        "nvrtcGetCUBIN"
    ],
    "hiprtcGetBitcodeSize": [
        "nvrtcGetCUBINSize"
    ],
    "hiprtcGetErrorString": [
        "nvrtcGetErrorString"
    ],
    "hiprtcGetLoweredName": [
        "nvrtcGetLoweredName"
    ],
    "hiprtcGetCode": [
        "nvrtcGetPTX"
    ],
    "hiprtcGetCodeSize": [
        "nvrtcGetPTXSize"
    ],
    "hiprtcGetProgramLog": [
        "nvrtcGetProgramLog"
    ],
    "hiprtcGetProgramLogSize": [
        "nvrtcGetProgramLogSize"
    ],
    "hiprtcVersion": [
        "nvrtcVersion"
    ],
    "hiprand": [
        "curand"
    ],
    "hiprand_discrete": [
        "curand_discrete"
    ],
    "hiprand_discrete4": [
        "curand_discrete4"
    ],
    "hiprand_init": [
        "curand_init"
    ],
    "hiprand_log_normal": [
        "curand_log_normal"
    ],
    "hiprand_log_normal2": [
        "curand_log_normal2"
    ],
    "hiprand_log_normal2_double": [
        "curand_log_normal2_double"
    ],
    "hiprand_log_normal4": [
        "curand_log_normal4"
    ],
    "hiprand_log_normal4_double": [
        "curand_log_normal4_double"
    ],
    "hiprand_log_normal_double": [
        "curand_log_normal_double"
    ],
    "hiprand_normal": [
        "curand_normal"
    ],
    "hiprand_normal2": [
        "curand_normal2"
    ],
    "hiprand_normal2_double": [
        "curand_normal2_double"
    ],
    "hiprand_normal4": [
        "curand_normal4"
    ],
    "hiprand_normal4_double": [
        "curand_normal4_double"
    ],
    "hiprand_normal_double": [
        "curand_normal_double"
    ],
    "hiprand_poisson": [
        "curand_poisson"
    ],
    "hiprand_poisson4": [
        "curand_poisson4"
    ],
    "hiprand_uniform": [
        "curand_uniform"
    ],
    "hiprand_uniform2_double": [
        "curand_uniform2_double"
    ],
    "hiprand_uniform4": [
        "curand_uniform4"
    ],
    "hiprand_uniform4_double": [
        "curand_uniform4_double"
    ],
    "hiprand_uniform_double": [
        "curand_uniform_double"
    ],
    "__half": [
        "__half"
    ],
    "__half2": [
        "__half2"
    ],
    "__half2_raw": [
        "__half2_raw"
    ],
    "__half_raw": [
        "__half_raw"
    ],
    "HIPContext": [
        "CUDAContext"
    ],
    "HIP_ARRAY3D_DESCRIPTOR": [
        "CUDA_ARRAY3D_DESCRIPTOR",
        "CUDA_ARRAY3D_DESCRIPTOR_st",
        "CUDA_ARRAY3D_DESCRIPTOR_v2"
    ],
    "HIP_ARRAY_DESCRIPTOR": [
        "CUDA_ARRAY_DESCRIPTOR",
        "CUDA_ARRAY_DESCRIPTOR_st",
        "CUDA_ARRAY_DESCRIPTOR_v1",
        "CUDA_ARRAY_DESCRIPTOR_v1_st",
        "CUDA_ARRAY_DESCRIPTOR_v2"
    ],
    "hipExternalMemoryBufferDesc": [
        "CUDA_EXTERNAL_MEMORY_BUFFER_DESC",
        "CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1",
        "cudaExternalMemoryBufferDesc"
    ],
    "hipExternalMemoryBufferDesc_st": [
        "CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st"
    ],
    "hipExternalMemoryHandleDesc": [
        "CUDA_EXTERNAL_MEMORY_HANDLE_DESC",
        "CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1",
        "cudaExternalMemoryHandleDesc"
    ],
    "hipExternalMemoryHandleDesc_st": [
        "CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st"
    ],
    "hipExternalSemaphoreHandleDesc": [
        "CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC",
        "CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1",
        "cudaExternalSemaphoreHandleDesc"
    ],
    "hipExternalSemaphoreHandleDesc_st": [
        "CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st"
    ],
    "hipExternalSemaphoreSignalParams": [
        "CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS",
        "CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1",
        "cudaExternalSemaphoreSignalParams",
        "cudaExternalSemaphoreSignalParams_v1"
    ],
    "hipExternalSemaphoreSignalParams_st": [
        "CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st"
    ],
    "hipExternalSemaphoreWaitParams": [
        "CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS",
        "CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1",
        "cudaExternalSemaphoreWaitParams",
        "cudaExternalSemaphoreWaitParams_v1"
    ],
    "hipExternalSemaphoreWaitParams_st": [
        "CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st"
    ],
    "hipHostNodeParams": [
        "CUDA_HOST_NODE_PARAMS",
        "CUDA_HOST_NODE_PARAMS_st",
        "CUDA_HOST_NODE_PARAMS_v1",
        "cudaHostNodeParams"
    ],
    "hipKernelNodeParams": [
        "CUDA_KERNEL_NODE_PARAMS",
        "CUDA_KERNEL_NODE_PARAMS_st",
        "CUDA_KERNEL_NODE_PARAMS_v1",
        "cudaKernelNodeParams"
    ],
    "hip_Memcpy2D": [
        "CUDA_MEMCPY2D",
        "CUDA_MEMCPY2D_st",
        "CUDA_MEMCPY2D_v1",
        "CUDA_MEMCPY2D_v1_st",
        "CUDA_MEMCPY2D_v2"
    ],
    "HIP_MEMCPY3D": [
        "CUDA_MEMCPY3D",
        "CUDA_MEMCPY3D_st",
        "CUDA_MEMCPY3D_v1",
        "CUDA_MEMCPY3D_v1_st",
        "CUDA_MEMCPY3D_v2"
    ],
    "hipMemsetParams": [
        "CUDA_MEMSET_NODE_PARAMS",
        "CUDA_MEMSET_NODE_PARAMS_st",
        "CUDA_MEMSET_NODE_PARAMS_v1",
        "cudaMemsetParams"
    ],
    "HIP_RESOURCE_DESC": [
        "CUDA_RESOURCE_DESC",
        "CUDA_RESOURCE_DESC_v1"
    ],
    "HIP_RESOURCE_DESC_st": [
        "CUDA_RESOURCE_DESC_st"
    ],
    "HIP_RESOURCE_VIEW_DESC": [
        "CUDA_RESOURCE_VIEW_DESC",
        "CUDA_RESOURCE_VIEW_DESC_v1"
    ],
    "HIP_RESOURCE_VIEW_DESC_st": [
        "CUDA_RESOURCE_VIEW_DESC_st"
    ],
    "HIP_TEXTURE_DESC": [
        "CUDA_TEXTURE_DESC",
        "CUDA_TEXTURE_DESC_v1"
    ],
    "HIP_TEXTURE_DESC_st": [
        "CUDA_TEXTURE_DESC_st"
    ],
    "hipGLDeviceList": [
        "CUGLDeviceList",
        "CUGLDeviceList_enum",
        "cudaGLDeviceList"
    ],
    "hipAccessPolicyWindow": [
        "CUaccessPolicyWindow",
        "CUaccessPolicyWindow_st",
        "cudaAccessPolicyWindow"
    ],
    "hipAccessProperty": [
        "CUaccessProperty",
        "CUaccessProperty_enum",
        "cudaAccessProperty"
    ],
    "HIPaddress_mode": [
        "CUaddress_mode"
    ],
    "HIPaddress_mode_enum": [
        "CUaddress_mode_enum"
    ],
    "hipArray_t": [
        "CUarray",
        "cudaArray_t"
    ],
    "hipArrayMapInfo": [
        "CUarrayMapInfo",
        "CUarrayMapInfo_st",
        "CUarrayMapInfo_v1"
    ],
    "hipArraySparseSubresourceType": [
        "CUarraySparseSubresourceType",
        "CUarraySparseSubresourceType_enum"
    ],
    "hipArray_Format": [
        "CUarray_format",
        "CUarray_format_enum"
    ],
    "hipArray": [
        "CUarray_st",
        "cudaArray"
    ],
    "hipComputeMode": [
        "CUcomputemode",
        "CUcomputemode_enum",
        "cudaComputeMode"
    ],
    "hipCtx_t": [
        "CUcontext"
    ],
    "ihipCtx_t": [
        "CUctx_st"
    ],
    "hipDevice_t": [
        "CUdevice",
        "CUdevice_v1"
    ],
    "hipDeviceP2PAttr": [
        "CUdevice_P2PAttribute",
        "CUdevice_P2PAttribute_enum",
        "cudaDeviceP2PAttr"
    ],
    "hipDeviceAttribute_t": [
        "CUdevice_attribute",
        "CUdevice_attribute_enum",
        "cudaDeviceAttr"
    ],
    "hipDeviceptr_t": [
        "CUdeviceptr",
        "CUdeviceptr_v1",
        "CUdeviceptr_v2"
    ],
    "hipEvent_t": [
        "CUevent",
        "cudaEvent_t"
    ],
    "ihipEvent_t": [
        "CUevent_st"
    ],
    "hipExternalMemory_t": [
        "CUexternalMemory",
        "cudaExternalMemory_t"
    ],
    "hipExternalMemoryHandleType": [
        "CUexternalMemoryHandleType",
        "cudaExternalMemoryHandleType"
    ],
    "hipExternalMemoryHandleType_enum": [
        "CUexternalMemoryHandleType_enum"
    ],
    "hipExternalSemaphore_t": [
        "CUexternalSemaphore",
        "cudaExternalSemaphore_t"
    ],
    "hipExternalSemaphoreHandleType": [
        "CUexternalSemaphoreHandleType",
        "cudaExternalSemaphoreHandleType"
    ],
    "hipExternalSemaphoreHandleType_enum": [
        "CUexternalSemaphoreHandleType_enum"
    ],
    "HIPfilter_mode": [
        "CUfilter_mode"
    ],
    "HIPfilter_mode_enum": [
        "CUfilter_mode_enum"
    ],
    "hipFuncCache_t": [
        "CUfunc_cache",
        "CUfunc_cache_enum",
        "cudaFuncCache"
    ],
    "ihipModuleSymbol_t": [
        "CUfunc_st"
    ],
    "hipFunction_t": [
        "CUfunction",
        "cudaFunction_t"
    ],
    "hipFunction_attribute": [
        "CUfunction_attribute",
        "CUfunction_attribute_enum"
    ],
    "hipGraph_t": [
        "CUgraph",
        "cudaGraph_t"
    ],
    "hipGraphExec_t": [
        "CUgraphExec",
        "cudaGraphExec_t"
    ],
    "hipGraphExecUpdateResult": [
        "CUgraphExecUpdateResult",
        "CUgraphExecUpdateResult_enum",
        "cudaGraphExecUpdateResult"
    ],
    "hipGraphExec": [
        "CUgraphExec_st"
    ],
    "hipGraphInstantiateFlags": [
        "CUgraphInstantiate_flags",
        "CUgraphInstantiate_flags_enum",
        "cudaGraphInstantiateFlags"
    ],
    "hipGraphMemAttributeType": [
        "CUgraphMem_attribute",
        "CUgraphMem_attribute_enum",
        "cudaGraphMemAttributeType"
    ],
    "hipGraphNode_t": [
        "CUgraphNode",
        "cudaGraphNode_t"
    ],
    "hipGraphNodeType": [
        "CUgraphNodeType",
        "CUgraphNodeType_enum",
        "cudaGraphNodeType"
    ],
    "hipGraphNode": [
        "CUgraphNode_st"
    ],
    "ihipGraph": [
        "CUgraph_st"
    ],
    "hipGraphicsRegisterFlags": [
        "CUgraphicsRegisterFlags",
        "CUgraphicsRegisterFlags_enum",
        "cudaGraphicsRegisterFlags"
    ],
    "hipGraphicsResource_t": [
        "CUgraphicsResource",
        "cudaGraphicsResource_t"
    ],
    "hipGraphicsResource": [
        "CUgraphicsResource_st",
        "cudaGraphicsResource"
    ],
    "hipHostFn_t": [
        "CUhostFn",
        "cudaHostFn_t"
    ],
    "hipIpcEventHandle_t": [
        "CUipcEventHandle",
        "CUipcEventHandle_v1",
        "cudaIpcEventHandle_t"
    ],
    "hipIpcEventHandle_st": [
        "CUipcEventHandle_st",
        "cudaIpcEventHandle_st"
    ],
    "hipIpcMemHandle_t": [
        "CUipcMemHandle",
        "CUipcMemHandle_v1",
        "cudaIpcMemHandle_t"
    ],
    "hipIpcMemHandle_st": [
        "CUipcMemHandle_st",
        "cudaIpcMemHandle_st"
    ],
    "hiprtcJITInputType": [
        "CUjitInputType",
        "CUjitInputType_enum"
    ],
    "hipJitOption": [
        "CUjit_option",
        "CUjit_option_enum"
    ],
    "hipKernelNodeAttrID": [
        "CUkernelNodeAttrID",
        "CUkernelNodeAttrID_enum",
        "cudaKernelNodeAttrID"
    ],
    "hipKernelNodeAttrValue": [
        "CUkernelNodeAttrValue",
        "CUkernelNodeAttrValue_union",
        "CUkernelNodeAttrValue_v1",
        "cudaKernelNodeAttrValue"
    ],
    "hipLimit_t": [
        "CUlimit",
        "CUlimit_enum",
        "cudaLimit"
    ],
    "hiprtcLinkState": [
        "CUlinkState"
    ],
    "ihiprtcLinkState": [
        "CUlinkState_st"
    ],
    "hipMemAccessDesc": [
        "CUmemAccessDesc",
        "CUmemAccessDesc_st",
        "CUmemAccessDesc_v1",
        "cudaMemAccessDesc"
    ],
    "hipMemAccessFlags": [
        "CUmemAccess_flags",
        "CUmemAccess_flags_enum",
        "cudaMemAccessFlags"
    ],
    "hipMemAllocationGranularity_flags": [
        "CUmemAllocationGranularity_flags",
        "CUmemAllocationGranularity_flags_enum"
    ],
    "hipMemAllocationHandleType": [
        "CUmemAllocationHandleType",
        "CUmemAllocationHandleType_enum",
        "cudaMemAllocationHandleType"
    ],
    "hipMemAllocationProp": [
        "CUmemAllocationProp",
        "CUmemAllocationProp_st",
        "CUmemAllocationProp_v1"
    ],
    "hipMemAllocationType": [
        "CUmemAllocationType",
        "CUmemAllocationType_enum",
        "cudaMemAllocationType"
    ],
    "hipMemGenericAllocationHandle_t": [
        "CUmemGenericAllocationHandle",
        "CUmemGenericAllocationHandle_v1"
    ],
    "hipMemHandleType": [
        "CUmemHandleType",
        "CUmemHandleType_enum"
    ],
    "hipMemLocation": [
        "CUmemLocation",
        "CUmemLocation_st",
        "CUmemLocation_v1",
        "cudaMemLocation"
    ],
    "hipMemLocationType": [
        "CUmemLocationType",
        "CUmemLocationType_enum",
        "cudaMemLocationType"
    ],
    "hipMemOperationType": [
        "CUmemOperationType",
        "CUmemOperationType_enum"
    ],
    "ihipMemPoolHandle_t": [
        "CUmemPoolHandle_st"
    ],
    "hipMemPoolProps": [
        "CUmemPoolProps",
        "CUmemPoolProps_st",
        "CUmemPoolProps_v1",
        "cudaMemPoolProps"
    ],
    "hipMemPoolPtrExportData": [
        "CUmemPoolPtrExportData",
        "CUmemPoolPtrExportData_st",
        "CUmemPoolPtrExportData_v1",
        "cudaMemPoolPtrExportData"
    ],
    "hipMemPoolAttr": [
        "CUmemPool_attribute",
        "CUmemPool_attribute_enum",
        "cudaMemPoolAttr"
    ],
    "hipMemoryAdvise": [
        "CUmem_advise",
        "CUmem_advise_enum",
        "cudaMemoryAdvise"
    ],
    "hipMemRangeAttribute": [
        "CUmem_range_attribute",
        "CUmem_range_attribute_enum",
        "cudaMemRangeAttribute"
    ],
    "hipMemPool_t": [
        "CUmemoryPool",
        "cudaMemPool_t"
    ],
    "hipMemoryType": [
        "CUmemorytype",
        "CUmemorytype_enum",
        "cudaMemoryType"
    ],
    "hipMipmappedArray_t": [
        "CUmipmappedArray",
        "cudaMipmappedArray_t"
    ],
    "hipMipmappedArray": [
        "CUmipmappedArray_st",
        "cudaMipmappedArray"
    ],
    "ihipModule_t": [
        "CUmod_st"
    ],
    "hipModule_t": [
        "CUmodule"
    ],
    "hipPointer_attribute": [
        "CUpointer_attribute",
        "CUpointer_attribute_enum"
    ],
    "HIPresourceViewFormat": [
        "CUresourceViewFormat"
    ],
    "HIPresourceViewFormat_enum": [
        "CUresourceViewFormat_enum"
    ],
    "HIPresourcetype": [
        "CUresourcetype"
    ],
    "HIPresourcetype_enum": [
        "CUresourcetype_enum"
    ],
    "hipError_t": [
        "CUresult",
        "cudaError",
        "cudaError_enum",
        "cudaError_t"
    ],
    "hipSharedMemConfig": [
        "CUsharedconfig",
        "CUsharedconfig_enum",
        "cudaSharedMemConfig"
    ],
    "hipStream_t": [
        "CUstream",
        "cudaStream_t"
    ],
    "hipStreamCallback_t": [
        "CUstreamCallback",
        "cudaStreamCallback_t"
    ],
    "hipStreamCaptureMode": [
        "CUstreamCaptureMode",
        "CUstreamCaptureMode_enum",
        "cudaStreamCaptureMode"
    ],
    "hipStreamCaptureStatus": [
        "CUstreamCaptureStatus",
        "CUstreamCaptureStatus_enum",
        "cudaStreamCaptureStatus"
    ],
    "hipStreamUpdateCaptureDependenciesFlags": [
        "CUstreamUpdateCaptureDependencies_flags",
        "CUstreamUpdateCaptureDependencies_flags_enum",
        "cudaStreamUpdateCaptureDependenciesFlags"
    ],
    "ihipStream_t": [
        "CUstream_st"
    ],
    "hipSurfaceObject_t": [
        "CUsurfObject",
        "CUsurfObject_v1",
        "cudaSurfaceObject_t"
    ],
    "hipTextureObject_t": [
        "CUtexObject",
        "CUtexObject_v1",
        "cudaTextureObject_t"
    ],
    "hipTexRef": [
        "CUtexref"
    ],
    "textureReference": [
        "CUtexref_st",
        "textureReference"
    ],
    "hipUserObject_t": [
        "CUuserObject",
        "cudaUserObject_t"
    ],
    "hipUserObjectRetainFlags": [
        "CUuserObjectRetain_flags",
        "CUuserObjectRetain_flags_enum",
        "cudaUserObjectRetainFlags"
    ],
    "hipUserObjectFlags": [
        "CUuserObject_flags",
        "CUuserObject_flags_enum",
        "cudaUserObjectFlags"
    ],
    "hipUserObject": [
        "CUuserObject_st"
    ],
    "hipUUID": [
        "CUuuid",
        "cudaUUID_t"
    ],
    "hipUUID_t": [
        "CUuuid_st"
    ],
    "GLenum": [
        "GLenum"
    ],
    "GLuint": [
        "GLuint"
    ],
    "bsric02Info_t": [
        "bsric02Info_t"
    ],
    "bsrilu02Info_t": [
        "bsrilu02Info_t"
    ],
    "bsrsm2Info": [
        "bsrsm2Info"
    ],
    "bsrsm2Info_t": [
        "bsrsm2Info_t"
    ],
    "bsrsv2Info_t": [
        "bsrsv2Info_t"
    ],
    "csrgemm2Info_t": [
        "csrgemm2Info_t"
    ],
    "csrilu02Info_t": [
        "csrilu02Info_t"
    ],
    "csrsm2Info_t": [
        "csrsm2Info_t"
    ],
    "csrsv2Info_t": [
        "csrsv2Info_t"
    ],
    "csru2csrInfo": [
        "csru2csrInfo"
    ],
    "csru2csrInfo_t": [
        "csru2csrInfo_t"
    ],
    "hipComplex": [
        "cuComplex"
    ],
    "hipDoubleComplex": [
        "cuDoubleComplex"
    ],
    "hipFloatComplex": [
        "cuFloatComplex"
    ],
    "hipblasAtomicsMode_t": [
        "cublasAtomicsMode_t"
    ],
    "hipblasDatatype_t": [
        "cublasComputeType_t",
        "cublasDataType_t",
        "cudaDataType",
        "cudaDataType_t"
    ],
    "hipblasDiagType_t": [
        "cublasDiagType_t"
    ],
    "hipblasFillMode_t": [
        "cublasFillMode_t"
    ],
    "hipblasGemmAlgo_t": [
        "cublasGemmAlgo_t"
    ],
    "hipblasHandle_t": [
        "cublasHandle_t"
    ],
    "hipblasOperation_t": [
        "cublasOperation_t"
    ],
    "hipblasPointerMode_t": [
        "cublasPointerMode_t"
    ],
    "hipblasSideMode_t": [
        "cublasSideMode_t"
    ],
    "hipblasStatus_t": [
        "cublasStatus",
        "cublasStatus_t"
    ],
    "hipArray_const_t": [
        "cudaArray_const_t"
    ],
    "hipChannelFormatDesc": [
        "cudaChannelFormatDesc"
    ],
    "hipChannelFormatKind": [
        "cudaChannelFormatKind"
    ],
    "hipDeviceProp_t": [
        "cudaDeviceProp"
    ],
    "hipExtent": [
        "cudaExtent"
    ],
    "hipFuncAttribute": [
        "cudaFuncAttribute"
    ],
    "hipFuncAttributes": [
        "cudaFuncAttributes"
    ],
    "hipLaunchParams": [
        "cudaLaunchParams"
    ],
    "hipMemcpy3DParms": [
        "cudaMemcpy3DParms"
    ],
    "hipMemcpyKind": [
        "cudaMemcpyKind"
    ],
    "hipMipmappedArray_const_t": [
        "cudaMipmappedArray_const_t"
    ],
    "hipPitchedPtr": [
        "cudaPitchedPtr"
    ],
    "hipPointerAttribute_t": [
        "cudaPointerAttributes"
    ],
    "hipPos": [
        "cudaPos"
    ],
    "hipResourceDesc": [
        "cudaResourceDesc"
    ],
    "hipResourceType": [
        "cudaResourceType"
    ],
    "hipResourceViewDesc": [
        "cudaResourceViewDesc"
    ],
    "hipResourceViewFormat": [
        "cudaResourceViewFormat"
    ],
    "hipSurfaceBoundaryMode": [
        "cudaSurfaceBoundaryMode"
    ],
    "hipTextureAddressMode": [
        "cudaTextureAddressMode"
    ],
    "hipTextureDesc": [
        "cudaTextureDesc"
    ],
    "hipTextureFilterMode": [
        "cudaTextureFilterMode"
    ],
    "hipTextureReadMode": [
        "cudaTextureReadMode"
    ],
    "hipdnnActivationDescriptor_t": [
        "cudnnActivationDescriptor_t"
    ],
    "hipdnnActivationMode_t": [
        "cudnnActivationMode_t"
    ],
    "hipdnnBatchNormMode_t": [
        "cudnnBatchNormMode_t"
    ],
    "hipdnnConvolutionBwdDataAlgoPerf_t": [
        "cudnnConvolutionBwdDataAlgoPerfStruct",
        "cudnnConvolutionBwdDataAlgoPerf_t"
    ],
    "hipdnnConvolutionBwdDataAlgo_t": [
        "cudnnConvolutionBwdDataAlgo_t"
    ],
    "hipdnnConvolutionBwdDataPreference_t": [
        "cudnnConvolutionBwdDataPreference_t"
    ],
    "hipdnnConvolutionBwdFilterAlgoPerf_t": [
        "cudnnConvolutionBwdFilterAlgoPerfStruct",
        "cudnnConvolutionBwdFilterAlgoPerf_t"
    ],
    "hipdnnConvolutionBwdFilterAlgo_t": [
        "cudnnConvolutionBwdFilterAlgo_t"
    ],
    "hipdnnConvolutionBwdFilterPreference_t": [
        "cudnnConvolutionBwdFilterPreference_t"
    ],
    "hipdnnConvolutionDescriptor_t": [
        "cudnnConvolutionDescriptor_t"
    ],
    "hipdnnConvolutionFwdAlgoPerf_t": [
        "cudnnConvolutionFwdAlgoPerfStruct",
        "cudnnConvolutionFwdAlgoPerf_t"
    ],
    "hipdnnConvolutionFwdAlgo_t": [
        "cudnnConvolutionFwdAlgo_t"
    ],
    "hipdnnConvolutionFwdPreference_t": [
        "cudnnConvolutionFwdPreference_t"
    ],
    "hipdnnConvolutionMode_t": [
        "cudnnConvolutionMode_t"
    ],
    "hipdnnDataType_t": [
        "cudnnDataType_t"
    ],
    "hipdnnDirectionMode_t": [
        "cudnnDirectionMode_t"
    ],
    "hipdnnDropoutDescriptor_t": [
        "cudnnDropoutDescriptor_t"
    ],
    "hipdnnFilterDescriptor_t": [
        "cudnnFilterDescriptor_t"
    ],
    "hipdnnHandle_t": [
        "cudnnHandle_t"
    ],
    "hipdnnIndicesType_t": [
        "cudnnIndicesType_t"
    ],
    "hipdnnLRNDescriptor_t": [
        "cudnnLRNDescriptor_t"
    ],
    "hipdnnLRNMode_t": [
        "cudnnLRNMode_t"
    ],
    "hipdnnMathType_t": [
        "cudnnMathType_t"
    ],
    "hipdnnNanPropagation_t": [
        "cudnnNanPropagation_t"
    ],
    "hipdnnOpTensorDescriptor_t": [
        "cudnnOpTensorDescriptor_t"
    ],
    "hipdnnOpTensorOp_t": [
        "cudnnOpTensorOp_t"
    ],
    "hipdnnPersistentRNNPlan_t": [
        "cudnnPersistentRNNPlan_t"
    ],
    "hipdnnPoolingDescriptor_t": [
        "cudnnPoolingDescriptor_t"
    ],
    "hipdnnPoolingMode_t": [
        "cudnnPoolingMode_t"
    ],
    "hipdnnRNNAlgo_t": [
        "cudnnRNNAlgo_t"
    ],
    "hipdnnRNNBiasMode_t": [
        "cudnnRNNBiasMode_t"
    ],
    "hipdnnRNNDescriptor_t": [
        "cudnnRNNDescriptor_t"
    ],
    "hipdnnRNNInputMode_t": [
        "cudnnRNNInputMode_t"
    ],
    "hipdnnRNNMode_t": [
        "cudnnRNNMode_t"
    ],
    "hipdnnReduceTensorDescriptor_t": [
        "cudnnReduceTensorDescriptor_t"
    ],
    "hipdnnReduceTensorIndices_t": [
        "cudnnReduceTensorIndices_t"
    ],
    "hipdnnReduceTensorOp_t": [
        "cudnnReduceTensorOp_t"
    ],
    "hipdnnSoftmaxAlgorithm_t": [
        "cudnnSoftmaxAlgorithm_t"
    ],
    "hipdnnSoftmaxMode_t": [
        "cudnnSoftmaxMode_t"
    ],
    "hipdnnStatus_t": [
        "cudnnStatus_t"
    ],
    "hipdnnTensorDescriptor_t": [
        "cudnnTensorDescriptor_t"
    ],
    "hipdnnTensorFormat_t": [
        "cudnnTensorFormat_t"
    ],
    "hipfftComplex": [
        "cufftComplex"
    ],
    "hipfftDoubleComplex": [
        "cufftDoubleComplex"
    ],
    "hipfftDoubleReal": [
        "cufftDoubleReal"
    ],
    "hipfftHandle": [
        "cufftHandle"
    ],
    "hipfftReal": [
        "cufftReal"
    ],
    "hipfftResult": [
        "cufftResult"
    ],
    "hipfftResult_t": [
        "cufftResult_t"
    ],
    "hipfftType": [
        "cufftType"
    ],
    "hipfftType_t": [
        "cufftType_t"
    ],
    "hipfftXtCallbackType": [
        "cufftXtCallbackType"
    ],
    "hipfftXtCallbackType_t": [
        "cufftXtCallbackType_t"
    ],
    "hiprandDirectionVectors32_t": [
        "curandDirectionVectors32_t"
    ],
    "hiprandDiscreteDistribution_st": [
        "curandDiscreteDistribution_st"
    ],
    "hiprandDiscreteDistribution_t": [
        "curandDiscreteDistribution_t"
    ],
    "hiprandGenerator_st": [
        "curandGenerator_st"
    ],
    "hiprandGenerator_t": [
        "curandGenerator_t"
    ],
    "hiprandRngType_t": [
        "curandRngType",
        "curandRngType_t"
    ],
    "hiprandState": [
        "curandState"
    ],
    "hiprandStateMRG32k3a": [
        "curandStateMRG32k3a"
    ],
    "hiprandStateMRG32k3a_t": [
        "curandStateMRG32k3a_t"
    ],
    "hiprandStateMtgp32": [
        "curandStateMtgp32"
    ],
    "hiprandStateMtgp32_t": [
        "curandStateMtgp32_t"
    ],
    "hiprandStatePhilox4_32_10": [
        "curandStatePhilox4_32_10"
    ],
    "hiprandStatePhilox4_32_10_t": [
        "curandStatePhilox4_32_10_t"
    ],
    "hiprandStateSobol32": [
        "curandStateSobol32"
    ],
    "hiprandStateSobol32_t": [
        "curandStateSobol32_t"
    ],
    "hiprandStateXORWOW": [
        "curandStateXORWOW"
    ],
    "hiprandStateXORWOW_t": [
        "curandStateXORWOW_t"
    ],
    "hiprandState_t": [
        "curandState_t"
    ],
    "hiprandStatus_t": [
        "curandStatus",
        "curandStatus_t"
    ],
    "hipsparseAction_t": [
        "cusparseAction_t"
    ],
    "hipsparseColorInfo_t": [
        "cusparseColorInfo_t"
    ],
    "hipsparseDiagType_t": [
        "cusparseDiagType_t"
    ],
    "hipsparseDirection_t": [
        "cusparseDirection_t"
    ],
    "hipsparseDnMatDescr": [
        "cusparseDnMatDescr"
    ],
    "hipsparseDnMatDescr_t": [
        "cusparseDnMatDescr_t"
    ],
    "hipsparseDnVecDescr_t": [
        "cusparseDnVecDescr_t"
    ],
    "hipsparseFillMode_t": [
        "cusparseFillMode_t"
    ],
    "hipsparseFormat_t": [
        "cusparseFormat_t"
    ],
    "hipsparseHandle_t": [
        "cusparseHandle_t"
    ],
    "hipsparseHybMat_t": [
        "cusparseHybMat_t"
    ],
    "hipsparseHybPartition_t": [
        "cusparseHybPartition_t"
    ],
    "hipsparseIndexBase_t": [
        "cusparseIndexBase_t"
    ],
    "hipsparseIndexType_t": [
        "cusparseIndexType_t"
    ],
    "hipsparseMatDescr_t": [
        "cusparseMatDescr_t"
    ],
    "hipsparseMatrixType_t": [
        "cusparseMatrixType_t"
    ],
    "hipsparseOperation_t": [
        "cusparseOperation_t"
    ],
    "hipsparseOrder_t": [
        "cusparseOrder_t"
    ],
    "hipsparsePointerMode_t": [
        "cusparsePointerMode_t"
    ],
    "hipsparseSDDMMAlg_t": [
        "cusparseSDDMMAlg_t"
    ],
    "hipsparseSolvePolicy_t": [
        "cusparseSolvePolicy_t"
    ],
    "hipsparseSpGEMMAlg_t": [
        "cusparseSpGEMMAlg_t"
    ],
    "hipsparseSpGEMMDescr": [
        "cusparseSpGEMMDescr"
    ],
    "hipsparseSpGEMMDescr_t": [
        "cusparseSpGEMMDescr_t"
    ],
    "hipsparseSpMMAlg_t": [
        "cusparseSpMMAlg_t"
    ],
    "hipsparseSpMVAlg_t": [
        "cusparseSpMVAlg_t"
    ],
    "hipsparseSpMatAttribute_t": [
        "cusparseSpMatAttribute_t"
    ],
    "hipsparseSpMatDescr_t": [
        "cusparseSpMatDescr_t"
    ],
    "hipsparseSpSMAlg_t": [
        "cusparseSpSMAlg_t"
    ],
    "hipsparseSpSMDescr": [
        "cusparseSpSMDescr"
    ],
    "hipsparseSpSMDescr_t": [
        "cusparseSpSMDescr_t"
    ],
    "hipsparseSpSVAlg_t": [
        "cusparseSpSVAlg_t"
    ],
    "hipsparseSpSVDescr": [
        "cusparseSpSVDescr"
    ],
    "hipsparseSpSVDescr_t": [
        "cusparseSpSVDescr_t"
    ],
    "hipsparseSpVecDescr_t": [
        "cusparseSpVecDescr_t"
    ],
    "hipsparseSparseToDenseAlg_t": [
        "cusparseSparseToDenseAlg_t"
    ],
    "hipsparseStatus_t": [
        "cusparseStatus_t"
    ],
    "hiprtcProgram": [
        "nvrtcProgram"
    ],
    "hiprtcResult": [
        "nvrtcResult"
    ],
    "pruneInfo_t": [
        "pruneInfo_t"
    ],
    "surfaceReference": [
        "surfaceReference"
    ],
    "HIPBLAS_ATOMICS_ALLOWED": [
        "CUBLAS_ATOMICS_ALLOWED"
    ],
    "HIPBLAS_ATOMICS_NOT_ALLOWED": [
        "CUBLAS_ATOMICS_NOT_ALLOWED"
    ],
    "HIPBLAS_DIAG_NON_UNIT": [
        "CUBLAS_DIAG_NON_UNIT"
    ],
    "HIPBLAS_DIAG_UNIT": [
        "CUBLAS_DIAG_UNIT"
    ],
    "HIPBLAS_FILL_MODE_FULL": [
        "CUBLAS_FILL_MODE_FULL"
    ],
    "HIPBLAS_FILL_MODE_LOWER": [
        "CUBLAS_FILL_MODE_LOWER"
    ],
    "HIPBLAS_FILL_MODE_UPPER": [
        "CUBLAS_FILL_MODE_UPPER"
    ],
    "HIPBLAS_GEMM_DEFAULT": [
        "CUBLAS_GEMM_DEFAULT",
        "CUBLAS_GEMM_DFALT"
    ],
    "HIPBLAS_OP_C": [
        "CUBLAS_OP_C",
        "CUBLAS_OP_HERMITAN"
    ],
    "HIPBLAS_OP_N": [
        "CUBLAS_OP_N"
    ],
    "HIPBLAS_OP_T": [
        "CUBLAS_OP_T"
    ],
    "HIPBLAS_POINTER_MODE_DEVICE": [
        "CUBLAS_POINTER_MODE_DEVICE"
    ],
    "HIPBLAS_POINTER_MODE_HOST": [
        "CUBLAS_POINTER_MODE_HOST"
    ],
    "HIPBLAS_SIDE_LEFT": [
        "CUBLAS_SIDE_LEFT"
    ],
    "HIPBLAS_SIDE_RIGHT": [
        "CUBLAS_SIDE_RIGHT"
    ],
    "HIPBLAS_STATUS_ALLOC_FAILED": [
        "CUBLAS_STATUS_ALLOC_FAILED"
    ],
    "HIPBLAS_STATUS_ARCH_MISMATCH": [
        "CUBLAS_STATUS_ARCH_MISMATCH"
    ],
    "HIPBLAS_STATUS_EXECUTION_FAILED": [
        "CUBLAS_STATUS_EXECUTION_FAILED"
    ],
    "HIPBLAS_STATUS_INTERNAL_ERROR": [
        "CUBLAS_STATUS_INTERNAL_ERROR"
    ],
    "HIPBLAS_STATUS_INVALID_VALUE": [
        "CUBLAS_STATUS_INVALID_VALUE"
    ],
    "HIPBLAS_STATUS_UNKNOWN": [
        "CUBLAS_STATUS_LICENSE_ERROR"
    ],
    "HIPBLAS_STATUS_MAPPING_ERROR": [
        "CUBLAS_STATUS_MAPPING_ERROR"
    ],
    "HIPBLAS_STATUS_NOT_INITIALIZED": [
        "CUBLAS_STATUS_NOT_INITIALIZED"
    ],
    "HIPBLAS_STATUS_NOT_SUPPORTED": [
        "CUBLAS_STATUS_NOT_SUPPORTED"
    ],
    "HIPBLAS_STATUS_SUCCESS": [
        "CUBLAS_STATUS_SUCCESS"
    ],
    "HIPBLAS_C_16B": [
        "CUDA_C_16BF"
    ],
    "HIPBLAS_C_16F": [
        "CUDA_C_16F"
    ],
    "HIPBLAS_C_32F": [
        "CUDA_C_32F"
    ],
    "HIPBLAS_C_32I": [
        "CUDA_C_32I"
    ],
    "HIPBLAS_C_32U": [
        "CUDA_C_32U"
    ],
    "HIPBLAS_C_64F": [
        "CUDA_C_64F"
    ],
    "HIPBLAS_C_8I": [
        "CUDA_C_8I"
    ],
    "HIPBLAS_C_8U": [
        "CUDA_C_8U"
    ],
    "hipErrorAlreadyAcquired": [
        "CUDA_ERROR_ALREADY_ACQUIRED",
        "cudaErrorAlreadyAcquired"
    ],
    "hipErrorAlreadyMapped": [
        "CUDA_ERROR_ALREADY_MAPPED",
        "cudaErrorAlreadyMapped"
    ],
    "hipErrorArrayIsMapped": [
        "CUDA_ERROR_ARRAY_IS_MAPPED",
        "cudaErrorArrayIsMapped"
    ],
    "hipErrorAssert": [
        "CUDA_ERROR_ASSERT",
        "cudaErrorAssert"
    ],
    "hipErrorCapturedEvent": [
        "CUDA_ERROR_CAPTURED_EVENT",
        "cudaErrorCapturedEvent"
    ],
    "hipErrorContextAlreadyCurrent": [
        "CUDA_ERROR_CONTEXT_ALREADY_CURRENT"
    ],
    "hipErrorContextAlreadyInUse": [
        "CUDA_ERROR_CONTEXT_ALREADY_IN_USE",
        "cudaErrorDeviceAlreadyInUse"
    ],
    "hipErrorContextIsDestroyed": [
        "CUDA_ERROR_CONTEXT_IS_DESTROYED",
        "cudaErrorContextIsDestroyed"
    ],
    "hipErrorCooperativeLaunchTooLarge": [
        "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE",
        "cudaErrorCooperativeLaunchTooLarge"
    ],
    "hipErrorDeinitialized": [
        "CUDA_ERROR_DEINITIALIZED",
        "cudaErrorCudartUnloading"
    ],
    "hipErrorECCNotCorrectable": [
        "CUDA_ERROR_ECC_UNCORRECTABLE",
        "cudaErrorECCUncorrectable"
    ],
    "hipErrorFileNotFound": [
        "CUDA_ERROR_FILE_NOT_FOUND",
        "cudaErrorFileNotFound"
    ],
    "hipErrorGraphExecUpdateFailure": [
        "CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE",
        "cudaErrorGraphExecUpdateFailure"
    ],
    "hipErrorHostMemoryAlreadyRegistered": [
        "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED",
        "cudaErrorHostMemoryAlreadyRegistered"
    ],
    "hipErrorHostMemoryNotRegistered": [
        "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED",
        "cudaErrorHostMemoryNotRegistered"
    ],
    "hipErrorIllegalAddress": [
        "CUDA_ERROR_ILLEGAL_ADDRESS",
        "cudaErrorIllegalAddress"
    ],
    "hipErrorIllegalState": [
        "CUDA_ERROR_ILLEGAL_STATE",
        "cudaErrorIllegalState"
    ],
    "hipErrorInvalidContext": [
        "CUDA_ERROR_INVALID_CONTEXT",
        "cudaErrorDeviceUninitialized"
    ],
    "hipErrorInvalidDevice": [
        "CUDA_ERROR_INVALID_DEVICE",
        "cudaErrorInvalidDevice"
    ],
    "hipErrorInvalidGraphicsContext": [
        "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT",
        "cudaErrorInvalidGraphicsContext"
    ],
    "hipErrorInvalidHandle": [
        "CUDA_ERROR_INVALID_HANDLE",
        "cudaErrorInvalidResourceHandle"
    ],
    "hipErrorInvalidImage": [
        "CUDA_ERROR_INVALID_IMAGE",
        "cudaErrorInvalidKernelImage"
    ],
    "hipErrorInvalidKernelFile": [
        "CUDA_ERROR_INVALID_PTX",
        "cudaErrorInvalidPtx"
    ],
    "hipErrorInvalidSource": [
        "CUDA_ERROR_INVALID_SOURCE",
        "cudaErrorInvalidSource"
    ],
    "hipErrorInvalidValue": [
        "CUDA_ERROR_INVALID_VALUE",
        "cudaErrorInvalidValue"
    ],
    "hipErrorLaunchFailure": [
        "CUDA_ERROR_LAUNCH_FAILED",
        "cudaErrorLaunchFailure"
    ],
    "hipErrorLaunchOutOfResources": [
        "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
        "cudaErrorLaunchOutOfResources"
    ],
    "hipErrorLaunchTimeOut": [
        "CUDA_ERROR_LAUNCH_TIMEOUT",
        "cudaErrorLaunchTimeout"
    ],
    "hipErrorMapFailed": [
        "CUDA_ERROR_MAP_FAILED",
        "cudaErrorMapBufferObjectFailed"
    ],
    "hipErrorNotFound": [
        "CUDA_ERROR_NOT_FOUND",
        "cudaErrorSymbolNotFound"
    ],
    "hipErrorNotInitialized": [
        "CUDA_ERROR_NOT_INITIALIZED",
        "cudaErrorInitializationError"
    ],
    "hipErrorNotMapped": [
        "CUDA_ERROR_NOT_MAPPED",
        "cudaErrorNotMapped"
    ],
    "hipErrorNotMappedAsArray": [
        "CUDA_ERROR_NOT_MAPPED_AS_ARRAY",
        "cudaErrorNotMappedAsArray"
    ],
    "hipErrorNotMappedAsPointer": [
        "CUDA_ERROR_NOT_MAPPED_AS_POINTER",
        "cudaErrorNotMappedAsPointer"
    ],
    "hipErrorNotReady": [
        "CUDA_ERROR_NOT_READY",
        "cudaErrorNotReady"
    ],
    "hipErrorNotSupported": [
        "CUDA_ERROR_NOT_SUPPORTED",
        "cudaErrorNotSupported"
    ],
    "hipErrorNoBinaryForGpu": [
        "CUDA_ERROR_NO_BINARY_FOR_GPU",
        "cudaErrorNoKernelImageForDevice"
    ],
    "hipErrorNoDevice": [
        "CUDA_ERROR_NO_DEVICE",
        "cudaErrorNoDevice"
    ],
    "hipErrorOperatingSystem": [
        "CUDA_ERROR_OPERATING_SYSTEM",
        "cudaErrorOperatingSystem"
    ],
    "hipErrorOutOfMemory": [
        "CUDA_ERROR_OUT_OF_MEMORY",
        "cudaErrorMemoryAllocation"
    ],
    "hipErrorPeerAccessAlreadyEnabled": [
        "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED",
        "cudaErrorPeerAccessAlreadyEnabled"
    ],
    "hipErrorPeerAccessNotEnabled": [
        "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED",
        "cudaErrorPeerAccessNotEnabled"
    ],
    "hipErrorPeerAccessUnsupported": [
        "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED",
        "cudaErrorPeerAccessUnsupported"
    ],
    "hipErrorSetOnActiveProcess": [
        "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE",
        "cudaErrorSetOnActiveProcess"
    ],
    "hipErrorProfilerAlreadyStarted": [
        "CUDA_ERROR_PROFILER_ALREADY_STARTED",
        "cudaErrorProfilerAlreadyStarted"
    ],
    "hipErrorProfilerAlreadyStopped": [
        "CUDA_ERROR_PROFILER_ALREADY_STOPPED",
        "cudaErrorProfilerAlreadyStopped"
    ],
    "hipErrorProfilerDisabled": [
        "CUDA_ERROR_PROFILER_DISABLED",
        "cudaErrorProfilerDisabled"
    ],
    "hipErrorProfilerNotInitialized": [
        "CUDA_ERROR_PROFILER_NOT_INITIALIZED",
        "cudaErrorProfilerNotInitialized"
    ],
    "hipErrorSharedObjectInitFailed": [
        "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED",
        "cudaErrorSharedObjectInitFailed"
    ],
    "hipErrorSharedObjectSymbolNotFound": [
        "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND",
        "cudaErrorSharedObjectSymbolNotFound"
    ],
    "hipErrorStreamCaptureImplicit": [
        "CUDA_ERROR_STREAM_CAPTURE_IMPLICIT",
        "cudaErrorStreamCaptureImplicit"
    ],
    "hipErrorStreamCaptureInvalidated": [
        "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED",
        "cudaErrorStreamCaptureInvalidated"
    ],
    "hipErrorStreamCaptureIsolation": [
        "CUDA_ERROR_STREAM_CAPTURE_ISOLATION",
        "cudaErrorStreamCaptureIsolation"
    ],
    "hipErrorStreamCaptureMerge": [
        "CUDA_ERROR_STREAM_CAPTURE_MERGE",
        "cudaErrorStreamCaptureMerge"
    ],
    "hipErrorStreamCaptureUnjoined": [
        "CUDA_ERROR_STREAM_CAPTURE_UNJOINED",
        "cudaErrorStreamCaptureUnjoined"
    ],
    "hipErrorStreamCaptureUnmatched": [
        "CUDA_ERROR_STREAM_CAPTURE_UNMATCHED",
        "cudaErrorStreamCaptureUnmatched"
    ],
    "hipErrorStreamCaptureUnsupported": [
        "CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED",
        "cudaErrorStreamCaptureUnsupported"
    ],
    "hipErrorStreamCaptureWrongThread": [
        "CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD",
        "cudaErrorStreamCaptureWrongThread"
    ],
    "hipErrorUnknown": [
        "CUDA_ERROR_UNKNOWN",
        "cudaErrorUnknown"
    ],
    "hipErrorUnmapFailed": [
        "CUDA_ERROR_UNMAP_FAILED",
        "cudaErrorUnmapBufferObjectFailed"
    ],
    "hipErrorUnsupportedLimit": [
        "CUDA_ERROR_UNSUPPORTED_LIMIT",
        "cudaErrorUnsupportedLimit"
    ],
    "hipGraphInstantiateFlagAutoFreeOnLaunch": [
        "CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH",
        "cudaGraphInstantiateFlagAutoFreeOnLaunch"
    ],
    "HIPBLAS_R_16B": [
        "CUDA_R_16BF"
    ],
    "HIPBLAS_R_16F": [
        "CUDA_R_16F"
    ],
    "HIPBLAS_R_32F": [
        "CUDA_R_32F"
    ],
    "HIPBLAS_R_32I": [
        "CUDA_R_32I"
    ],
    "HIPBLAS_R_32U": [
        "CUDA_R_32U"
    ],
    "HIPBLAS_R_64F": [
        "CUDA_R_64F"
    ],
    "HIPBLAS_R_8I": [
        "CUDA_R_8I"
    ],
    "HIPBLAS_R_8U": [
        "CUDA_R_8U"
    ],
    "hipSuccess": [
        "CUDA_SUCCESS",
        "cudaSuccess"
    ],
    "HIPDNN_16BIT_INDICES": [
        "CUDNN_16BIT_INDICES"
    ],
    "HIPDNN_32BIT_INDICES": [
        "CUDNN_32BIT_INDICES"
    ],
    "HIPDNN_64BIT_INDICES": [
        "CUDNN_64BIT_INDICES"
    ],
    "HIPDNN_8BIT_INDICES": [
        "CUDNN_8BIT_INDICES"
    ],
    "HIPDNN_ACTIVATION_CLIPPED_RELU": [
        "CUDNN_ACTIVATION_CLIPPED_RELU"
    ],
    "HIPDNN_ACTIVATION_ELU": [
        "CUDNN_ACTIVATION_ELU"
    ],
    "HIPDNN_ACTIVATION_PATHTRU": [
        "CUDNN_ACTIVATION_IDENTITY"
    ],
    "HIPDNN_ACTIVATION_RELU": [
        "CUDNN_ACTIVATION_RELU"
    ],
    "HIPDNN_ACTIVATION_SIGMOID": [
        "CUDNN_ACTIVATION_SIGMOID"
    ],
    "HIPDNN_ACTIVATION_SWISH": [
        "CUDNN_ACTIVATION_SWISH"
    ],
    "HIPDNN_ACTIVATION_TANH": [
        "CUDNN_ACTIVATION_TANH"
    ],
    "HIPDNN_BATCHNORM_PER_ACTIVATION": [
        "CUDNN_BATCHNORM_PER_ACTIVATION"
    ],
    "HIPDNN_BATCHNORM_SPATIAL": [
        "CUDNN_BATCHNORM_SPATIAL"
    ],
    "HIPDNN_BATCHNORM_SPATIAL_PERSISTENT": [
        "CUDNN_BATCHNORM_SPATIAL_PERSISTENT"
    ],
    "HIPDNN_BIDIRECTIONAL": [
        "CUDNN_BIDIRECTIONAL"
    ],
    "HIPDNN_BN_MIN_EPSILON": [
        "CUDNN_BN_MIN_EPSILON"
    ],
    "HIPDNN_CONVOLUTION": [
        "CUDNN_CONVOLUTION"
    ],
    "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_0": [
        "CUDNN_CONVOLUTION_BWD_DATA_ALGO_0"
    ],
    "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1": [
        "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1"
    ],
    "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_TRANSPOSE_GEMM": [
        "CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT"
    ],
    "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT": [
        "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT"
    ],
    "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING": [
        "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING"
    ],
    "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD": [
        "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD"
    ],
    "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED": [
        "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED"
    ],
    "HIPDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE": [
        "CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE"
    ],
    "HIPDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST": [
        "CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST"
    ],
    "HIPDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT": [
        "CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT"
    ],
    "HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0": [
        "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0"
    ],
    "HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1": [
        "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1"
    ],
    "HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_3": [
        "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3"
    ],
    "HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT": [
        "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT"
    ],
    "HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT": [
        "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT"
    ],
    "HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING": [
        "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING"
    ],
    "HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD": [
        "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD"
    ],
    "HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED": [
        "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED"
    ],
    "HIPDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE": [
        "CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE"
    ],
    "HIPDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST": [
        "CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST"
    ],
    "HIPDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT": [
        "CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT"
    ],
    "HIPDNN_CONVOLUTION_FWD_ALGO_COUNT": [
        "CUDNN_CONVOLUTION_FWD_ALGO_COUNT"
    ],
    "HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT": [
        "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT"
    ],
    "HIPDNN_CONVOLUTION_FWD_ALGO_FFT": [
        "CUDNN_CONVOLUTION_FWD_ALGO_FFT"
    ],
    "HIPDNN_CONVOLUTION_FWD_ALGO_FFT_TILING": [
        "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING"
    ],
    "HIPDNN_CONVOLUTION_FWD_ALGO_GEMM": [
        "CUDNN_CONVOLUTION_FWD_ALGO_GEMM"
    ],
    "HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM": [
        "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM"
    ],
    "HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM": [
        "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM"
    ],
    "HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD": [
        "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD"
    ],
    "HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED": [
        "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED"
    ],
    "HIPDNN_CONVOLUTION_FWD_NO_WORKSPACE": [
        "CUDNN_CONVOLUTION_FWD_NO_WORKSPACE"
    ],
    "HIPDNN_CONVOLUTION_FWD_PREFER_FASTEST": [
        "CUDNN_CONVOLUTION_FWD_PREFER_FASTEST"
    ],
    "HIPDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT": [
        "CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT"
    ],
    "HIPDNN_CROSS_CORRELATION": [
        "CUDNN_CROSS_CORRELATION"
    ],
    "HIPDNN_DATA_DOUBLE": [
        "CUDNN_DATA_DOUBLE"
    ],
    "HIPDNN_DATA_FLOAT": [
        "CUDNN_DATA_FLOAT"
    ],
    "HIPDNN_DATA_HALF": [
        "CUDNN_DATA_HALF"
    ],
    "HIPDNN_DATA_INT32": [
        "CUDNN_DATA_INT32"
    ],
    "HIPDNN_DATA_INT8": [
        "CUDNN_DATA_INT8"
    ],
    "HIPDNN_DATA_INT8x4": [
        "CUDNN_DATA_INT8x4"
    ],
    "HIPDNN_DEFAULT_MATH": [
        "CUDNN_DEFAULT_MATH"
    ],
    "HIPDNN_GRU": [
        "CUDNN_GRU"
    ],
    "HIPDNN_LINEAR_INPUT": [
        "CUDNN_LINEAR_INPUT"
    ],
    "HIPDNN_LRN_CROSS_CHANNEL": [
        "CUDNN_LRN_CROSS_CHANNEL_DIM1"
    ],
    "HIPDNN_LSTM": [
        "CUDNN_LSTM"
    ],
    "HIPDNN_NOT_PROPAGATE_NAN": [
        "CUDNN_NOT_PROPAGATE_NAN"
    ],
    "HIPDNN_OP_TENSOR_ADD": [
        "CUDNN_OP_TENSOR_ADD"
    ],
    "HIPDNN_OP_TENSOR_MAX": [
        "CUDNN_OP_TENSOR_MAX"
    ],
    "HIPDNN_OP_TENSOR_MIN": [
        "CUDNN_OP_TENSOR_MIN"
    ],
    "HIPDNN_OP_TENSOR_MUL": [
        "CUDNN_OP_TENSOR_MUL"
    ],
    "HIPDNN_OP_TENSOR_SQRT": [
        "CUDNN_OP_TENSOR_SQRT"
    ],
    "HIPDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING": [
        "CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING"
    ],
    "HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING": [
        "CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING"
    ],
    "HIPDNN_POOLING_MAX": [
        "CUDNN_POOLING_MAX"
    ],
    "HIPDNN_POOLING_MAX_DETERMINISTIC": [
        "CUDNN_POOLING_MAX_DETERMINISTIC"
    ],
    "HIPDNN_PROPAGATE_NAN": [
        "CUDNN_PROPAGATE_NAN"
    ],
    "HIPDNN_REDUCE_TENSOR_ADD": [
        "CUDNN_REDUCE_TENSOR_ADD"
    ],
    "HIPDNN_REDUCE_TENSOR_AMAX": [
        "CUDNN_REDUCE_TENSOR_AMAX"
    ],
    "HIPDNN_REDUCE_TENSOR_AVG": [
        "CUDNN_REDUCE_TENSOR_AVG"
    ],
    "HIPDNN_REDUCE_TENSOR_FLATTENED_INDICES": [
        "CUDNN_REDUCE_TENSOR_FLATTENED_INDICES"
    ],
    "HIPDNN_REDUCE_TENSOR_MAX": [
        "CUDNN_REDUCE_TENSOR_MAX"
    ],
    "HIPDNN_REDUCE_TENSOR_MIN": [
        "CUDNN_REDUCE_TENSOR_MIN"
    ],
    "HIPDNN_REDUCE_TENSOR_MUL": [
        "CUDNN_REDUCE_TENSOR_MUL"
    ],
    "HIPDNN_REDUCE_TENSOR_MUL_NO_ZEROS": [
        "CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS"
    ],
    "HIPDNN_REDUCE_TENSOR_NORM1": [
        "CUDNN_REDUCE_TENSOR_NORM1"
    ],
    "HIPDNN_REDUCE_TENSOR_NORM2": [
        "CUDNN_REDUCE_TENSOR_NORM2"
    ],
    "HIPDNN_REDUCE_TENSOR_NO_INDICES": [
        "CUDNN_REDUCE_TENSOR_NO_INDICES"
    ],
    "HIPDNN_RNN_ALGO_PERSIST_DYNAMIC": [
        "CUDNN_RNN_ALGO_PERSIST_DYNAMIC"
    ],
    "HIPDNN_RNN_ALGO_PERSIST_STATIC": [
        "CUDNN_RNN_ALGO_PERSIST_STATIC"
    ],
    "HIPDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H": [
        "CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H"
    ],
    "HIPDNN_RNN_ALGO_STANDARD": [
        "CUDNN_RNN_ALGO_STANDARD"
    ],
    "HIPDNN_RNN_WITH_BIAS": [
        "CUDNN_RNN_DOUBLE_BIAS",
        "CUDNN_RNN_SINGLE_INP_BIAS",
        "CUDNN_RNN_SINGLE_REC_BIAS"
    ],
    "HIPDNN_RNN_NO_BIAS": [
        "CUDNN_RNN_NO_BIAS"
    ],
    "HIPDNN_RNN_RELU": [
        "CUDNN_RNN_RELU"
    ],
    "HIPDNN_RNN_TANH": [
        "CUDNN_RNN_TANH"
    ],
    "HIPDNN_SKIP_INPUT": [
        "CUDNN_SKIP_INPUT"
    ],
    "HIPDNN_SOFTMAX_ACCURATE": [
        "CUDNN_SOFTMAX_ACCURATE"
    ],
    "HIPDNN_SOFTMAX_FAST": [
        "CUDNN_SOFTMAX_FAST"
    ],
    "HIPDNN_SOFTMAX_LOG": [
        "CUDNN_SOFTMAX_LOG"
    ],
    "HIPDNN_SOFTMAX_MODE_CHANNEL": [
        "CUDNN_SOFTMAX_MODE_CHANNEL"
    ],
    "HIPDNN_SOFTMAX_MODE_INSTANCE": [
        "CUDNN_SOFTMAX_MODE_INSTANCE"
    ],
    "HIPDNN_STATUS_ALLOC_FAILED": [
        "CUDNN_STATUS_ALLOC_FAILED"
    ],
    "HIPDNN_STATUS_ARCH_MISMATCH": [
        "CUDNN_STATUS_ARCH_MISMATCH"
    ],
    "HIPDNN_STATUS_BAD_PARAM": [
        "CUDNN_STATUS_BAD_PARAM"
    ],
    "HIPDNN_STATUS_EXECUTION_FAILED": [
        "CUDNN_STATUS_EXECUTION_FAILED"
    ],
    "HIPDNN_STATUS_INTERNAL_ERROR": [
        "CUDNN_STATUS_INTERNAL_ERROR"
    ],
    "HIPDNN_STATUS_INVALID_VALUE": [
        "CUDNN_STATUS_INVALID_VALUE"
    ],
    "HIPDNN_STATUS_LICENSE_ERROR": [
        "CUDNN_STATUS_LICENSE_ERROR"
    ],
    "HIPDNN_STATUS_MAPPING_ERROR": [
        "CUDNN_STATUS_MAPPING_ERROR"
    ],
    "HIPDNN_STATUS_NOT_INITIALIZED": [
        "CUDNN_STATUS_NOT_INITIALIZED"
    ],
    "HIPDNN_STATUS_NOT_SUPPORTED": [
        "CUDNN_STATUS_NOT_SUPPORTED"
    ],
    "HIPDNN_STATUS_RUNTIME_PREREQUISITE_MISSING": [
        "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING"
    ],
    "HIPDNN_STATUS_SUCCESS": [
        "CUDNN_STATUS_SUCCESS"
    ],
    "HIPDNN_TENSOR_NCHW": [
        "CUDNN_TENSOR_NCHW"
    ],
    "HIPDNN_TENSOR_NCHW_VECT_C": [
        "CUDNN_TENSOR_NCHW_VECT_C"
    ],
    "HIPDNN_TENSOR_NHWC": [
        "CUDNN_TENSOR_NHWC"
    ],
    "HIPDNN_TENSOR_OP_MATH": [
        "CUDNN_TENSOR_OP_MATH"
    ],
    "HIPDNN_UNIDIRECTIONAL": [
        "CUDNN_UNIDIRECTIONAL"
    ],
    "HIPDNN_VERSION": [
        "CUDNN_VERSION"
    ],
    "HIPFFT_ALLOC_FAILED": [
        "CUFFT_ALLOC_FAILED"
    ],
    "HIPFFT_C2C": [
        "CUFFT_C2C"
    ],
    "HIPFFT_C2R": [
        "CUFFT_C2R"
    ],
    "HIPFFT_CB_LD_COMPLEX": [
        "CUFFT_CB_LD_COMPLEX"
    ],
    "HIPFFT_CB_LD_COMPLEX_DOUBLE": [
        "CUFFT_CB_LD_COMPLEX_DOUBLE"
    ],
    "HIPFFT_CB_LD_REAL": [
        "CUFFT_CB_LD_REAL"
    ],
    "HIPFFT_CB_LD_REAL_DOUBLE": [
        "CUFFT_CB_LD_REAL_DOUBLE"
    ],
    "HIPFFT_CB_ST_COMPLEX": [
        "CUFFT_CB_ST_COMPLEX"
    ],
    "HIPFFT_CB_ST_COMPLEX_DOUBLE": [
        "CUFFT_CB_ST_COMPLEX_DOUBLE"
    ],
    "HIPFFT_CB_ST_REAL": [
        "CUFFT_CB_ST_REAL"
    ],
    "HIPFFT_CB_ST_REAL_DOUBLE": [
        "CUFFT_CB_ST_REAL_DOUBLE"
    ],
    "HIPFFT_CB_UNDEFINED": [
        "CUFFT_CB_UNDEFINED"
    ],
    "HIPFFT_D2Z": [
        "CUFFT_D2Z"
    ],
    "HIPFFT_EXEC_FAILED": [
        "CUFFT_EXEC_FAILED"
    ],
    "HIPFFT_FORWARD": [
        "CUFFT_FORWARD"
    ],
    "HIPFFT_INCOMPLETE_PARAMETER_LIST": [
        "CUFFT_INCOMPLETE_PARAMETER_LIST"
    ],
    "HIPFFT_INTERNAL_ERROR": [
        "CUFFT_INTERNAL_ERROR"
    ],
    "HIPFFT_INVALID_DEVICE": [
        "CUFFT_INVALID_DEVICE"
    ],
    "HIPFFT_INVALID_PLAN": [
        "CUFFT_INVALID_PLAN"
    ],
    "HIPFFT_INVALID_SIZE": [
        "CUFFT_INVALID_SIZE"
    ],
    "HIPFFT_INVALID_TYPE": [
        "CUFFT_INVALID_TYPE"
    ],
    "HIPFFT_INVALID_VALUE": [
        "CUFFT_INVALID_VALUE"
    ],
    "HIPFFT_BACKWARD": [
        "CUFFT_INVERSE"
    ],
    "HIPFFT_NOT_IMPLEMENTED": [
        "CUFFT_NOT_IMPLEMENTED"
    ],
    "HIPFFT_NOT_SUPPORTED": [
        "CUFFT_NOT_SUPPORTED"
    ],
    "HIPFFT_NO_WORKSPACE": [
        "CUFFT_NO_WORKSPACE"
    ],
    "HIPFFT_PARSE_ERROR": [
        "CUFFT_PARSE_ERROR"
    ],
    "HIPFFT_R2C": [
        "CUFFT_R2C"
    ],
    "HIPFFT_SETUP_FAILED": [
        "CUFFT_SETUP_FAILED"
    ],
    "HIPFFT_SUCCESS": [
        "CUFFT_SUCCESS"
    ],
    "HIPFFT_UNALIGNED_DATA": [
        "CUFFT_UNALIGNED_DATA"
    ],
    "HIPFFT_Z2D": [
        "CUFFT_Z2D"
    ],
    "HIPFFT_Z2Z": [
        "CUFFT_Z2Z"
    ],
    "HIPRAND_RNG_PSEUDO_DEFAULT": [
        "CURAND_RNG_PSEUDO_DEFAULT"
    ],
    "HIPRAND_RNG_PSEUDO_MRG32K3A": [
        "CURAND_RNG_PSEUDO_MRG32K3A"
    ],
    "HIPRAND_RNG_PSEUDO_MT19937": [
        "CURAND_RNG_PSEUDO_MT19937"
    ],
    "HIPRAND_RNG_PSEUDO_MTGP32": [
        "CURAND_RNG_PSEUDO_MTGP32"
    ],
    "HIPRAND_RNG_PSEUDO_PHILOX4_32_10": [
        "CURAND_RNG_PSEUDO_PHILOX4_32_10"
    ],
    "HIPRAND_RNG_PSEUDO_XORWOW": [
        "CURAND_RNG_PSEUDO_XORWOW"
    ],
    "HIPRAND_RNG_QUASI_DEFAULT": [
        "CURAND_RNG_QUASI_DEFAULT"
    ],
    "HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32": [
        "CURAND_RNG_QUASI_SCRAMBLED_SOBOL32"
    ],
    "HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64": [
        "CURAND_RNG_QUASI_SCRAMBLED_SOBOL64"
    ],
    "HIPRAND_RNG_QUASI_SOBOL32": [
        "CURAND_RNG_QUASI_SOBOL32"
    ],
    "HIPRAND_RNG_QUASI_SOBOL64": [
        "CURAND_RNG_QUASI_SOBOL64"
    ],
    "HIPRAND_RNG_TEST": [
        "CURAND_RNG_TEST"
    ],
    "HIPRAND_STATUS_ALLOCATION_FAILED": [
        "CURAND_STATUS_ALLOCATION_FAILED"
    ],
    "HIPRAND_STATUS_ARCH_MISMATCH": [
        "CURAND_STATUS_ARCH_MISMATCH"
    ],
    "HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED": [
        "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED"
    ],
    "HIPRAND_STATUS_INITIALIZATION_FAILED": [
        "CURAND_STATUS_INITIALIZATION_FAILED"
    ],
    "HIPRAND_STATUS_INTERNAL_ERROR": [
        "CURAND_STATUS_INTERNAL_ERROR"
    ],
    "HIPRAND_STATUS_LAUNCH_FAILURE": [
        "CURAND_STATUS_LAUNCH_FAILURE"
    ],
    "HIPRAND_STATUS_LENGTH_NOT_MULTIPLE": [
        "CURAND_STATUS_LENGTH_NOT_MULTIPLE"
    ],
    "HIPRAND_STATUS_NOT_INITIALIZED": [
        "CURAND_STATUS_NOT_INITIALIZED"
    ],
    "HIPRAND_STATUS_OUT_OF_RANGE": [
        "CURAND_STATUS_OUT_OF_RANGE"
    ],
    "HIPRAND_STATUS_PREEXISTING_FAILURE": [
        "CURAND_STATUS_PREEXISTING_FAILURE"
    ],
    "HIPRAND_STATUS_SUCCESS": [
        "CURAND_STATUS_SUCCESS"
    ],
    "HIPRAND_STATUS_TYPE_ERROR": [
        "CURAND_STATUS_TYPE_ERROR"
    ],
    "HIPRAND_STATUS_VERSION_MISMATCH": [
        "CURAND_STATUS_VERSION_MISMATCH"
    ],
    "HIPSPARSE_ACTION_NUMERIC": [
        "CUSPARSE_ACTION_NUMERIC"
    ],
    "HIPSPARSE_ACTION_SYMBOLIC": [
        "CUSPARSE_ACTION_SYMBOLIC"
    ],
    "HIPSPARSE_COOMM_ALG1": [
        "CUSPARSE_COOMM_ALG1"
    ],
    "HIPSPARSE_COOMM_ALG2": [
        "CUSPARSE_COOMM_ALG2"
    ],
    "HIPSPARSE_COOMM_ALG3": [
        "CUSPARSE_COOMM_ALG3"
    ],
    "HIPSPARSE_COOMV_ALG": [
        "CUSPARSE_COOMV_ALG"
    ],
    "HIPSPARSE_CSRMM_ALG1": [
        "CUSPARSE_CSRMM_ALG1"
    ],
    "HIPSPARSE_CSRMV_ALG1": [
        "CUSPARSE_CSRMV_ALG1"
    ],
    "HIPSPARSE_CSRMV_ALG2": [
        "CUSPARSE_CSRMV_ALG2"
    ],
    "HIPSPARSE_DIAG_TYPE_NON_UNIT": [
        "CUSPARSE_DIAG_TYPE_NON_UNIT"
    ],
    "HIPSPARSE_DIAG_TYPE_UNIT": [
        "CUSPARSE_DIAG_TYPE_UNIT"
    ],
    "HIPSPARSE_DIRECTION_COLUMN": [
        "CUSPARSE_DIRECTION_COLUMN"
    ],
    "HIPSPARSE_DIRECTION_ROW": [
        "CUSPARSE_DIRECTION_ROW"
    ],
    "HIPSPARSE_FILL_MODE_LOWER": [
        "CUSPARSE_FILL_MODE_LOWER"
    ],
    "HIPSPARSE_FILL_MODE_UPPER": [
        "CUSPARSE_FILL_MODE_UPPER"
    ],
    "HIPSPARSE_FORMAT_BLOCKED_ELL": [
        "CUSPARSE_FORMAT_BLOCKED_ELL"
    ],
    "HIPSPARSE_FORMAT_COO": [
        "CUSPARSE_FORMAT_COO"
    ],
    "HIPSPARSE_FORMAT_COO_AOS": [
        "CUSPARSE_FORMAT_COO_AOS"
    ],
    "HIPSPARSE_FORMAT_CSC": [
        "CUSPARSE_FORMAT_CSC"
    ],
    "HIPSPARSE_FORMAT_CSR": [
        "CUSPARSE_FORMAT_CSR"
    ],
    "HIPSPARSE_HYB_PARTITION_AUTO": [
        "CUSPARSE_HYB_PARTITION_AUTO"
    ],
    "HIPSPARSE_HYB_PARTITION_MAX": [
        "CUSPARSE_HYB_PARTITION_MAX"
    ],
    "HIPSPARSE_HYB_PARTITION_USER": [
        "CUSPARSE_HYB_PARTITION_USER"
    ],
    "HIPSPARSE_INDEX_16U": [
        "CUSPARSE_INDEX_16U"
    ],
    "HIPSPARSE_INDEX_32I": [
        "CUSPARSE_INDEX_32I"
    ],
    "HIPSPARSE_INDEX_64I": [
        "CUSPARSE_INDEX_64I"
    ],
    "HIPSPARSE_INDEX_BASE_ONE": [
        "CUSPARSE_INDEX_BASE_ONE"
    ],
    "HIPSPARSE_INDEX_BASE_ZERO": [
        "CUSPARSE_INDEX_BASE_ZERO"
    ],
    "HIPSPARSE_MATRIX_TYPE_GENERAL": [
        "CUSPARSE_MATRIX_TYPE_GENERAL"
    ],
    "HIPSPARSE_MATRIX_TYPE_HERMITIAN": [
        "CUSPARSE_MATRIX_TYPE_HERMITIAN"
    ],
    "HIPSPARSE_MATRIX_TYPE_SYMMETRIC": [
        "CUSPARSE_MATRIX_TYPE_SYMMETRIC"
    ],
    "HIPSPARSE_MATRIX_TYPE_TRIANGULAR": [
        "CUSPARSE_MATRIX_TYPE_TRIANGULAR"
    ],
    "HIPSPARSE_MM_ALG_DEFAULT": [
        "CUSPARSE_MM_ALG_DEFAULT"
    ],
    "HIPSPARSE_MV_ALG_DEFAULT": [
        "CUSPARSE_MV_ALG_DEFAULT"
    ],
    "HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE": [
        "CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE"
    ],
    "HIPSPARSE_OPERATION_NON_TRANSPOSE": [
        "CUSPARSE_OPERATION_NON_TRANSPOSE"
    ],
    "HIPSPARSE_OPERATION_TRANSPOSE": [
        "CUSPARSE_OPERATION_TRANSPOSE"
    ],
    "HIPSPARSE_ORDER_COL": [
        "CUSPARSE_ORDER_COL"
    ],
    "HIPSPARSE_ORDER_ROW": [
        "CUSPARSE_ORDER_ROW"
    ],
    "HIPSPARSE_POINTER_MODE_DEVICE": [
        "CUSPARSE_POINTER_MODE_DEVICE"
    ],
    "HIPSPARSE_POINTER_MODE_HOST": [
        "CUSPARSE_POINTER_MODE_HOST"
    ],
    "HIPSPARSE_SDDMM_ALG_DEFAULT": [
        "CUSPARSE_SDDMM_ALG_DEFAULT"
    ],
    "HIPSPARSE_SOLVE_POLICY_NO_LEVEL": [
        "CUSPARSE_SOLVE_POLICY_NO_LEVEL"
    ],
    "HIPSPARSE_SOLVE_POLICY_USE_LEVEL": [
        "CUSPARSE_SOLVE_POLICY_USE_LEVEL"
    ],
    "HIPSPARSE_SPARSETODENSE_ALG_DEFAULT": [
        "CUSPARSE_SPARSETODENSE_ALG_DEFAULT"
    ],
    "HIPSPARSE_SPGEMM_DEFAULT": [
        "CUSPARSE_SPGEMM_DEFAULT"
    ],
    "HIPSPARSE_SPMAT_DIAG_TYPE": [
        "CUSPARSE_SPMAT_DIAG_TYPE"
    ],
    "HIPSPARSE_SPMAT_FILL_MODE": [
        "CUSPARSE_SPMAT_FILL_MODE"
    ],
    "HIPSPARSE_SPMM_ALG_DEFAULT": [
        "CUSPARSE_SPMM_ALG_DEFAULT"
    ],
    "HIPSPARSE_SPMM_BLOCKED_ELL_ALG1": [
        "CUSPARSE_SPMM_BLOCKED_ELL_ALG1"
    ],
    "HIPSPARSE_SPMM_COO_ALG1": [
        "CUSPARSE_SPMM_COO_ALG1"
    ],
    "HIPSPARSE_SPMM_COO_ALG2": [
        "CUSPARSE_SPMM_COO_ALG2"
    ],
    "HIPSPARSE_SPMM_COO_ALG3": [
        "CUSPARSE_SPMM_COO_ALG3"
    ],
    "HIPSPARSE_SPMM_COO_ALG4": [
        "CUSPARSE_SPMM_COO_ALG4"
    ],
    "HIPSPARSE_SPMM_CSR_ALG1": [
        "CUSPARSE_SPMM_CSR_ALG1"
    ],
    "HIPSPARSE_SPMM_CSR_ALG2": [
        "CUSPARSE_SPMM_CSR_ALG2"
    ],
    "HIPSPARSE_SPMM_CSR_ALG3": [
        "CUSPARSE_SPMM_CSR_ALG3"
    ],
    "HIPSPARSE_SPMV_ALG_DEFAULT": [
        "CUSPARSE_SPMV_ALG_DEFAULT"
    ],
    "HIPSPARSE_SPMV_COO_ALG1": [
        "CUSPARSE_SPMV_COO_ALG1"
    ],
    "HIPSPARSE_SPMV_COO_ALG2": [
        "CUSPARSE_SPMV_COO_ALG2"
    ],
    "HIPSPARSE_SPMV_CSR_ALG1": [
        "CUSPARSE_SPMV_CSR_ALG1"
    ],
    "HIPSPARSE_SPMV_CSR_ALG2": [
        "CUSPARSE_SPMV_CSR_ALG2"
    ],
    "HIPSPARSE_SPSM_ALG_DEFAULT": [
        "CUSPARSE_SPSM_ALG_DEFAULT"
    ],
    "HIPSPARSE_SPSV_ALG_DEFAULT": [
        "CUSPARSE_SPSV_ALG_DEFAULT"
    ],
    "HIPSPARSE_STATUS_ALLOC_FAILED": [
        "CUSPARSE_STATUS_ALLOC_FAILED"
    ],
    "HIPSPARSE_STATUS_ARCH_MISMATCH": [
        "CUSPARSE_STATUS_ARCH_MISMATCH"
    ],
    "HIPSPARSE_STATUS_EXECUTION_FAILED": [
        "CUSPARSE_STATUS_EXECUTION_FAILED"
    ],
    "HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES": [
        "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES"
    ],
    "HIPSPARSE_STATUS_INTERNAL_ERROR": [
        "CUSPARSE_STATUS_INTERNAL_ERROR"
    ],
    "HIPSPARSE_STATUS_INVALID_VALUE": [
        "CUSPARSE_STATUS_INVALID_VALUE"
    ],
    "HIPSPARSE_STATUS_MAPPING_ERROR": [
        "CUSPARSE_STATUS_MAPPING_ERROR"
    ],
    "HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED": [
        "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED"
    ],
    "HIPSPARSE_STATUS_NOT_INITIALIZED": [
        "CUSPARSE_STATUS_NOT_INITIALIZED"
    ],
    "HIPSPARSE_STATUS_NOT_SUPPORTED": [
        "CUSPARSE_STATUS_NOT_SUPPORTED"
    ],
    "HIPSPARSE_STATUS_SUCCESS": [
        "CUSPARSE_STATUS_SUCCESS"
    ],
    "HIPSPARSE_STATUS_ZERO_PIVOT": [
        "CUSPARSE_STATUS_ZERO_PIVOT"
    ],
    "hipAccessPropertyNormal": [
        "CU_ACCESS_PROPERTY_NORMAL",
        "cudaAccessPropertyNormal"
    ],
    "hipAccessPropertyPersisting": [
        "CU_ACCESS_PROPERTY_PERSISTING",
        "cudaAccessPropertyPersisting"
    ],
    "hipAccessPropertyStreaming": [
        "CU_ACCESS_PROPERTY_STREAMING",
        "cudaAccessPropertyStreaming"
    ],
    "HIP_AD_FORMAT_FLOAT": [
        "CU_AD_FORMAT_FLOAT"
    ],
    "HIP_AD_FORMAT_HALF": [
        "CU_AD_FORMAT_HALF"
    ],
    "HIP_AD_FORMAT_SIGNED_INT16": [
        "CU_AD_FORMAT_SIGNED_INT16"
    ],
    "HIP_AD_FORMAT_SIGNED_INT32": [
        "CU_AD_FORMAT_SIGNED_INT32"
    ],
    "HIP_AD_FORMAT_SIGNED_INT8": [
        "CU_AD_FORMAT_SIGNED_INT8"
    ],
    "HIP_AD_FORMAT_UNSIGNED_INT16": [
        "CU_AD_FORMAT_UNSIGNED_INT16"
    ],
    "HIP_AD_FORMAT_UNSIGNED_INT32": [
        "CU_AD_FORMAT_UNSIGNED_INT32"
    ],
    "HIP_AD_FORMAT_UNSIGNED_INT8": [
        "CU_AD_FORMAT_UNSIGNED_INT8"
    ],
    "hipArraySparseSubresourceTypeMiptail": [
        "CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL"
    ],
    "hipArraySparseSubresourceTypeSparseLevel": [
        "CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL"
    ],
    "hipComputeModeDefault": [
        "CU_COMPUTEMODE_DEFAULT",
        "cudaComputeModeDefault"
    ],
    "hipComputeModeExclusive": [
        "CU_COMPUTEMODE_EXCLUSIVE",
        "cudaComputeModeExclusive"
    ],
    "hipComputeModeExclusiveProcess": [
        "CU_COMPUTEMODE_EXCLUSIVE_PROCESS",
        "cudaComputeModeExclusiveProcess"
    ],
    "hipComputeModeProhibited": [
        "CU_COMPUTEMODE_PROHIBITED",
        "cudaComputeModeProhibited"
    ],
    "hipDeviceScheduleBlockingSync": [
        "CU_CTX_BLOCKING_SYNC",
        "CU_CTX_SCHED_BLOCKING_SYNC",
        "cudaDeviceBlockingSync",
        "cudaDeviceScheduleBlockingSync"
    ],
    "hipDeviceLmemResizeToMax": [
        "CU_CTX_LMEM_RESIZE_TO_MAX",
        "cudaDeviceLmemResizeToMax"
    ],
    "hipDeviceMapHost": [
        "CU_CTX_MAP_HOST",
        "cudaDeviceMapHost"
    ],
    "hipDeviceScheduleAuto": [
        "CU_CTX_SCHED_AUTO",
        "cudaDeviceScheduleAuto"
    ],
    "hipDeviceScheduleMask": [
        "CU_CTX_SCHED_MASK",
        "cudaDeviceScheduleMask"
    ],
    "hipDeviceScheduleSpin": [
        "CU_CTX_SCHED_SPIN",
        "cudaDeviceScheduleSpin"
    ],
    "hipDeviceScheduleYield": [
        "CU_CTX_SCHED_YIELD",
        "cudaDeviceScheduleYield"
    ],
    "hipDeviceAttributeAsyncEngineCount": [
        "CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT",
        "CU_DEVICE_ATTRIBUTE_GPU_OVERLAP",
        "cudaDevAttrAsyncEngineCount",
        "cudaDevAttrGpuOverlap"
    ],
    "hipDeviceAttributeCanMapHostMemory": [
        "CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY",
        "cudaDevAttrCanMapHostMemory"
    ],
    "hipDeviceAttributeCanUseHostPointerForRegisteredMem": [
        "CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM",
        "cudaDevAttrCanUseHostPointerForRegisteredMem"
    ],
    "hipDeviceAttributeCanUseStreamWaitValue": [
        "CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR",
        "cudaDevAttrReserved94"
    ],
    "hipDeviceAttributeClockRate": [
        "CU_DEVICE_ATTRIBUTE_CLOCK_RATE",
        "cudaDevAttrClockRate"
    ],
    "hipDeviceAttributeComputeCapabilityMajor": [
        "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR",
        "cudaDevAttrComputeCapabilityMajor"
    ],
    "hipDeviceAttributeComputeCapabilityMinor": [
        "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR",
        "cudaDevAttrComputeCapabilityMinor"
    ],
    "hipDeviceAttributeComputeMode": [
        "CU_DEVICE_ATTRIBUTE_COMPUTE_MODE",
        "cudaDevAttrComputeMode"
    ],
    "hipDeviceAttributeComputePreemptionSupported": [
        "CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED",
        "cudaDevAttrComputePreemptionSupported"
    ],
    "hipDeviceAttributeConcurrentKernels": [
        "CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS",
        "cudaDevAttrConcurrentKernels"
    ],
    "hipDeviceAttributeConcurrentManagedAccess": [
        "CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS",
        "cudaDevAttrConcurrentManagedAccess"
    ],
    "hipDeviceAttributeCooperativeLaunch": [
        "CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH",
        "cudaDevAttrCooperativeLaunch"
    ],
    "hipDeviceAttributeCooperativeMultiDeviceLaunch": [
        "CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH",
        "cudaDevAttrCooperativeMultiDeviceLaunch"
    ],
    "hipDeviceAttributeDirectManagedMemAccessFromHost": [
        "CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST",
        "cudaDevAttrDirectManagedMemAccessFromHost"
    ],
    "hipDeviceAttributeEccEnabled": [
        "CU_DEVICE_ATTRIBUTE_ECC_ENABLED",
        "cudaDevAttrEccEnabled"
    ],
    "hipDeviceAttributeGlobalL1CacheSupported": [
        "CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED",
        "cudaDevAttrGlobalL1CacheSupported"
    ],
    "hipDeviceAttributeMemoryBusWidth": [
        "CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH",
        "cudaDevAttrGlobalMemoryBusWidth"
    ],
    "hipDeviceAttributeHostNativeAtomicSupported": [
        "CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED",
        "cudaDevAttrHostNativeAtomicSupported"
    ],
    "hipDeviceAttributeIntegrated": [
        "CU_DEVICE_ATTRIBUTE_INTEGRATED",
        "cudaDevAttrIntegrated"
    ],
    "hipDeviceAttributeKernelExecTimeout": [
        "CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT",
        "cudaDevAttrKernelExecTimeout"
    ],
    "hipDeviceAttributeL2CacheSize": [
        "CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE",
        "cudaDevAttrL2CacheSize"
    ],
    "hipDeviceAttributeLocalL1CacheSupported": [
        "CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED",
        "cudaDevAttrLocalL1CacheSupported"
    ],
    "hipDeviceAttributeManagedMemory": [
        "CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY",
        "cudaDevAttrManagedMemory"
    ],
    "hipDeviceAttributeMaxSurface1DLayered": [
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH",
        "cudaDevAttrMaxSurface1DLayeredWidth"
    ],
    "hipDeviceAttributeMaxSurface1D": [
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH",
        "cudaDevAttrMaxSurface1DWidth"
    ],
    "hipDeviceAttributeMaxSurface2D": [
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT",
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH",
        "cudaDevAttrMaxSurface2DHeight",
        "cudaDevAttrMaxSurface2DWidth"
    ],
    "hipDeviceAttributeMaxSurface2DLayered": [
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT",
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH",
        "cudaDevAttrMaxSurface2DLayeredHeight",
        "cudaDevAttrMaxSurface2DLayeredWidth"
    ],
    "hipDeviceAttributeMaxSurface3D": [
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH",
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT",
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH",
        "cudaDevAttrMaxSurface3DDepth",
        "cudaDevAttrMaxSurface3DHeight",
        "cudaDevAttrMaxSurface3DWidth"
    ],
    "hipDeviceAttributeMaxSurfaceCubemapLayered": [
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH",
        "cudaDevAttrMaxSurfaceCubemapLayeredWidth"
    ],
    "hipDeviceAttributeMaxSurfaceCubemap": [
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH",
        "cudaDevAttrMaxSurfaceCubemapWidth"
    ],
    "hipDeviceAttributeMaxTexture1DLayered": [
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH",
        "cudaDevAttrMaxTexture1DLayeredWidth"
    ],
    "hipDeviceAttributeMaxTexture1DLinear": [
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH",
        "cudaDevAttrMaxTexture1DLinearWidth"
    ],
    "hipDeviceAttributeMaxTexture1DMipmap": [
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH",
        "cudaDevAttrMaxTexture1DMipmappedWidth"
    ],
    "hipDeviceAttributeMaxTexture1DWidth": [
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH",
        "cudaDevAttrMaxTexture1DWidth"
    ],
    "hipDeviceAttributeMaxTexture2DLayered": [
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT",
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH",
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT",
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH",
        "cudaDevAttrMaxTexture2DLayeredHeight",
        "cudaDevAttrMaxTexture2DLayeredWidth"
    ],
    "hipDeviceAttributeMaxTexture2DGather": [
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT",
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH",
        "cudaDevAttrMaxTexture2DGatherHeight",
        "cudaDevAttrMaxTexture2DGatherWidth"
    ],
    "hipDeviceAttributeMaxTexture2DHeight": [
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT",
        "cudaDevAttrMaxTexture2DHeight"
    ],
    "hipDeviceAttributeMaxTexture2DLinear": [
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT",
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH",
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH",
        "cudaDevAttrMaxTexture2DLinearHeight",
        "cudaDevAttrMaxTexture2DLinearPitch",
        "cudaDevAttrMaxTexture2DLinearWidth"
    ],
    "hipDeviceAttributeMaxTexture2DMipmap": [
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT",
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH",
        "cudaDevAttrMaxTexture2DMipmappedHeight",
        "cudaDevAttrMaxTexture2DMipmappedWidth"
    ],
    "hipDeviceAttributeMaxTexture2DWidth": [
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH",
        "cudaDevAttrMaxTexture2DWidth"
    ],
    "hipDeviceAttributeMaxTexture3DDepth": [
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH",
        "cudaDevAttrMaxTexture3DDepth"
    ],
    "hipDeviceAttributeMaxTexture3DAlt": [
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE",
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE",
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE",
        "cudaDevAttrMaxTexture3DDepthAlt",
        "cudaDevAttrMaxTexture3DHeightAlt",
        "cudaDevAttrMaxTexture3DWidthAlt"
    ],
    "hipDeviceAttributeMaxTexture3DHeight": [
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT",
        "cudaDevAttrMaxTexture3DHeight"
    ],
    "hipDeviceAttributeMaxTexture3DWidth": [
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH",
        "cudaDevAttrMaxTexture3DWidth"
    ],
    "hipDeviceAttributeMaxTextureCubemapLayered": [
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH",
        "cudaDevAttrMaxTextureCubemapLayeredWidth"
    ],
    "hipDeviceAttributeMaxTextureCubemap": [
        "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH",
        "cudaDevAttrMaxTextureCubemapWidth"
    ],
    "hipDeviceAttributeMaxBlocksPerMultiprocessor": [
        "CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR",
        "cudaDevAttrMaxBlocksPerMultiprocessor"
    ],
    "hipDeviceAttributeMaxBlockDimX": [
        "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X",
        "cudaDevAttrMaxBlockDimX"
    ],
    "hipDeviceAttributeMaxBlockDimY": [
        "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y",
        "cudaDevAttrMaxBlockDimY"
    ],
    "hipDeviceAttributeMaxBlockDimZ": [
        "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z",
        "cudaDevAttrMaxBlockDimZ"
    ],
    "hipDeviceAttributeMaxGridDimX": [
        "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X",
        "cudaDevAttrMaxGridDimX"
    ],
    "hipDeviceAttributeMaxGridDimY": [
        "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y",
        "cudaDevAttrMaxGridDimY"
    ],
    "hipDeviceAttributeMaxGridDimZ": [
        "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z",
        "cudaDevAttrMaxGridDimZ"
    ],
    "hipDeviceAttributeMaxPitch": [
        "CU_DEVICE_ATTRIBUTE_MAX_PITCH",
        "cudaDevAttrMaxPitch"
    ],
    "hipDeviceAttributeMaxRegistersPerBlock": [
        "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK",
        "CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK",
        "cudaDevAttrMaxRegistersPerBlock"
    ],
    "hipDeviceAttributeMaxRegistersPerMultiprocessor": [
        "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR",
        "cudaDevAttrMaxRegistersPerMultiprocessor"
    ],
    "hipDeviceAttributeMaxSharedMemoryPerBlock": [
        "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK",
        "CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK",
        "cudaDevAttrMaxSharedMemoryPerBlock"
    ],
    "hipDeviceAttributeSharedMemPerBlockOptin": [
        "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN",
        "cudaDevAttrMaxSharedMemoryPerBlockOptin"
    ],
    "hipDeviceAttributeMaxSharedMemoryPerMultiprocessor": [
        "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR",
        "cudaDevAttrMaxSharedMemoryPerMultiprocessor"
    ],
    "hipDeviceAttributeMaxThreadsPerBlock": [
        "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK",
        "cudaDevAttrMaxThreadsPerBlock"
    ],
    "hipDeviceAttributeMaxThreadsPerMultiProcessor": [
        "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR",
        "cudaDevAttrMaxThreadsPerMultiProcessor"
    ],
    "hipDeviceAttributeMemoryClockRate": [
        "CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE",
        "cudaDevAttrMemoryClockRate"
    ],
    "hipDeviceAttributeMemoryPoolsSupported": [
        "CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED",
        "cudaDevAttrMemoryPoolsSupported"
    ],
    "hipDeviceAttributeMultiprocessorCount": [
        "CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT",
        "cudaDevAttrMultiProcessorCount"
    ],
    "hipDeviceAttributeIsMultiGpuBoard": [
        "CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD",
        "cudaDevAttrIsMultiGpuBoard"
    ],
    "hipDeviceAttributeMultiGpuBoardGroupId": [
        "CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID"
    ],
    "hipDeviceAttributePageableMemoryAccess": [
        "CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS",
        "cudaDevAttrPageableMemoryAccess"
    ],
    "hipDeviceAttributePageableMemoryAccessUsesHostPageTables": [
        "CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES",
        "cudaDevAttrPageableMemoryAccessUsesHostPageTables"
    ],
    "hipDeviceAttributePciBusId": [
        "CU_DEVICE_ATTRIBUTE_PCI_BUS_ID",
        "cudaDevAttrPciBusId"
    ],
    "hipDeviceAttributePciDeviceId": [
        "CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID",
        "cudaDevAttrPciDeviceId"
    ],
    "hipDeviceAttributePciDomainID": [
        "CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID",
        "cudaDevAttrPciDomainId"
    ],
    "hipDeviceAttributeSingleToDoublePrecisionPerfRatio": [
        "CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO",
        "cudaDevAttrSingleToDoublePrecisionPerfRatio"
    ],
    "hipDeviceAttributeStreamPrioritiesSupported": [
        "CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED",
        "cudaDevAttrStreamPrioritiesSupported"
    ],
    "hipDeviceAttributeSurfaceAlignment": [
        "CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT",
        "cudaDevAttrSurfaceAlignment"
    ],
    "hipDeviceAttributeTccDriver": [
        "CU_DEVICE_ATTRIBUTE_TCC_DRIVER",
        "cudaDevAttrTccDriver"
    ],
    "hipDeviceAttributeTextureAlignment": [
        "CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT",
        "cudaDevAttrTextureAlignment"
    ],
    "hipDeviceAttributeTexturePitchAlignment": [
        "CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT",
        "cudaDevAttrTexturePitchAlignment"
    ],
    "hipDeviceAttributeTotalConstantMemory": [
        "CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY",
        "cudaDevAttrTotalConstantMemory"
    ],
    "hipDeviceAttributeUnifiedAddressing": [
        "CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING",
        "cudaDevAttrUnifiedAddressing"
    ],
    "hipDeviceAttributeVirtualMemoryManagementSupported": [
        "CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED"
    ],
    "hipDeviceAttributeWarpSize": [
        "CU_DEVICE_ATTRIBUTE_WARP_SIZE",
        "cudaDevAttrWarpSize"
    ],
    "hipDevP2PAttrHipArrayAccessSupported": [
        "CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED",
        "CU_DEVICE_P2P_ATTRIBUTE_ARRAY_ACCESS_ACCESS_SUPPORTED",
        "CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED",
        "cudaDevP2PAttrCudaArrayAccessSupported"
    ],
    "hipDevP2PAttrAccessSupported": [
        "CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED",
        "cudaDevP2PAttrAccessSupported"
    ],
    "hipDevP2PAttrNativeAtomicSupported": [
        "CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED",
        "cudaDevP2PAttrNativeAtomicSupported"
    ],
    "hipDevP2PAttrPerformanceRank": [
        "CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK",
        "cudaDevP2PAttrPerformanceRank"
    ],
    "hipEventBlockingSync": [
        "CU_EVENT_BLOCKING_SYNC",
        "cudaEventBlockingSync"
    ],
    "hipEventDefault": [
        "CU_EVENT_DEFAULT",
        "cudaEventDefault"
    ],
    "hipEventDisableTiming": [
        "CU_EVENT_DISABLE_TIMING",
        "cudaEventDisableTiming"
    ],
    "hipEventInterprocess": [
        "CU_EVENT_INTERPROCESS",
        "cudaEventInterprocess"
    ],
    "hipExternalMemoryHandleTypeD3D11Resource": [
        "CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE",
        "cudaExternalMemoryHandleTypeD3D11Resource"
    ],
    "hipExternalMemoryHandleTypeD3D11ResourceKmt": [
        "CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT",
        "cudaExternalMemoryHandleTypeD3D11ResourceKmt"
    ],
    "hipExternalMemoryHandleTypeD3D12Heap": [
        "CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP",
        "cudaExternalMemoryHandleTypeD3D12Heap"
    ],
    "hipExternalMemoryHandleTypeD3D12Resource": [
        "CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE",
        "cudaExternalMemoryHandleTypeD3D12Resource"
    ],
    "hipExternalMemoryHandleTypeOpaqueFd": [
        "CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD",
        "cudaExternalMemoryHandleTypeOpaqueFd"
    ],
    "hipExternalMemoryHandleTypeOpaqueWin32": [
        "CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32",
        "cudaExternalMemoryHandleTypeOpaqueWin32"
    ],
    "hipExternalMemoryHandleTypeOpaqueWin32Kmt": [
        "CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT",
        "cudaExternalMemoryHandleTypeOpaqueWin32Kmt"
    ],
    "hipExternalSemaphoreHandleTypeD3D12Fence": [
        "CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE",
        "cudaExternalSemaphoreHandleTypeD3D12Fence"
    ],
    "hipExternalSemaphoreHandleTypeOpaqueFd": [
        "CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD",
        "cudaExternalSemaphoreHandleTypeOpaqueFd"
    ],
    "hipExternalSemaphoreHandleTypeOpaqueWin32": [
        "CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32",
        "cudaExternalSemaphoreHandleTypeOpaqueWin32"
    ],
    "hipExternalSemaphoreHandleTypeOpaqueWin32Kmt": [
        "CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT",
        "cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt"
    ],
    "HIP_FUNC_ATTRIBUTE_BINARY_VERSION": [
        "CU_FUNC_ATTRIBUTE_BINARY_VERSION"
    ],
    "HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA": [
        "CU_FUNC_ATTRIBUTE_CACHE_MODE_CA"
    ],
    "HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES": [
        "CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES"
    ],
    "HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES": [
        "CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES"
    ],
    "HIP_FUNC_ATTRIBUTE_MAX": [
        "CU_FUNC_ATTRIBUTE_MAX"
    ],
    "HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES": [
        "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES"
    ],
    "HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK": [
        "CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK"
    ],
    "HIP_FUNC_ATTRIBUTE_NUM_REGS": [
        "CU_FUNC_ATTRIBUTE_NUM_REGS"
    ],
    "HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT": [
        "CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT"
    ],
    "HIP_FUNC_ATTRIBUTE_PTX_VERSION": [
        "CU_FUNC_ATTRIBUTE_PTX_VERSION"
    ],
    "HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES": [
        "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES"
    ],
    "hipFuncCachePreferEqual": [
        "CU_FUNC_CACHE_PREFER_EQUAL",
        "cudaFuncCachePreferEqual"
    ],
    "hipFuncCachePreferL1": [
        "CU_FUNC_CACHE_PREFER_L1",
        "cudaFuncCachePreferL1"
    ],
    "hipFuncCachePreferNone": [
        "CU_FUNC_CACHE_PREFER_NONE",
        "cudaFuncCachePreferNone"
    ],
    "hipFuncCachePreferShared": [
        "CU_FUNC_CACHE_PREFER_SHARED",
        "cudaFuncCachePreferShared"
    ],
    "hipGLDeviceListAll": [
        "CU_GL_DEVICE_LIST_ALL",
        "cudaGLDeviceListAll"
    ],
    "hipGLDeviceListCurrentFrame": [
        "CU_GL_DEVICE_LIST_CURRENT_FRAME",
        "cudaGLDeviceListCurrentFrame"
    ],
    "hipGLDeviceListNextFrame": [
        "CU_GL_DEVICE_LIST_NEXT_FRAME",
        "cudaGLDeviceListNextFrame"
    ],
    "hipGraphicsRegisterFlagsNone": [
        "CU_GRAPHICS_REGISTER_FLAGS_NONE",
        "cudaGraphicsRegisterFlagsNone"
    ],
    "hipGraphicsRegisterFlagsReadOnly": [
        "CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY",
        "cudaGraphicsRegisterFlagsReadOnly"
    ],
    "hipGraphicsRegisterFlagsSurfaceLoadStore": [
        "CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST",
        "cudaGraphicsRegisterFlagsSurfaceLoadStore"
    ],
    "hipGraphicsRegisterFlagsTextureGather": [
        "CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER",
        "cudaGraphicsRegisterFlagsTextureGather"
    ],
    "hipGraphicsRegisterFlagsWriteDiscard": [
        "CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD",
        "cudaGraphicsRegisterFlagsWriteDiscard"
    ],
    "hipGraphExecUpdateError": [
        "CU_GRAPH_EXEC_UPDATE_ERROR",
        "cudaGraphExecUpdateError"
    ],
    "hipGraphExecUpdateErrorFunctionChanged": [
        "CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED",
        "cudaGraphExecUpdateErrorFunctionChanged"
    ],
    "hipGraphExecUpdateErrorNodeTypeChanged": [
        "CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED",
        "cudaGraphExecUpdateErrorNodeTypeChanged"
    ],
    "hipGraphExecUpdateErrorNotSupported": [
        "CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED",
        "cudaGraphExecUpdateErrorNotSupported"
    ],
    "hipGraphExecUpdateErrorParametersChanged": [
        "CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED",
        "cudaGraphExecUpdateErrorParametersChanged"
    ],
    "hipGraphExecUpdateErrorTopologyChanged": [
        "CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED",
        "cudaGraphExecUpdateErrorTopologyChanged"
    ],
    "hipGraphExecUpdateErrorUnsupportedFunctionChange": [
        "CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE",
        "cudaGraphExecUpdateErrorUnsupportedFunctionChange"
    ],
    "hipGraphExecUpdateSuccess": [
        "CU_GRAPH_EXEC_UPDATE_SUCCESS",
        "cudaGraphExecUpdateSuccess"
    ],
    "hipGraphMemAttrReservedMemCurrent": [
        "CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT",
        "cudaGraphMemAttrReservedMemCurrent"
    ],
    "hipGraphMemAttrReservedMemHigh": [
        "CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH",
        "cudaGraphMemAttrReservedMemHigh"
    ],
    "hipGraphMemAttrUsedMemCurrent": [
        "CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT",
        "cudaGraphMemAttrUsedMemCurrent"
    ],
    "hipGraphMemAttrUsedMemHigh": [
        "CU_GRAPH_MEM_ATTR_USED_MEM_HIGH",
        "cudaGraphMemAttrUsedMemHigh"
    ],
    "hipGraphNodeTypeCount": [
        "CU_GRAPH_NODE_TYPE_COUNT",
        "cudaGraphNodeTypeCount"
    ],
    "hipGraphNodeTypeEmpty": [
        "CU_GRAPH_NODE_TYPE_EMPTY",
        "cudaGraphNodeTypeEmpty"
    ],
    "hipGraphNodeTypeEventRecord": [
        "CU_GRAPH_NODE_TYPE_EVENT_RECORD",
        "cudaGraphNodeTypeEventRecord"
    ],
    "hipGraphNodeTypeExtSemaphoreSignal": [
        "CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL",
        "cudaGraphNodeTypeExtSemaphoreSignal"
    ],
    "hipGraphNodeTypeExtSemaphoreWait": [
        "CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT",
        "cudaGraphNodeTypeExtSemaphoreWait"
    ],
    "hipGraphNodeTypeGraph": [
        "CU_GRAPH_NODE_TYPE_GRAPH",
        "cudaGraphNodeTypeGraph"
    ],
    "hipGraphNodeTypeHost": [
        "CU_GRAPH_NODE_TYPE_HOST",
        "cudaGraphNodeTypeHost"
    ],
    "hipGraphNodeTypeKernel": [
        "CU_GRAPH_NODE_TYPE_KERNEL",
        "cudaGraphNodeTypeKernel"
    ],
    "hipGraphNodeTypeMemcpy": [
        "CU_GRAPH_NODE_TYPE_MEMCPY",
        "cudaGraphNodeTypeMemcpy"
    ],
    "hipGraphNodeTypeMemset": [
        "CU_GRAPH_NODE_TYPE_MEMSET",
        "cudaGraphNodeTypeMemset"
    ],
    "hipGraphNodeTypeWaitEvent": [
        "CU_GRAPH_NODE_TYPE_WAIT_EVENT",
        "cudaGraphNodeTypeWaitEvent"
    ],
    "hipGraphUserObjectMove": [
        "CU_GRAPH_USER_OBJECT_MOVE",
        "cudaGraphUserObjectMove"
    ],
    "hipIpcMemLazyEnablePeerAccess": [
        "CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS",
        "cudaIpcMemLazyEnablePeerAccess"
    ],
    "HIPRTC_JIT_CACHE_MODE": [
        "CU_JIT_CACHE_MODE"
    ],
    "HIPRTC_JIT_ERROR_LOG_BUFFER": [
        "CU_JIT_ERROR_LOG_BUFFER"
    ],
    "HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES": [
        "CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES"
    ],
    "HIPRTC_JIT_FALLBACK_STRATEGY": [
        "CU_JIT_FALLBACK_STRATEGY"
    ],
    "HIPRTC_JIT_FAST_COMPILE": [
        "CU_JIT_FAST_COMPILE"
    ],
    "HIPRTC_JIT_GENERATE_DEBUG_INFO": [
        "CU_JIT_GENERATE_DEBUG_INFO"
    ],
    "HIPRTC_JIT_GENERATE_LINE_INFO": [
        "CU_JIT_GENERATE_LINE_INFO"
    ],
    "HIPRTC_JIT_INFO_LOG_BUFFER": [
        "CU_JIT_INFO_LOG_BUFFER"
    ],
    "HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES": [
        "CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES"
    ],
    "HIPRTC_JIT_INPUT_CUBIN": [
        "CU_JIT_INPUT_CUBIN"
    ],
    "HIPRTC_JIT_INPUT_FATBINARY": [
        "CU_JIT_INPUT_FATBINARY"
    ],
    "HIPRTC_JIT_INPUT_LIBRARY": [
        "CU_JIT_INPUT_LIBRARY"
    ],
    "HIPRTC_JIT_INPUT_NVVM": [
        "CU_JIT_INPUT_NVVM"
    ],
    "HIPRTC_JIT_INPUT_OBJECT": [
        "CU_JIT_INPUT_OBJECT"
    ],
    "HIPRTC_JIT_INPUT_PTX": [
        "CU_JIT_INPUT_PTX"
    ],
    "HIPRTC_JIT_LOG_VERBOSE": [
        "CU_JIT_LOG_VERBOSE"
    ],
    "HIPRTC_JIT_MAX_REGISTERS": [
        "CU_JIT_MAX_REGISTERS"
    ],
    "HIPRTC_JIT_NEW_SM3X_OPT": [
        "CU_JIT_NEW_SM3X_OPT"
    ],
    "HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES": [
        "CU_JIT_NUM_INPUT_TYPES"
    ],
    "HIPRTC_JIT_NUM_OPTIONS": [
        "CU_JIT_NUM_OPTIONS"
    ],
    "HIPRTC_JIT_OPTIMIZATION_LEVEL": [
        "CU_JIT_OPTIMIZATION_LEVEL"
    ],
    "HIPRTC_JIT_TARGET": [
        "CU_JIT_TARGET"
    ],
    "HIPRTC_JIT_TARGET_FROM_HIPCONTEXT": [
        "CU_JIT_TARGET_FROM_CUCONTEXT"
    ],
    "HIPRTC_JIT_THREADS_PER_BLOCK": [
        "CU_JIT_THREADS_PER_BLOCK"
    ],
    "HIPRTC_JIT_WALL_TIME": [
        "CU_JIT_WALL_TIME"
    ],
    "hipKernelNodeAttributeAccessPolicyWindow": [
        "CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW",
        "cudaKernelNodeAttributeAccessPolicyWindow"
    ],
    "hipKernelNodeAttributeCooperative": [
        "CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE",
        "cudaKernelNodeAttributeCooperative"
    ],
    "hipLimitMallocHeapSize": [
        "CU_LIMIT_MALLOC_HEAP_SIZE",
        "cudaLimitMallocHeapSize"
    ],
    "hipLimitPrintfFifoSize": [
        "CU_LIMIT_PRINTF_FIFO_SIZE",
        "cudaLimitPrintfFifoSize"
    ],
    "hipLimitStackSize": [
        "CU_LIMIT_STACK_SIZE",
        "cudaLimitStackSize"
    ],
    "hipMemoryTypeArray": [
        "CU_MEMORYTYPE_ARRAY"
    ],
    "hipMemoryTypeDevice": [
        "CU_MEMORYTYPE_DEVICE",
        "cudaMemoryTypeDevice"
    ],
    "hipMemoryTypeHost": [
        "CU_MEMORYTYPE_HOST",
        "cudaMemoryTypeHost"
    ],
    "hipMemoryTypeUnified": [
        "CU_MEMORYTYPE_UNIFIED"
    ],
    "hipMemPoolAttrReleaseThreshold": [
        "CU_MEMPOOL_ATTR_RELEASE_THRESHOLD",
        "cudaMemPoolAttrReleaseThreshold"
    ],
    "hipMemPoolAttrReservedMemCurrent": [
        "CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT",
        "cudaMemPoolAttrReservedMemCurrent"
    ],
    "hipMemPoolAttrReservedMemHigh": [
        "CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH",
        "cudaMemPoolAttrReservedMemHigh"
    ],
    "hipMemPoolReuseAllowInternalDependencies": [
        "CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES",
        "cudaMemPoolReuseAllowInternalDependencies"
    ],
    "hipMemPoolReuseAllowOpportunistic": [
        "CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC",
        "cudaMemPoolReuseAllowOpportunistic"
    ],
    "hipMemPoolReuseFollowEventDependencies": [
        "CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES",
        "cudaMemPoolReuseFollowEventDependencies"
    ],
    "hipMemPoolAttrUsedMemCurrent": [
        "CU_MEMPOOL_ATTR_USED_MEM_CURRENT",
        "cudaMemPoolAttrUsedMemCurrent"
    ],
    "hipMemPoolAttrUsedMemHigh": [
        "CU_MEMPOOL_ATTR_USED_MEM_HIGH",
        "cudaMemPoolAttrUsedMemHigh"
    ],
    "hipMemAccessFlagsProtNone": [
        "CU_MEM_ACCESS_FLAGS_PROT_NONE",
        "cudaMemAccessFlagsProtNone"
    ],
    "hipMemAccessFlagsProtRead": [
        "CU_MEM_ACCESS_FLAGS_PROT_READ",
        "cudaMemAccessFlagsProtRead"
    ],
    "hipMemAccessFlagsProtReadWrite": [
        "CU_MEM_ACCESS_FLAGS_PROT_READWRITE",
        "cudaMemAccessFlagsProtReadWrite"
    ],
    "hipMemAdviseSetAccessedBy": [
        "CU_MEM_ADVISE_SET_ACCESSED_BY",
        "cudaMemAdviseSetAccessedBy"
    ],
    "hipMemAdviseSetPreferredLocation": [
        "CU_MEM_ADVISE_SET_PREFERRED_LOCATION",
        "cudaMemAdviseSetPreferredLocation"
    ],
    "hipMemAdviseSetReadMostly": [
        "CU_MEM_ADVISE_SET_READ_MOSTLY",
        "cudaMemAdviseSetReadMostly"
    ],
    "hipMemAdviseUnsetAccessedBy": [
        "CU_MEM_ADVISE_UNSET_ACCESSED_BY",
        "cudaMemAdviseUnsetAccessedBy"
    ],
    "hipMemAdviseUnsetPreferredLocation": [
        "CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION",
        "cudaMemAdviseUnsetPreferredLocation"
    ],
    "hipMemAdviseUnsetReadMostly": [
        "CU_MEM_ADVISE_UNSET_READ_MOSTLY",
        "cudaMemAdviseUnsetReadMostly"
    ],
    "hipMemAllocationTypeInvalid": [
        "CU_MEM_ALLOCATION_TYPE_INVALID",
        "cudaMemAllocationTypeInvalid"
    ],
    "hipMemAllocationTypeMax": [
        "CU_MEM_ALLOCATION_TYPE_MAX",
        "cudaMemAllocationTypeMax"
    ],
    "hipMemAllocationTypePinned": [
        "CU_MEM_ALLOCATION_TYPE_PINNED",
        "cudaMemAllocationTypePinned"
    ],
    "hipMemAllocationGranularityMinimum": [
        "CU_MEM_ALLOC_GRANULARITY_MINIMUM"
    ],
    "hipMemAllocationGranularityRecommended": [
        "CU_MEM_ALLOC_GRANULARITY_RECOMMENDED"
    ],
    "hipMemAttachGlobal": [
        "CU_MEM_ATTACH_GLOBAL",
        "cudaMemAttachGlobal"
    ],
    "hipMemAttachHost": [
        "CU_MEM_ATTACH_HOST",
        "cudaMemAttachHost"
    ],
    "hipMemAttachSingle": [
        "CU_MEM_ATTACH_SINGLE",
        "cudaMemAttachSingle"
    ],
    "hipMemHandleTypeGeneric": [
        "CU_MEM_HANDLE_TYPE_GENERIC"
    ],
    "hipMemHandleTypeNone": [
        "CU_MEM_HANDLE_TYPE_NONE",
        "cudaMemHandleTypeNone"
    ],
    "hipMemHandleTypePosixFileDescriptor": [
        "CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR",
        "cudaMemHandleTypePosixFileDescriptor"
    ],
    "hipMemHandleTypeWin32": [
        "CU_MEM_HANDLE_TYPE_WIN32",
        "cudaMemHandleTypeWin32"
    ],
    "hipMemHandleTypeWin32Kmt": [
        "CU_MEM_HANDLE_TYPE_WIN32_KMT",
        "cudaMemHandleTypeWin32Kmt"
    ],
    "hipMemLocationTypeDevice": [
        "CU_MEM_LOCATION_TYPE_DEVICE",
        "cudaMemLocationTypeDevice"
    ],
    "hipMemLocationTypeInvalid": [
        "CU_MEM_LOCATION_TYPE_INVALID",
        "cudaMemLocationTypeInvalid"
    ],
    "hipMemOperationTypeMap": [
        "CU_MEM_OPERATION_TYPE_MAP"
    ],
    "hipMemOperationTypeUnmap": [
        "CU_MEM_OPERATION_TYPE_UNMAP"
    ],
    "hipMemRangeAttributeAccessedBy": [
        "CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY",
        "cudaMemRangeAttributeAccessedBy"
    ],
    "hipMemRangeAttributeLastPrefetchLocation": [
        "CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION",
        "cudaMemRangeAttributeLastPrefetchLocation"
    ],
    "hipMemRangeAttributePreferredLocation": [
        "CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION",
        "cudaMemRangeAttributePreferredLocation"
    ],
    "hipMemRangeAttributeReadMostly": [
        "CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY",
        "cudaMemRangeAttributeReadMostly"
    ],
    "hipOccupancyDefault": [
        "CU_OCCUPANCY_DEFAULT",
        "cudaOccupancyDefault"
    ],
    "HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS": [
        "CU_POINTER_ATTRIBUTE_ACCESS_FLAGS"
    ],
    "HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES": [
        "CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES"
    ],
    "HIP_POINTER_ATTRIBUTE_BUFFER_ID": [
        "CU_POINTER_ATTRIBUTE_BUFFER_ID"
    ],
    "HIP_POINTER_ATTRIBUTE_CONTEXT": [
        "CU_POINTER_ATTRIBUTE_CONTEXT"
    ],
    "HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL": [
        "CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL"
    ],
    "HIP_POINTER_ATTRIBUTE_DEVICE_POINTER": [
        "CU_POINTER_ATTRIBUTE_DEVICE_POINTER"
    ],
    "HIP_POINTER_ATTRIBUTE_HOST_POINTER": [
        "CU_POINTER_ATTRIBUTE_HOST_POINTER"
    ],
    "HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE": [
        "CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE"
    ],
    "HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE": [
        "CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE"
    ],
    "HIP_POINTER_ATTRIBUTE_IS_MANAGED": [
        "CU_POINTER_ATTRIBUTE_IS_MANAGED"
    ],
    "HIP_POINTER_ATTRIBUTE_MAPPED": [
        "CU_POINTER_ATTRIBUTE_MAPPED"
    ],
    "HIP_POINTER_ATTRIBUTE_MEMORY_TYPE": [
        "CU_POINTER_ATTRIBUTE_MEMORY_TYPE"
    ],
    "HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE": [
        "CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE"
    ],
    "HIP_POINTER_ATTRIBUTE_P2P_TOKENS": [
        "CU_POINTER_ATTRIBUTE_P2P_TOKENS"
    ],
    "HIP_POINTER_ATTRIBUTE_RANGE_SIZE": [
        "CU_POINTER_ATTRIBUTE_RANGE_SIZE"
    ],
    "HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR": [
        "CU_POINTER_ATTRIBUTE_RANGE_START_ADDR"
    ],
    "HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS": [
        "CU_POINTER_ATTRIBUTE_SYNC_MEMOPS"
    ],
    "HIP_RESOURCE_TYPE_ARRAY": [
        "CU_RESOURCE_TYPE_ARRAY"
    ],
    "HIP_RESOURCE_TYPE_LINEAR": [
        "CU_RESOURCE_TYPE_LINEAR"
    ],
    "HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY": [
        "CU_RESOURCE_TYPE_MIPMAPPED_ARRAY"
    ],
    "HIP_RESOURCE_TYPE_PITCH2D": [
        "CU_RESOURCE_TYPE_PITCH2D"
    ],
    "HIP_RES_VIEW_FORMAT_FLOAT_1X16": [
        "CU_RES_VIEW_FORMAT_FLOAT_1X16"
    ],
    "HIP_RES_VIEW_FORMAT_FLOAT_1X32": [
        "CU_RES_VIEW_FORMAT_FLOAT_1X32"
    ],
    "HIP_RES_VIEW_FORMAT_FLOAT_2X16": [
        "CU_RES_VIEW_FORMAT_FLOAT_2X16"
    ],
    "HIP_RES_VIEW_FORMAT_FLOAT_2X32": [
        "CU_RES_VIEW_FORMAT_FLOAT_2X32"
    ],
    "HIP_RES_VIEW_FORMAT_FLOAT_4X16": [
        "CU_RES_VIEW_FORMAT_FLOAT_4X16"
    ],
    "HIP_RES_VIEW_FORMAT_FLOAT_4X32": [
        "CU_RES_VIEW_FORMAT_FLOAT_4X32"
    ],
    "HIP_RES_VIEW_FORMAT_NONE": [
        "CU_RES_VIEW_FORMAT_NONE"
    ],
    "HIP_RES_VIEW_FORMAT_SIGNED_BC4": [
        "CU_RES_VIEW_FORMAT_SIGNED_BC4"
    ],
    "HIP_RES_VIEW_FORMAT_SIGNED_BC5": [
        "CU_RES_VIEW_FORMAT_SIGNED_BC5"
    ],
    "HIP_RES_VIEW_FORMAT_SIGNED_BC6H": [
        "CU_RES_VIEW_FORMAT_SIGNED_BC6H"
    ],
    "HIP_RES_VIEW_FORMAT_SINT_1X16": [
        "CU_RES_VIEW_FORMAT_SINT_1X16"
    ],
    "HIP_RES_VIEW_FORMAT_SINT_1X32": [
        "CU_RES_VIEW_FORMAT_SINT_1X32"
    ],
    "HIP_RES_VIEW_FORMAT_SINT_1X8": [
        "CU_RES_VIEW_FORMAT_SINT_1X8"
    ],
    "HIP_RES_VIEW_FORMAT_SINT_2X16": [
        "CU_RES_VIEW_FORMAT_SINT_2X16"
    ],
    "HIP_RES_VIEW_FORMAT_SINT_2X32": [
        "CU_RES_VIEW_FORMAT_SINT_2X32"
    ],
    "HIP_RES_VIEW_FORMAT_SINT_2X8": [
        "CU_RES_VIEW_FORMAT_SINT_2X8"
    ],
    "HIP_RES_VIEW_FORMAT_SINT_4X16": [
        "CU_RES_VIEW_FORMAT_SINT_4X16"
    ],
    "HIP_RES_VIEW_FORMAT_SINT_4X32": [
        "CU_RES_VIEW_FORMAT_SINT_4X32"
    ],
    "HIP_RES_VIEW_FORMAT_SINT_4X8": [
        "CU_RES_VIEW_FORMAT_SINT_4X8"
    ],
    "HIP_RES_VIEW_FORMAT_UINT_1X16": [
        "CU_RES_VIEW_FORMAT_UINT_1X16"
    ],
    "HIP_RES_VIEW_FORMAT_UINT_1X32": [
        "CU_RES_VIEW_FORMAT_UINT_1X32"
    ],
    "HIP_RES_VIEW_FORMAT_UINT_1X8": [
        "CU_RES_VIEW_FORMAT_UINT_1X8"
    ],
    "HIP_RES_VIEW_FORMAT_UINT_2X16": [
        "CU_RES_VIEW_FORMAT_UINT_2X16"
    ],
    "HIP_RES_VIEW_FORMAT_UINT_2X32": [
        "CU_RES_VIEW_FORMAT_UINT_2X32"
    ],
    "HIP_RES_VIEW_FORMAT_UINT_2X8": [
        "CU_RES_VIEW_FORMAT_UINT_2X8"
    ],
    "HIP_RES_VIEW_FORMAT_UINT_4X16": [
        "CU_RES_VIEW_FORMAT_UINT_4X16"
    ],
    "HIP_RES_VIEW_FORMAT_UINT_4X32": [
        "CU_RES_VIEW_FORMAT_UINT_4X32"
    ],
    "HIP_RES_VIEW_FORMAT_UINT_4X8": [
        "CU_RES_VIEW_FORMAT_UINT_4X8"
    ],
    "HIP_RES_VIEW_FORMAT_UNSIGNED_BC1": [
        "CU_RES_VIEW_FORMAT_UNSIGNED_BC1"
    ],
    "HIP_RES_VIEW_FORMAT_UNSIGNED_BC2": [
        "CU_RES_VIEW_FORMAT_UNSIGNED_BC2"
    ],
    "HIP_RES_VIEW_FORMAT_UNSIGNED_BC3": [
        "CU_RES_VIEW_FORMAT_UNSIGNED_BC3"
    ],
    "HIP_RES_VIEW_FORMAT_UNSIGNED_BC4": [
        "CU_RES_VIEW_FORMAT_UNSIGNED_BC4"
    ],
    "HIP_RES_VIEW_FORMAT_UNSIGNED_BC5": [
        "CU_RES_VIEW_FORMAT_UNSIGNED_BC5"
    ],
    "HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H": [
        "CU_RES_VIEW_FORMAT_UNSIGNED_BC6H"
    ],
    "HIP_RES_VIEW_FORMAT_UNSIGNED_BC7": [
        "CU_RES_VIEW_FORMAT_UNSIGNED_BC7"
    ],
    "hipSharedMemBankSizeDefault": [
        "CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE",
        "cudaSharedMemBankSizeDefault"
    ],
    "hipSharedMemBankSizeEightByte": [
        "CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE",
        "cudaSharedMemBankSizeEightByte"
    ],
    "hipSharedMemBankSizeFourByte": [
        "CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE",
        "cudaSharedMemBankSizeFourByte"
    ],
    "hipStreamAddCaptureDependencies": [
        "CU_STREAM_ADD_CAPTURE_DEPENDENCIES",
        "cudaStreamAddCaptureDependencies"
    ],
    "hipStreamCaptureModeGlobal": [
        "CU_STREAM_CAPTURE_MODE_GLOBAL",
        "cudaStreamCaptureModeGlobal"
    ],
    "hipStreamCaptureModeRelaxed": [
        "CU_STREAM_CAPTURE_MODE_RELAXED",
        "cudaStreamCaptureModeRelaxed"
    ],
    "hipStreamCaptureModeThreadLocal": [
        "CU_STREAM_CAPTURE_MODE_THREAD_LOCAL",
        "cudaStreamCaptureModeThreadLocal"
    ],
    "hipStreamCaptureStatusActive": [
        "CU_STREAM_CAPTURE_STATUS_ACTIVE",
        "cudaStreamCaptureStatusActive"
    ],
    "hipStreamCaptureStatusInvalidated": [
        "CU_STREAM_CAPTURE_STATUS_INVALIDATED",
        "cudaStreamCaptureStatusInvalidated"
    ],
    "hipStreamCaptureStatusNone": [
        "CU_STREAM_CAPTURE_STATUS_NONE",
        "cudaStreamCaptureStatusNone"
    ],
    "hipStreamDefault": [
        "CU_STREAM_DEFAULT",
        "cudaStreamDefault"
    ],
    "hipStreamNonBlocking": [
        "CU_STREAM_NON_BLOCKING",
        "cudaStreamNonBlocking"
    ],
    "hipStreamSetCaptureDependencies": [
        "CU_STREAM_SET_CAPTURE_DEPENDENCIES",
        "cudaStreamSetCaptureDependencies"
    ],
    "hipStreamWaitValueAnd": [
        "CU_STREAM_WAIT_VALUE_AND"
    ],
    "hipStreamWaitValueEq": [
        "CU_STREAM_WAIT_VALUE_EQ"
    ],
    "hipStreamWaitValueGte": [
        "CU_STREAM_WAIT_VALUE_GEQ"
    ],
    "hipStreamWaitValueNor": [
        "CU_STREAM_WAIT_VALUE_NOR"
    ],
    "HIP_TR_ADDRESS_MODE_BORDER": [
        "CU_TR_ADDRESS_MODE_BORDER"
    ],
    "HIP_TR_ADDRESS_MODE_CLAMP": [
        "CU_TR_ADDRESS_MODE_CLAMP"
    ],
    "HIP_TR_ADDRESS_MODE_MIRROR": [
        "CU_TR_ADDRESS_MODE_MIRROR"
    ],
    "HIP_TR_ADDRESS_MODE_WRAP": [
        "CU_TR_ADDRESS_MODE_WRAP"
    ],
    "HIP_TR_FILTER_MODE_LINEAR": [
        "CU_TR_FILTER_MODE_LINEAR"
    ],
    "HIP_TR_FILTER_MODE_POINT": [
        "CU_TR_FILTER_MODE_POINT"
    ],
    "hipUserObjectNoDestructorSync": [
        "CU_USER_OBJECT_NO_DESTRUCTOR_SYNC",
        "cudaUserObjectNoDestructorSync"
    ],
    "HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE": [
        "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE"
    ],
    "HIPRTC_ERROR_COMPILATION": [
        "NVRTC_ERROR_COMPILATION"
    ],
    "HIPRTC_ERROR_INTERNAL_ERROR": [
        "NVRTC_ERROR_INTERNAL_ERROR"
    ],
    "HIPRTC_ERROR_INVALID_INPUT": [
        "NVRTC_ERROR_INVALID_INPUT"
    ],
    "HIPRTC_ERROR_INVALID_OPTION": [
        "NVRTC_ERROR_INVALID_OPTION"
    ],
    "HIPRTC_ERROR_INVALID_PROGRAM": [
        "NVRTC_ERROR_INVALID_PROGRAM"
    ],
    "HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID": [
        "NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID"
    ],
    "HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION": [
        "NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION"
    ],
    "HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION": [
        "NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION"
    ],
    "HIPRTC_ERROR_OUT_OF_MEMORY": [
        "NVRTC_ERROR_OUT_OF_MEMORY"
    ],
    "HIPRTC_ERROR_PROGRAM_CREATION_FAILURE": [
        "NVRTC_ERROR_PROGRAM_CREATION_FAILURE"
    ],
    "HIPRTC_SUCCESS": [
        "NVRTC_SUCCESS"
    ],
    "hipAddressModeBorder": [
        "cudaAddressModeBorder"
    ],
    "hipAddressModeClamp": [
        "cudaAddressModeClamp"
    ],
    "hipAddressModeMirror": [
        "cudaAddressModeMirror"
    ],
    "hipAddressModeWrap": [
        "cudaAddressModeWrap"
    ],
    "hipBoundaryModeClamp": [
        "cudaBoundaryModeClamp"
    ],
    "hipBoundaryModeTrap": [
        "cudaBoundaryModeTrap"
    ],
    "hipBoundaryModeZero": [
        "cudaBoundaryModeZero"
    ],
    "hipChannelFormatKindFloat": [
        "cudaChannelFormatKindFloat"
    ],
    "hipChannelFormatKindNone": [
        "cudaChannelFormatKindNone"
    ],
    "hipChannelFormatKindSigned": [
        "cudaChannelFormatKindSigned"
    ],
    "hipChannelFormatKindUnsigned": [
        "cudaChannelFormatKindUnsigned"
    ],
    "hipDeviceAttributeMultiGpuBoardGroupID": [
        "cudaDevAttrMultiGpuBoardGroupID"
    ],
    "hipErrorInsufficientDriver": [
        "cudaErrorInsufficientDriver"
    ],
    "hipErrorInvalidConfiguration": [
        "cudaErrorInvalidConfiguration"
    ],
    "hipErrorInvalidDeviceFunction": [
        "cudaErrorInvalidDeviceFunction"
    ],
    "hipErrorInvalidDevicePointer": [
        "cudaErrorInvalidDevicePointer"
    ],
    "hipErrorInvalidMemcpyDirection": [
        "cudaErrorInvalidMemcpyDirection"
    ],
    "hipErrorInvalidPitchValue": [
        "cudaErrorInvalidPitchValue"
    ],
    "hipErrorInvalidSymbol": [
        "cudaErrorInvalidSymbol"
    ],
    "hipErrorMissingConfiguration": [
        "cudaErrorMissingConfiguration"
    ],
    "hipErrorPriorLaunchFailure": [
        "cudaErrorPriorLaunchFailure"
    ],
    "hipFilterModeLinear": [
        "cudaFilterModeLinear"
    ],
    "hipFilterModePoint": [
        "cudaFilterModePoint"
    ],
    "hipFuncAttributeMax": [
        "cudaFuncAttributeMax"
    ],
    "hipFuncAttributeMaxDynamicSharedMemorySize": [
        "cudaFuncAttributeMaxDynamicSharedMemorySize"
    ],
    "hipFuncAttributePreferredSharedMemoryCarveout": [
        "cudaFuncAttributePreferredSharedMemoryCarveout"
    ],
    "hipMemcpyDefault": [
        "cudaMemcpyDefault"
    ],
    "hipMemcpyDeviceToDevice": [
        "cudaMemcpyDeviceToDevice"
    ],
    "hipMemcpyDeviceToHost": [
        "cudaMemcpyDeviceToHost"
    ],
    "hipMemcpyHostToDevice": [
        "cudaMemcpyHostToDevice"
    ],
    "hipMemcpyHostToHost": [
        "cudaMemcpyHostToHost"
    ],
    "hipMemoryTypeManaged": [
        "cudaMemoryTypeManaged"
    ],
    "hipReadModeElementType": [
        "cudaReadModeElementType"
    ],
    "hipReadModeNormalizedFloat": [
        "cudaReadModeNormalizedFloat"
    ],
    "hipResViewFormatFloat1": [
        "cudaResViewFormatFloat1"
    ],
    "hipResViewFormatFloat2": [
        "cudaResViewFormatFloat2"
    ],
    "hipResViewFormatFloat4": [
        "cudaResViewFormatFloat4"
    ],
    "hipResViewFormatHalf1": [
        "cudaResViewFormatHalf1"
    ],
    "hipResViewFormatHalf2": [
        "cudaResViewFormatHalf2"
    ],
    "hipResViewFormatHalf4": [
        "cudaResViewFormatHalf4"
    ],
    "hipResViewFormatNone": [
        "cudaResViewFormatNone"
    ],
    "hipResViewFormatSignedBlockCompressed4": [
        "cudaResViewFormatSignedBlockCompressed4"
    ],
    "hipResViewFormatSignedBlockCompressed5": [
        "cudaResViewFormatSignedBlockCompressed5"
    ],
    "hipResViewFormatSignedBlockCompressed6H": [
        "cudaResViewFormatSignedBlockCompressed6H"
    ],
    "hipResViewFormatSignedChar1": [
        "cudaResViewFormatSignedChar1"
    ],
    "hipResViewFormatSignedChar2": [
        "cudaResViewFormatSignedChar2"
    ],
    "hipResViewFormatSignedChar4": [
        "cudaResViewFormatSignedChar4"
    ],
    "hipResViewFormatSignedInt1": [
        "cudaResViewFormatSignedInt1"
    ],
    "hipResViewFormatSignedInt2": [
        "cudaResViewFormatSignedInt2"
    ],
    "hipResViewFormatSignedInt4": [
        "cudaResViewFormatSignedInt4"
    ],
    "hipResViewFormatSignedShort1": [
        "cudaResViewFormatSignedShort1"
    ],
    "hipResViewFormatSignedShort2": [
        "cudaResViewFormatSignedShort2"
    ],
    "hipResViewFormatSignedShort4": [
        "cudaResViewFormatSignedShort4"
    ],
    "hipResViewFormatUnsignedBlockCompressed1": [
        "cudaResViewFormatUnsignedBlockCompressed1"
    ],
    "hipResViewFormatUnsignedBlockCompressed2": [
        "cudaResViewFormatUnsignedBlockCompressed2"
    ],
    "hipResViewFormatUnsignedBlockCompressed3": [
        "cudaResViewFormatUnsignedBlockCompressed3"
    ],
    "hipResViewFormatUnsignedBlockCompressed4": [
        "cudaResViewFormatUnsignedBlockCompressed4"
    ],
    "hipResViewFormatUnsignedBlockCompressed5": [
        "cudaResViewFormatUnsignedBlockCompressed5"
    ],
    "hipResViewFormatUnsignedBlockCompressed6H": [
        "cudaResViewFormatUnsignedBlockCompressed6H"
    ],
    "hipResViewFormatUnsignedBlockCompressed7": [
        "cudaResViewFormatUnsignedBlockCompressed7"
    ],
    "hipResViewFormatUnsignedChar1": [
        "cudaResViewFormatUnsignedChar1"
    ],
    "hipResViewFormatUnsignedChar2": [
        "cudaResViewFormatUnsignedChar2"
    ],
    "hipResViewFormatUnsignedChar4": [
        "cudaResViewFormatUnsignedChar4"
    ],
    "hipResViewFormatUnsignedInt1": [
        "cudaResViewFormatUnsignedInt1"
    ],
    "hipResViewFormatUnsignedInt2": [
        "cudaResViewFormatUnsignedInt2"
    ],
    "hipResViewFormatUnsignedInt4": [
        "cudaResViewFormatUnsignedInt4"
    ],
    "hipResViewFormatUnsignedShort1": [
        "cudaResViewFormatUnsignedShort1"
    ],
    "hipResViewFormatUnsignedShort2": [
        "cudaResViewFormatUnsignedShort2"
    ],
    "hipResViewFormatUnsignedShort4": [
        "cudaResViewFormatUnsignedShort4"
    ],
    "hipResourceTypeArray": [
        "cudaResourceTypeArray"
    ],
    "hipResourceTypeLinear": [
        "cudaResourceTypeLinear"
    ],
    "hipResourceTypeMipmappedArray": [
        "cudaResourceTypeMipmappedArray"
    ],
    "hipResourceTypePitch2D": [
        "cudaResourceTypePitch2D"
    ],
    "CUB_MAX": [
        "CUB_MAX"
    ],
    "CUB_MIN": [
        "CUB_MIN"
    ],
    "BEGIN_HIPCUB_NAMESPACE": [
        "CUB_NAMESPACE_BEGIN"
    ],
    "END_HIPCUB_NAMESPACE": [
        "CUB_NAMESPACE_END"
    ],
    "HIPCUB_ARCH": [
        "CUB_PTX_ARCH"
    ],
    "HIPCUB_WARP_THREADS": [
        "CUB_PTX_WARP_THREADS"
    ],
    "HIPCUB_RUNTIME_FUNCTION": [
        "CUB_RUNTIME_FUNCTION"
    ],
    "HIPCUB_STDERR": [
        "CUB_STDERR"
    ],
    "hipArrayCubemap": [
        "CUDA_ARRAY3D_CUBEMAP",
        "cudaArrayCubemap"
    ],
    "hipArrayLayered": [
        "CUDA_ARRAY3D_LAYERED",
        "cudaArrayLayered"
    ],
    "hipArraySurfaceLoadStore": [
        "CUDA_ARRAY3D_SURFACE_LDST",
        "cudaArraySurfaceLoadStore"
    ],
    "hipArrayTextureGather": [
        "CUDA_ARRAY3D_TEXTURE_GATHER",
        "cudaArrayTextureGather"
    ],
    "hipCooperativeLaunchMultiDeviceNoPostSync": [
        "CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC",
        "cudaCooperativeLaunchMultiDeviceNoPostSync"
    ],
    "hipCooperativeLaunchMultiDeviceNoPreSync": [
        "CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC",
        "cudaCooperativeLaunchMultiDeviceNoPreSync"
    ],
    "HIP_IPC_HANDLE_SIZE": [
        "CUDA_IPC_HANDLE_SIZE",
        "CU_IPC_HANDLE_SIZE"
    ],
    "HIPRAND_VERSION": [
        "CURAND_VERSION"
    ],
    "hipCpuDeviceId": [
        "CU_DEVICE_CPU",
        "cudaCpuDeviceId"
    ],
    "hipInvalidDeviceId": [
        "CU_DEVICE_INVALID",
        "cudaInvalidDeviceId"
    ],
    "HIP_LAUNCH_PARAM_BUFFER_POINTER": [
        "CU_LAUNCH_PARAM_BUFFER_POINTER"
    ],
    "HIP_LAUNCH_PARAM_BUFFER_SIZE": [
        "CU_LAUNCH_PARAM_BUFFER_SIZE"
    ],
    "HIP_LAUNCH_PARAM_END": [
        "CU_LAUNCH_PARAM_END"
    ],
    "hipHostMallocMapped": [
        "CU_MEMHOSTALLOC_DEVICEMAP",
        "cudaHostAllocMapped"
    ],
    "hipHostMallocPortable": [
        "CU_MEMHOSTALLOC_PORTABLE",
        "cudaHostAllocPortable"
    ],
    "hipHostMallocWriteCombined": [
        "CU_MEMHOSTALLOC_WRITECOMBINED",
        "cudaHostAllocWriteCombined"
    ],
    "hipHostRegisterMapped": [
        "CU_MEMHOSTREGISTER_DEVICEMAP",
        "cudaHostRegisterMapped"
    ],
    "hipHostRegisterIoMemory": [
        "CU_MEMHOSTREGISTER_IOMEMORY",
        "cudaHostRegisterIoMemory"
    ],
    "hipHostRegisterPortable": [
        "CU_MEMHOSTREGISTER_PORTABLE",
        "cudaHostRegisterPortable"
    ],
    "hipStreamPerThread": [
        "CU_STREAM_PER_THREAD",
        "cudaStreamPerThread"
    ],
    "HIP_TRSA_OVERRIDE_FORMAT": [
        "CU_TRSA_OVERRIDE_FORMAT"
    ],
    "HIP_TRSF_NORMALIZED_COORDINATES": [
        "CU_TRSF_NORMALIZED_COORDINATES"
    ],
    "HIP_TRSF_READ_AS_INTEGER": [
        "CU_TRSF_READ_AS_INTEGER"
    ],
    "HIP_TRSF_SRGB": [
        "CU_TRSF_SRGB"
    ],
    "HipcubDebug": [
        "CubDebug"
    ],
    "REGISTER_HIP_OPERATOR": [
        "REGISTER_CUDA_OPERATOR"
    ],
    "REGISTER_HIP_OPERATOR_CREATOR": [
        "REGISTER_CUDA_OPERATOR_CREATOR"
    ],
    "_HipcubLog": [
        "_CubLog"
    ],
    "__HIPCUB_ALIGN_BYTES": [
        "__CUB_ALIGN_BYTES"
    ],
    "__HIPCC__": [
        "__CUDACC__"
    ],
    "hipArrayDefault": [
        "cudaArrayDefault"
    ],
    "hipHostMallocDefault": [
        "cudaHostAllocDefault"
    ],
    "hipHostRegisterDefault": [
        "cudaHostRegisterDefault"
    ],
    "hipTextureType1D": [
        "cudaTextureType1D"
    ],
    "hipTextureType1DLayered": [
        "cudaTextureType1DLayered"
    ],
    "hipTextureType2D": [
        "cudaTextureType2D"
    ],
    "hipTextureType2DLayered": [
        "cudaTextureType2DLayered"
    ],
    "hipTextureType3D": [
        "cudaTextureType3D"
    ],
    "hipTextureTypeCubemap": [
        "cudaTextureTypeCubemap"
    ],
    "hipTextureTypeCubemapLayered": [
        "cudaTextureTypeCubemapLayered"
    ]
}

def get_bool_environ_var(env_var, default):
    yes_vals = ("true", "1", "t", "y", "yes")
    no_vals = ("false", "0", "f", "n", "no")
    value = os.environ.get(env_var, default).lower()
    if value in yes_vals:
        return True
    elif value in no_vals:
        return False
    else:
        allowed_vals = ", ".join([f"'{a}'" for a in (list(yes_vals)+list(no_vals))])
        raise RuntimeError(f"value of '{env_var}' must be one of (case-insensitive): {allowed_vals}")

accept_cuda_enum_names = get_bool_environ_var("HIP_PYTHON_ENUM_ACCEPT_CUDA_NAMES","true") # Accept CUDA enum names in addition to HIP ones
generate_fake_enums = get_bool_environ_var("HIP_PYTHON_ENUM_HALLUCINATE_MEMBERS","false")  # Hallucinate fake enum constants if member does not exist

def _get_hip_name(cuda_name):
    global cuda2hip
    return cuda2hip.get(cuda_name,None)

class FakeEnumType():
    """Mimicks the orginal enum type this 
    is derived from.
    """
    
    def __init__(self):
        pass
    
    @property
    def name(self):
        return self._name_

    @property
    def value(self):
        return self._value_
        
    def __eq__(self,other):
        if isinstance(other,self._orig_enum_type_):
            return self.value == other.value
        return False
    
    @property
    def __class__(self):
        """Overwrite __class__ to satisfy __isinstance__ check.
        """
        return self._orig_enum_type_

    def __repr__(self):        
        """Mimicks enum.Enum.__repr__"""
        return "<%s.%s: %r>" % (
                self.__class__.__name__, self._name_, self._value_)
                
    def __str__(self):
        """Mimicks enum.Enum.__str__"""
        return "%s.%s" % (self.__class__.__name__, self._name_)

class _EnumMeta(enum.EnumMeta):
        
    def __getattribute__(cls,name):
        global _get_hip_name
        global accept_cuda_enum_names
        global generate_fake_enums
        try:
            return super().__getattribute__(name)
        except AttributeError as ae:
            if not accept_cuda_enum_names:
                raise ae
            hip_name = _get_hip_name(name)
            if hip_name != None:
                return super().__getattribute__(hip_name)
            elif not generate_fake_enums:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1
                enum_types = list(cls._member_map_.values())
                enum_class = enum_types[0].__class__
                new = type(
                    name, 
                    (FakeEnumType,),
                    {"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}
                )()
                return new
        
class IntEnum(enum.IntEnum,metaclass=_EnumMeta):
    pass
