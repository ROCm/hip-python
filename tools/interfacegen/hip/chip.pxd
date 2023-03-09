#AMD_COPYRIGHT

ctypedef struct __hip_python_helper_type_66:
    hipArray_t array
ctypedef struct __hip_python_helper_type_67:
    hipMipmappedArray_t mipmap
ctypedef struct __hip_python_helper_type_68:
    pass
ctypedef struct __hip_python_helper_type_69:
    pass
ctypedef union __hip_python_helper_type_70:
    pass
ctypedef struct __hip_python_helper_type_71:
    hipArray_t hArray
ctypedef struct __hip_python_helper_type_72:
    hipMipmappedArray_t hMipmappedArray
ctypedef struct __hip_python_helper_type_73:
    pass
ctypedef struct __hip_python_helper_type_74:
    pass
ctypedef struct __hip_python_helper_type_75:
    int[32] reserved
ctypedef union __hip_python_helper_type_76:
    pass
ctypedef struct __hip_python_helper_type_66:
    hipArray_t array
ctypedef struct __hip_python_helper_type_67:
    hipMipmappedArray_t mipmap
ctypedef struct __hip_python_helper_type_68:
    pass
ctypedef struct __hip_python_helper_type_69:
    pass
ctypedef union __hip_python_helper_type_70:
    pass
ctypedef struct __hip_python_helper_type_71:
    hipArray_t hArray
ctypedef struct __hip_python_helper_type_72:
    hipMipmappedArray_t hMipmappedArray
ctypedef struct __hip_python_helper_type_73:
    pass
ctypedef struct __hip_python_helper_type_74:
    pass
ctypedef struct __hip_python_helper_type_75:
    int[32] reserved
ctypedef union __hip_python_helper_type_76:
    pass
ctypedef struct __hip_python_helper_type_66:
    hipArray_t array
ctypedef struct __hip_python_helper_type_67:
    hipMipmappedArray_t mipmap
ctypedef struct __hip_python_helper_type_68:
    pass
ctypedef struct __hip_python_helper_type_69:
    pass
ctypedef union __hip_python_helper_type_70:
    pass
ctypedef struct __hip_python_helper_type_71:
    hipArray_t hArray
ctypedef struct __hip_python_helper_type_72:
    hipMipmappedArray_t hMipmappedArray
ctypedef struct __hip_python_helper_type_73:
    pass
ctypedef struct __hip_python_helper_type_74:
    pass
ctypedef struct __hip_python_helper_type_75:
    int[32] reserved
ctypedef union __hip_python_helper_type_76:
    pass
ctypedef struct __hip_python_helper_type_66:
    hipArray_t array
ctypedef struct __hip_python_helper_type_67:
    hipMipmappedArray_t mipmap
ctypedef struct __hip_python_helper_type_68:
    pass
ctypedef struct __hip_python_helper_type_69:
    pass
ctypedef union __hip_python_helper_type_70:
    pass
ctypedef struct __hip_python_helper_type_71:
    hipArray_t hArray
ctypedef struct __hip_python_helper_type_72:
    hipMipmappedArray_t hMipmappedArray
ctypedef struct __hip_python_helper_type_73:
    pass
ctypedef struct __hip_python_helper_type_74:
    pass
ctypedef struct __hip_python_helper_type_75:
    int[32] reserved
ctypedef union __hip_python_helper_type_76:
    pass
ctypedef struct __hip_python_helper_type_77:
    void * handle
    const void * name
ctypedef union __hip_python_helper_type_78:
    int fd
    struct (unnamed struct at /opt/rocm/include/hip/hip_runtime_api.h:963:5) win32
ctypedef struct __hip_python_helper_type_79:
    void * handle
    const void * name
ctypedef union __hip_python_helper_type_80:
    int fd
    struct (unnamed struct at /opt/rocm/include/hip/hip_runtime_api.h:987:5) win32
ctypedef struct __hip_python_helper_type_81:
    unsigned long long value
ctypedef struct __hip_python_helper_type_82:
    unsigned long long key
ctypedef struct __hip_python_helper_type_83:
    struct (unnamed struct at /opt/rocm/include/hip/hip_runtime_api.h:997:5) fence
    struct (unnamed struct at /opt/rocm/include/hip/hip_runtime_api.h:1000:5) keyedMutex
    unsigned int[12] reserved
ctypedef struct __hip_python_helper_type_84:
    unsigned long long value
ctypedef struct __hip_python_helper_type_85:
    unsigned long long key
    unsigned int timeoutMs
ctypedef struct __hip_python_helper_type_86:
    struct (unnamed struct at /opt/rocm/include/hip/hip_runtime_api.h:1013:5) fence
    struct (unnamed struct at /opt/rocm/include/hip/hip_runtime_api.h:1016:5) keyedMutex
    unsigned int[10] reserved
ctypedef struct __hip_python_helper_type_87:
    unsigned char compressionType
    unsigned char gpuDirectRDMACapable
    unsigned short usage
ctypedef union __hip_python_helper_type_88:
    hipMipmappedArray mipmap
    hipArray_t array
ctypedef struct __hip_python_helper_type_89:
    unsigned int level
    unsigned int layer
    unsigned int offsetX
    unsigned int offsetY
    unsigned int offsetZ
    unsigned int extentWidth
    unsigned int extentHeight
    unsigned int extentDepth
ctypedef struct __hip_python_helper_type_90:
    unsigned int layer
    unsigned long long offset
    unsigned long long size
ctypedef union __hip_python_helper_type_91:
    struct (unnamed struct at /opt/rocm/include/hip/hip_runtime_api.h:1281:10) sparseLevel
    struct (unnamed struct at /opt/rocm/include/hip/hip_runtime_api.h:1291:10) miptail
ctypedef union __hip_python_helper_type_92:
    hipMemGenericAllocationHandle_t memHandle
cdef extern from "hip/driver_types.h":
    cdef int HIP_INCLUDE_HIP_DRIVER_TYPES_H
    cdef int HIP_INCLUDE_HIP_HIP_COMMON_H
    cdef int HIP_PUBLIC_API
    cdef int HIP_INTERNAL_EXPORTED_API
    cdef int HIP_INCLUDE_HIP_HCC_DETAIL_DRIVER_TYPES_H
    cdef int HIP_TRSA_OVERRIDE_FORMAT
    cdef int HIP_TRSF_READ_AS_INTEGER
    cdef int HIP_TRSF_NORMALIZED_COORDINATES
    cdef int HIP_TRSF_SRGB
    ctypedef hipDeviceptr_t hipDeviceptr_t
    cdef enum hipChannelFormatKind:
        hipChannelFormatKindSigned
        hipChannelFormatKindUnsigned
        hipChannelFormatKindFloat
        hipChannelFormatKindNone
    ctypedef hipChannelFormatKind hipChannelFormatKind
    cdef struct hipChannelFormatDesc:
        int x
        int y
        int z
        int w
        enum hipChannelFormatKind f
    ctypedef hipChannelFormatDesc hipChannelFormatDesc
    cdef enum hipArray_Format:
        HIP_AD_FORMAT_UNSIGNED_INT8
        HIP_AD_FORMAT_UNSIGNED_INT16
        HIP_AD_FORMAT_UNSIGNED_INT32
        HIP_AD_FORMAT_SIGNED_INT8
        HIP_AD_FORMAT_SIGNED_INT16
        HIP_AD_FORMAT_SIGNED_INT32
        HIP_AD_FORMAT_HALF
        HIP_AD_FORMAT_FLOAT
    ctypedef hipArray_Format hipArray_Format
    cdef struct HIP_ARRAY_DESCRIPTOR:
        pass
    ctypedef HIP_ARRAY_DESCRIPTOR HIP_ARRAY_DESCRIPTOR
    cdef struct HIP_ARRAY3D_DESCRIPTOR:
        pass
    ctypedef HIP_ARRAY3D_DESCRIPTOR HIP_ARRAY3D_DESCRIPTOR
    cdef struct hipArray:
        pass
    ctypedef hipArray hipArray
    cdef struct hip_Memcpy2D:
        pass
    ctypedef hip_Memcpy2D hip_Memcpy2D
    ctypedef hipArray_t hipArray_t
    ctypedef hiparray hiparray
    ctypedef hipArray_const_t hipArray_const_t
    cdef struct hipMipmappedArray:
        void * data
        struct hipChannelFormatDesc desc
        unsigned int type
        unsigned int width
        unsigned int height
        unsigned int depth
        unsigned int min_mipmap_level
        unsigned int max_mipmap_level
        unsigned int flags
        enum hipArray_Format format
    ctypedef hipMipmappedArray hipMipmappedArray
    ctypedef hipMipmappedArray_t hipMipmappedArray_t
    ctypedef hipMipmappedArray_const_t hipMipmappedArray_const_t
    cdef enum hipResourceType:
        hipResourceTypeArray
        hipResourceTypeMipmappedArray
        hipResourceTypeLinear
        hipResourceTypePitch2D
    ctypedef hipResourceType hipResourceType
    cdef enum HIPresourcetype_enum:
        HIP_RESOURCE_TYPE_ARRAY
        HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY
        HIP_RESOURCE_TYPE_LINEAR
        HIP_RESOURCE_TYPE_PITCH2D
    ctypedef HIPresourcetype_enum HIPresourcetype
    ctypedef HIPresourcetype_enum hipResourcetype
    cdef enum HIPaddress_mode_enum:
        HIP_TR_ADDRESS_MODE_WRAP
        HIP_TR_ADDRESS_MODE_CLAMP
        HIP_TR_ADDRESS_MODE_MIRROR
        HIP_TR_ADDRESS_MODE_BORDER
    ctypedef HIPaddress_mode_enum HIPaddress_mode
    cdef enum HIPfilter_mode_enum:
        HIP_TR_FILTER_MODE_POINT
        HIP_TR_FILTER_MODE_LINEAR
    ctypedef HIPfilter_mode_enum HIPfilter_mode
    cdef struct HIP_TEXTURE_DESC_st:
        HIPaddress_mode[3] addressMode
        HIPfilter_mode filterMode
        unsigned int flags
        unsigned int maxAnisotropy
        HIPfilter_mode mipmapFilterMode
        float mipmapLevelBias
        float minMipmapLevelClamp
        float maxMipmapLevelClamp
        float[4] borderColor
        int[12] reserved
    ctypedef HIP_TEXTURE_DESC_st HIP_TEXTURE_DESC
    cdef enum hipResourceViewFormat:
        hipResViewFormatNone
        hipResViewFormatUnsignedChar1
        hipResViewFormatUnsignedChar2
        hipResViewFormatUnsignedChar4
        hipResViewFormatSignedChar1
        hipResViewFormatSignedChar2
        hipResViewFormatSignedChar4
        hipResViewFormatUnsignedShort1
        hipResViewFormatUnsignedShort2
        hipResViewFormatUnsignedShort4
        hipResViewFormatSignedShort1
        hipResViewFormatSignedShort2
        hipResViewFormatSignedShort4
        hipResViewFormatUnsignedInt1
        hipResViewFormatUnsignedInt2
        hipResViewFormatUnsignedInt4
        hipResViewFormatSignedInt1
        hipResViewFormatSignedInt2
        hipResViewFormatSignedInt4
        hipResViewFormatHalf1
        hipResViewFormatHalf2
        hipResViewFormatHalf4
        hipResViewFormatFloat1
        hipResViewFormatFloat2
        hipResViewFormatFloat4
        hipResViewFormatUnsignedBlockCompressed1
        hipResViewFormatUnsignedBlockCompressed2
        hipResViewFormatUnsignedBlockCompressed3
        hipResViewFormatUnsignedBlockCompressed4
        hipResViewFormatSignedBlockCompressed4
        hipResViewFormatUnsignedBlockCompressed5
        hipResViewFormatSignedBlockCompressed5
        hipResViewFormatUnsignedBlockCompressed6H
        hipResViewFormatSignedBlockCompressed6H
        hipResViewFormatUnsignedBlockCompressed7
    ctypedef hipResourceViewFormat hipResourceViewFormat
    cdef enum HIPresourceViewFormat_enum:
        HIP_RES_VIEW_FORMAT_NONE
        HIP_RES_VIEW_FORMAT_UINT_1X8
        HIP_RES_VIEW_FORMAT_UINT_2X8
        HIP_RES_VIEW_FORMAT_UINT_4X8
        HIP_RES_VIEW_FORMAT_SINT_1X8
        HIP_RES_VIEW_FORMAT_SINT_2X8
        HIP_RES_VIEW_FORMAT_SINT_4X8
        HIP_RES_VIEW_FORMAT_UINT_1X16
        HIP_RES_VIEW_FORMAT_UINT_2X16
        HIP_RES_VIEW_FORMAT_UINT_4X16
        HIP_RES_VIEW_FORMAT_SINT_1X16
        HIP_RES_VIEW_FORMAT_SINT_2X16
        HIP_RES_VIEW_FORMAT_SINT_4X16
        HIP_RES_VIEW_FORMAT_UINT_1X32
        HIP_RES_VIEW_FORMAT_UINT_2X32
        HIP_RES_VIEW_FORMAT_UINT_4X32
        HIP_RES_VIEW_FORMAT_SINT_1X32
        HIP_RES_VIEW_FORMAT_SINT_2X32
        HIP_RES_VIEW_FORMAT_SINT_4X32
        HIP_RES_VIEW_FORMAT_FLOAT_1X16
        HIP_RES_VIEW_FORMAT_FLOAT_2X16
        HIP_RES_VIEW_FORMAT_FLOAT_4X16
        HIP_RES_VIEW_FORMAT_FLOAT_1X32
        HIP_RES_VIEW_FORMAT_FLOAT_2X32
        HIP_RES_VIEW_FORMAT_FLOAT_4X32
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC1
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC2
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC3
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC4
        HIP_RES_VIEW_FORMAT_SIGNED_BC4
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC5
        HIP_RES_VIEW_FORMAT_SIGNED_BC5
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H
        HIP_RES_VIEW_FORMAT_SIGNED_BC6H
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC7
    ctypedef HIPresourceViewFormat_enum HIPresourceViewFormat
    cdef struct hipResourceDesc:
        pass
    ctypedef hipResourceDesc hipResourceDesc
    cdef struct HIP_RESOURCE_DESC_st:
        pass
    ctypedef HIP_RESOURCE_DESC_st HIP_RESOURCE_DESC
    cdef struct hipResourceViewDesc:
        pass
    cdef struct HIP_RESOURCE_VIEW_DESC_st:
        pass
    ctypedef HIP_RESOURCE_VIEW_DESC_st HIP_RESOURCE_VIEW_DESC
    cdef enum hipMemcpyKind:
        hipMemcpyHostToHost
        hipMemcpyHostToDevice
        hipMemcpyDeviceToHost
        hipMemcpyDeviceToDevice
        hipMemcpyDefault
    ctypedef hipMemcpyKind hipMemcpyKind
    cdef struct hipPitchedPtr:
        pass
    ctypedef hipPitchedPtr hipPitchedPtr
    cdef struct hipExtent:
        pass
    ctypedef hipExtent hipExtent
    cdef struct hipPos:
        pass
    ctypedef hipPos hipPos
    cdef struct hipMemcpy3DParms:
        pass
    ctypedef hipMemcpy3DParms hipMemcpy3DParms
    cdef struct HIP_MEMCPY3D:
        pass
    ctypedef HIP_MEMCPY3D HIP_MEMCPY3D
    cdef enum hipFunction_attribute:
        HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
        HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
        HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
        HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
        HIP_FUNC_ATTRIBUTE_NUM_REGS
        HIP_FUNC_ATTRIBUTE_PTX_VERSION
        HIP_FUNC_ATTRIBUTE_BINARY_VERSION
        HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA
        HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
        HIP_FUNC_ATTRIBUTE_MAX
    ctypedef hipFunction_attribute hipFunction_attribute
    cdef enum hipPointer_attribute:
        HIP_POINTER_ATTRIBUTE_CONTEXT
        HIP_POINTER_ATTRIBUTE_MEMORY_TYPE
        HIP_POINTER_ATTRIBUTE_DEVICE_POINTER
        HIP_POINTER_ATTRIBUTE_HOST_POINTER
        HIP_POINTER_ATTRIBUTE_P2P_TOKENS
        HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS
        HIP_POINTER_ATTRIBUTE_BUFFER_ID
        HIP_POINTER_ATTRIBUTE_IS_MANAGED
        HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL
        HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE
        HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR
        HIP_POINTER_ATTRIBUTE_RANGE_SIZE
        HIP_POINTER_ATTRIBUTE_MAPPED
        HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES
        HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE
        HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS
        HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE
    ctypedef hipPointer_attribute hipPointer_attribute
cdef extern from "hip/surface_types.h":
    cdef int HIP_INCLUDE_HIP_SURFACE_TYPES_H
    cdef int HIP_INCLUDE_HIP_DRIVER_TYPES_H
    cdef int HIP_INCLUDE_HIP_HIP_COMMON_H
    cdef int HIP_PUBLIC_API
    cdef int HIP_INTERNAL_EXPORTED_API
    cdef int HIP_INCLUDE_HIP_HCC_DETAIL_DRIVER_TYPES_H
    cdef int HIP_TRSA_OVERRIDE_FORMAT
    cdef int HIP_TRSF_READ_AS_INTEGER
    cdef int HIP_TRSF_NORMALIZED_COORDINATES
    cdef int HIP_TRSF_SRGB
    ctypedef hipDeviceptr_t hipDeviceptr_t
    cdef enum hipChannelFormatKind:
        hipChannelFormatKindSigned
        hipChannelFormatKindUnsigned
        hipChannelFormatKindFloat
        hipChannelFormatKindNone
    ctypedef hipChannelFormatKind hipChannelFormatKind
    cdef struct hipChannelFormatDesc:
        int x
        int y
        int z
        int w
        enum hipChannelFormatKind f
    ctypedef hipChannelFormatDesc hipChannelFormatDesc
    cdef enum hipArray_Format:
        HIP_AD_FORMAT_UNSIGNED_INT8
        HIP_AD_FORMAT_UNSIGNED_INT16
        HIP_AD_FORMAT_UNSIGNED_INT32
        HIP_AD_FORMAT_SIGNED_INT8
        HIP_AD_FORMAT_SIGNED_INT16
        HIP_AD_FORMAT_SIGNED_INT32
        HIP_AD_FORMAT_HALF
        HIP_AD_FORMAT_FLOAT
    ctypedef hipArray_Format hipArray_Format
    cdef struct HIP_ARRAY_DESCRIPTOR:
        pass
    ctypedef HIP_ARRAY_DESCRIPTOR HIP_ARRAY_DESCRIPTOR
    cdef struct HIP_ARRAY3D_DESCRIPTOR:
        pass
    ctypedef HIP_ARRAY3D_DESCRIPTOR HIP_ARRAY3D_DESCRIPTOR
    cdef struct hipArray:
        pass
    ctypedef hipArray hipArray
    cdef struct hip_Memcpy2D:
        pass
    ctypedef hip_Memcpy2D hip_Memcpy2D
    ctypedef hipArray_t hipArray_t
    ctypedef hiparray hiparray
    ctypedef hipArray_const_t hipArray_const_t
    cdef struct hipMipmappedArray:
        void * data
        struct hipChannelFormatDesc desc
        unsigned int type
        unsigned int width
        unsigned int height
        unsigned int depth
        unsigned int min_mipmap_level
        unsigned int max_mipmap_level
        unsigned int flags
        enum hipArray_Format format
    ctypedef hipMipmappedArray hipMipmappedArray
    ctypedef hipMipmappedArray_t hipMipmappedArray_t
    ctypedef hipMipmappedArray_const_t hipMipmappedArray_const_t
    cdef enum hipResourceType:
        hipResourceTypeArray
        hipResourceTypeMipmappedArray
        hipResourceTypeLinear
        hipResourceTypePitch2D
    ctypedef hipResourceType hipResourceType
    cdef enum HIPresourcetype_enum:
        HIP_RESOURCE_TYPE_ARRAY
        HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY
        HIP_RESOURCE_TYPE_LINEAR
        HIP_RESOURCE_TYPE_PITCH2D
    ctypedef HIPresourcetype_enum HIPresourcetype
    ctypedef HIPresourcetype_enum hipResourcetype
    cdef enum HIPaddress_mode_enum:
        HIP_TR_ADDRESS_MODE_WRAP
        HIP_TR_ADDRESS_MODE_CLAMP
        HIP_TR_ADDRESS_MODE_MIRROR
        HIP_TR_ADDRESS_MODE_BORDER
    ctypedef HIPaddress_mode_enum HIPaddress_mode
    cdef enum HIPfilter_mode_enum:
        HIP_TR_FILTER_MODE_POINT
        HIP_TR_FILTER_MODE_LINEAR
    ctypedef HIPfilter_mode_enum HIPfilter_mode
    cdef struct HIP_TEXTURE_DESC_st:
        HIPaddress_mode[3] addressMode
        HIPfilter_mode filterMode
        unsigned int flags
        unsigned int maxAnisotropy
        HIPfilter_mode mipmapFilterMode
        float mipmapLevelBias
        float minMipmapLevelClamp
        float maxMipmapLevelClamp
        float[4] borderColor
        int[12] reserved
    ctypedef HIP_TEXTURE_DESC_st HIP_TEXTURE_DESC
    cdef enum hipResourceViewFormat:
        hipResViewFormatNone
        hipResViewFormatUnsignedChar1
        hipResViewFormatUnsignedChar2
        hipResViewFormatUnsignedChar4
        hipResViewFormatSignedChar1
        hipResViewFormatSignedChar2
        hipResViewFormatSignedChar4
        hipResViewFormatUnsignedShort1
        hipResViewFormatUnsignedShort2
        hipResViewFormatUnsignedShort4
        hipResViewFormatSignedShort1
        hipResViewFormatSignedShort2
        hipResViewFormatSignedShort4
        hipResViewFormatUnsignedInt1
        hipResViewFormatUnsignedInt2
        hipResViewFormatUnsignedInt4
        hipResViewFormatSignedInt1
        hipResViewFormatSignedInt2
        hipResViewFormatSignedInt4
        hipResViewFormatHalf1
        hipResViewFormatHalf2
        hipResViewFormatHalf4
        hipResViewFormatFloat1
        hipResViewFormatFloat2
        hipResViewFormatFloat4
        hipResViewFormatUnsignedBlockCompressed1
        hipResViewFormatUnsignedBlockCompressed2
        hipResViewFormatUnsignedBlockCompressed3
        hipResViewFormatUnsignedBlockCompressed4
        hipResViewFormatSignedBlockCompressed4
        hipResViewFormatUnsignedBlockCompressed5
        hipResViewFormatSignedBlockCompressed5
        hipResViewFormatUnsignedBlockCompressed6H
        hipResViewFormatSignedBlockCompressed6H
        hipResViewFormatUnsignedBlockCompressed7
    ctypedef hipResourceViewFormat hipResourceViewFormat
    cdef enum HIPresourceViewFormat_enum:
        HIP_RES_VIEW_FORMAT_NONE
        HIP_RES_VIEW_FORMAT_UINT_1X8
        HIP_RES_VIEW_FORMAT_UINT_2X8
        HIP_RES_VIEW_FORMAT_UINT_4X8
        HIP_RES_VIEW_FORMAT_SINT_1X8
        HIP_RES_VIEW_FORMAT_SINT_2X8
        HIP_RES_VIEW_FORMAT_SINT_4X8
        HIP_RES_VIEW_FORMAT_UINT_1X16
        HIP_RES_VIEW_FORMAT_UINT_2X16
        HIP_RES_VIEW_FORMAT_UINT_4X16
        HIP_RES_VIEW_FORMAT_SINT_1X16
        HIP_RES_VIEW_FORMAT_SINT_2X16
        HIP_RES_VIEW_FORMAT_SINT_4X16
        HIP_RES_VIEW_FORMAT_UINT_1X32
        HIP_RES_VIEW_FORMAT_UINT_2X32
        HIP_RES_VIEW_FORMAT_UINT_4X32
        HIP_RES_VIEW_FORMAT_SINT_1X32
        HIP_RES_VIEW_FORMAT_SINT_2X32
        HIP_RES_VIEW_FORMAT_SINT_4X32
        HIP_RES_VIEW_FORMAT_FLOAT_1X16
        HIP_RES_VIEW_FORMAT_FLOAT_2X16
        HIP_RES_VIEW_FORMAT_FLOAT_4X16
        HIP_RES_VIEW_FORMAT_FLOAT_1X32
        HIP_RES_VIEW_FORMAT_FLOAT_2X32
        HIP_RES_VIEW_FORMAT_FLOAT_4X32
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC1
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC2
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC3
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC4
        HIP_RES_VIEW_FORMAT_SIGNED_BC4
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC5
        HIP_RES_VIEW_FORMAT_SIGNED_BC5
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H
        HIP_RES_VIEW_FORMAT_SIGNED_BC6H
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC7
    ctypedef HIPresourceViewFormat_enum HIPresourceViewFormat
    cdef struct hipResourceDesc:
        pass
    ctypedef hipResourceDesc hipResourceDesc
    cdef struct HIP_RESOURCE_DESC_st:
        pass
    ctypedef HIP_RESOURCE_DESC_st HIP_RESOURCE_DESC
    cdef struct hipResourceViewDesc:
        pass
    cdef struct HIP_RESOURCE_VIEW_DESC_st:
        pass
    ctypedef HIP_RESOURCE_VIEW_DESC_st HIP_RESOURCE_VIEW_DESC
    cdef enum hipMemcpyKind:
        hipMemcpyHostToHost
        hipMemcpyHostToDevice
        hipMemcpyDeviceToHost
        hipMemcpyDeviceToDevice
        hipMemcpyDefault
    ctypedef hipMemcpyKind hipMemcpyKind
    cdef struct hipPitchedPtr:
        pass
    ctypedef hipPitchedPtr hipPitchedPtr
    cdef struct hipExtent:
        pass
    ctypedef hipExtent hipExtent
    cdef struct hipPos:
        pass
    ctypedef hipPos hipPos
    cdef struct hipMemcpy3DParms:
        pass
    ctypedef hipMemcpy3DParms hipMemcpy3DParms
    cdef struct HIP_MEMCPY3D:
        pass
    ctypedef HIP_MEMCPY3D HIP_MEMCPY3D
    cdef enum hipFunction_attribute:
        HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
        HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
        HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
        HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
        HIP_FUNC_ATTRIBUTE_NUM_REGS
        HIP_FUNC_ATTRIBUTE_PTX_VERSION
        HIP_FUNC_ATTRIBUTE_BINARY_VERSION
        HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA
        HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
        HIP_FUNC_ATTRIBUTE_MAX
    ctypedef hipFunction_attribute hipFunction_attribute
    cdef enum hipPointer_attribute:
        HIP_POINTER_ATTRIBUTE_CONTEXT
        HIP_POINTER_ATTRIBUTE_MEMORY_TYPE
        HIP_POINTER_ATTRIBUTE_DEVICE_POINTER
        HIP_POINTER_ATTRIBUTE_HOST_POINTER
        HIP_POINTER_ATTRIBUTE_P2P_TOKENS
        HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS
        HIP_POINTER_ATTRIBUTE_BUFFER_ID
        HIP_POINTER_ATTRIBUTE_IS_MANAGED
        HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL
        HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE
        HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR
        HIP_POINTER_ATTRIBUTE_RANGE_SIZE
        HIP_POINTER_ATTRIBUTE_MAPPED
        HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES
        HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE
        HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS
        HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE
    ctypedef hipPointer_attribute hipPointer_attribute
    ctypedef hipSurfaceObject_t hipSurfaceObject_t
    cdef enum hipSurfaceBoundaryMode:
        hipBoundaryModeZero
        hipBoundaryModeTrap
        hipBoundaryModeClamp
cdef extern from "hip/texture_types.h":
    cdef int HIP_INCLUDE_HIP_TEXTURE_TYPES_H
    cdef int HIP_INCLUDE_HIP_HIP_COMMON_H
    cdef int HIP_PUBLIC_API
    cdef int HIP_INTERNAL_EXPORTED_API
    cdef int HIP_INCLUDE_HIP_CHANNEL_DESCRIPTOR_H
    cdef int HIP_INCLUDE_HIP_AMD_DETAIL_CHANNEL_DESCRIPTOR_H
    cdef int HIP_INCLUDE_HIP_DRIVER_TYPES_H
    cdef int HIP_INCLUDE_HIP_HCC_DETAIL_DRIVER_TYPES_H
    cdef int HIP_TRSA_OVERRIDE_FORMAT
    cdef int HIP_TRSF_READ_AS_INTEGER
    cdef int HIP_TRSF_NORMALIZED_COORDINATES
    cdef int HIP_TRSF_SRGB
    cdef int HIP_INCLUDE_HIP_AMD_DETAIL_HIP_VECTOR_TYPES_H
    cdef int HIP_INCLUDE_HIP_AMD_DETAIL_HOST_DEFINES_H
    cdef int HIP_INCLUDE_HIP_HCC_DETAIL_HOST_DEFINES_H
    cdef int hipTextureType1D
    cdef int hipTextureType2D
    cdef int hipTextureType3D
    cdef int hipTextureTypeCubemap
    cdef int hipTextureType1DLayered
    cdef int hipTextureType2DLayered
    cdef int hipTextureTypeCubemapLayered
    cdef int HIP_IMAGE_OBJECT_SIZE_DWORD
    cdef int HIP_SAMPLER_OBJECT_SIZE_DWORD
    cdef int HIP_SAMPLER_OBJECT_OFFSET_DWORD
    cdef int HIP_TEXTURE_OBJECT_SIZE_DWORD
    ctypedef hipDeviceptr_t hipDeviceptr_t
    cdef enum hipChannelFormatKind:
        hipChannelFormatKindSigned
        hipChannelFormatKindUnsigned
        hipChannelFormatKindFloat
        hipChannelFormatKindNone
    ctypedef hipChannelFormatKind hipChannelFormatKind
    cdef struct hipChannelFormatDesc:
        int x
        int y
        int z
        int w
        enum hipChannelFormatKind f
    ctypedef hipChannelFormatDesc hipChannelFormatDesc
    cdef enum hipArray_Format:
        HIP_AD_FORMAT_UNSIGNED_INT8
        HIP_AD_FORMAT_UNSIGNED_INT16
        HIP_AD_FORMAT_UNSIGNED_INT32
        HIP_AD_FORMAT_SIGNED_INT8
        HIP_AD_FORMAT_SIGNED_INT16
        HIP_AD_FORMAT_SIGNED_INT32
        HIP_AD_FORMAT_HALF
        HIP_AD_FORMAT_FLOAT
    ctypedef hipArray_Format hipArray_Format
    cdef struct HIP_ARRAY_DESCRIPTOR:
        pass
    ctypedef HIP_ARRAY_DESCRIPTOR HIP_ARRAY_DESCRIPTOR
    cdef struct HIP_ARRAY3D_DESCRIPTOR:
        pass
    ctypedef HIP_ARRAY3D_DESCRIPTOR HIP_ARRAY3D_DESCRIPTOR
    cdef struct hipArray:
        pass
    ctypedef hipArray hipArray
    cdef struct hip_Memcpy2D:
        pass
    ctypedef hip_Memcpy2D hip_Memcpy2D
    ctypedef hipArray_t hipArray_t
    ctypedef hiparray hiparray
    ctypedef hipArray_const_t hipArray_const_t
    cdef struct hipMipmappedArray:
        void * data
        struct hipChannelFormatDesc desc
        unsigned int type
        unsigned int width
        unsigned int height
        unsigned int depth
        unsigned int min_mipmap_level
        unsigned int max_mipmap_level
        unsigned int flags
        enum hipArray_Format format
    ctypedef hipMipmappedArray hipMipmappedArray
    ctypedef hipMipmappedArray_t hipMipmappedArray_t
    ctypedef hipMipmappedArray_const_t hipMipmappedArray_const_t
    cdef enum hipResourceType:
        hipResourceTypeArray
        hipResourceTypeMipmappedArray
        hipResourceTypeLinear
        hipResourceTypePitch2D
    ctypedef hipResourceType hipResourceType
    cdef enum HIPresourcetype_enum:
        HIP_RESOURCE_TYPE_ARRAY
        HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY
        HIP_RESOURCE_TYPE_LINEAR
        HIP_RESOURCE_TYPE_PITCH2D
    ctypedef HIPresourcetype_enum HIPresourcetype
    ctypedef HIPresourcetype_enum hipResourcetype
    cdef enum HIPaddress_mode_enum:
        HIP_TR_ADDRESS_MODE_WRAP
        HIP_TR_ADDRESS_MODE_CLAMP
        HIP_TR_ADDRESS_MODE_MIRROR
        HIP_TR_ADDRESS_MODE_BORDER
    ctypedef HIPaddress_mode_enum HIPaddress_mode
    cdef enum HIPfilter_mode_enum:
        HIP_TR_FILTER_MODE_POINT
        HIP_TR_FILTER_MODE_LINEAR
    ctypedef HIPfilter_mode_enum HIPfilter_mode
    cdef struct HIP_TEXTURE_DESC_st:
        HIPaddress_mode[3] addressMode
        HIPfilter_mode filterMode
        unsigned int flags
        unsigned int maxAnisotropy
        HIPfilter_mode mipmapFilterMode
        float mipmapLevelBias
        float minMipmapLevelClamp
        float maxMipmapLevelClamp
        float[4] borderColor
        int[12] reserved
    ctypedef HIP_TEXTURE_DESC_st HIP_TEXTURE_DESC
    cdef enum hipResourceViewFormat:
        hipResViewFormatNone
        hipResViewFormatUnsignedChar1
        hipResViewFormatUnsignedChar2
        hipResViewFormatUnsignedChar4
        hipResViewFormatSignedChar1
        hipResViewFormatSignedChar2
        hipResViewFormatSignedChar4
        hipResViewFormatUnsignedShort1
        hipResViewFormatUnsignedShort2
        hipResViewFormatUnsignedShort4
        hipResViewFormatSignedShort1
        hipResViewFormatSignedShort2
        hipResViewFormatSignedShort4
        hipResViewFormatUnsignedInt1
        hipResViewFormatUnsignedInt2
        hipResViewFormatUnsignedInt4
        hipResViewFormatSignedInt1
        hipResViewFormatSignedInt2
        hipResViewFormatSignedInt4
        hipResViewFormatHalf1
        hipResViewFormatHalf2
        hipResViewFormatHalf4
        hipResViewFormatFloat1
        hipResViewFormatFloat2
        hipResViewFormatFloat4
        hipResViewFormatUnsignedBlockCompressed1
        hipResViewFormatUnsignedBlockCompressed2
        hipResViewFormatUnsignedBlockCompressed3
        hipResViewFormatUnsignedBlockCompressed4
        hipResViewFormatSignedBlockCompressed4
        hipResViewFormatUnsignedBlockCompressed5
        hipResViewFormatSignedBlockCompressed5
        hipResViewFormatUnsignedBlockCompressed6H
        hipResViewFormatSignedBlockCompressed6H
        hipResViewFormatUnsignedBlockCompressed7
    ctypedef hipResourceViewFormat hipResourceViewFormat
    cdef enum HIPresourceViewFormat_enum:
        HIP_RES_VIEW_FORMAT_NONE
        HIP_RES_VIEW_FORMAT_UINT_1X8
        HIP_RES_VIEW_FORMAT_UINT_2X8
        HIP_RES_VIEW_FORMAT_UINT_4X8
        HIP_RES_VIEW_FORMAT_SINT_1X8
        HIP_RES_VIEW_FORMAT_SINT_2X8
        HIP_RES_VIEW_FORMAT_SINT_4X8
        HIP_RES_VIEW_FORMAT_UINT_1X16
        HIP_RES_VIEW_FORMAT_UINT_2X16
        HIP_RES_VIEW_FORMAT_UINT_4X16
        HIP_RES_VIEW_FORMAT_SINT_1X16
        HIP_RES_VIEW_FORMAT_SINT_2X16
        HIP_RES_VIEW_FORMAT_SINT_4X16
        HIP_RES_VIEW_FORMAT_UINT_1X32
        HIP_RES_VIEW_FORMAT_UINT_2X32
        HIP_RES_VIEW_FORMAT_UINT_4X32
        HIP_RES_VIEW_FORMAT_SINT_1X32
        HIP_RES_VIEW_FORMAT_SINT_2X32
        HIP_RES_VIEW_FORMAT_SINT_4X32
        HIP_RES_VIEW_FORMAT_FLOAT_1X16
        HIP_RES_VIEW_FORMAT_FLOAT_2X16
        HIP_RES_VIEW_FORMAT_FLOAT_4X16
        HIP_RES_VIEW_FORMAT_FLOAT_1X32
        HIP_RES_VIEW_FORMAT_FLOAT_2X32
        HIP_RES_VIEW_FORMAT_FLOAT_4X32
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC1
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC2
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC3
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC4
        HIP_RES_VIEW_FORMAT_SIGNED_BC4
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC5
        HIP_RES_VIEW_FORMAT_SIGNED_BC5
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H
        HIP_RES_VIEW_FORMAT_SIGNED_BC6H
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC7
    ctypedef HIPresourceViewFormat_enum HIPresourceViewFormat
    cdef struct hipResourceDesc:
        pass
    ctypedef hipResourceDesc hipResourceDesc
    cdef struct HIP_RESOURCE_DESC_st:
        pass
    ctypedef HIP_RESOURCE_DESC_st HIP_RESOURCE_DESC
    cdef struct hipResourceViewDesc:
        pass
    cdef struct HIP_RESOURCE_VIEW_DESC_st:
        pass
    ctypedef HIP_RESOURCE_VIEW_DESC_st HIP_RESOURCE_VIEW_DESC
    cdef enum hipMemcpyKind:
        hipMemcpyHostToHost
        hipMemcpyHostToDevice
        hipMemcpyDeviceToHost
        hipMemcpyDeviceToDevice
        hipMemcpyDefault
    ctypedef hipMemcpyKind hipMemcpyKind
    cdef struct hipPitchedPtr:
        pass
    ctypedef hipPitchedPtr hipPitchedPtr
    cdef struct hipExtent:
        pass
    ctypedef hipExtent hipExtent
    cdef struct hipPos:
        pass
    ctypedef hipPos hipPos
    cdef struct hipMemcpy3DParms:
        pass
    ctypedef hipMemcpy3DParms hipMemcpy3DParms
    cdef struct HIP_MEMCPY3D:
        pass
    ctypedef HIP_MEMCPY3D HIP_MEMCPY3D
    cdef enum hipFunction_attribute:
        HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
        HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
        HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
        HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
        HIP_FUNC_ATTRIBUTE_NUM_REGS
        HIP_FUNC_ATTRIBUTE_PTX_VERSION
        HIP_FUNC_ATTRIBUTE_BINARY_VERSION
        HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA
        HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
        HIP_FUNC_ATTRIBUTE_MAX
    ctypedef hipFunction_attribute hipFunction_attribute
    cdef enum hipPointer_attribute:
        HIP_POINTER_ATTRIBUTE_CONTEXT
        HIP_POINTER_ATTRIBUTE_MEMORY_TYPE
        HIP_POINTER_ATTRIBUTE_DEVICE_POINTER
        HIP_POINTER_ATTRIBUTE_HOST_POINTER
        HIP_POINTER_ATTRIBUTE_P2P_TOKENS
        HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS
        HIP_POINTER_ATTRIBUTE_BUFFER_ID
        HIP_POINTER_ATTRIBUTE_IS_MANAGED
        HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL
        HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE
        HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR
        HIP_POINTER_ATTRIBUTE_RANGE_SIZE
        HIP_POINTER_ATTRIBUTE_MAPPED
        HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES
        HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE
        HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS
        HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE
    ctypedef hipPointer_attribute hipPointer_attribute
    struct hipChannelFormatDesc hipCreateChannelDesc(int x,int y,int z,int w,enum hipChannelFormatKind f) nogil
    ctypedef hipTextureObject_t hipTextureObject_t
    cdef enum hipTextureAddressMode:
        hipAddressModeWrap
        hipAddressModeClamp
        hipAddressModeMirror
        hipAddressModeBorder
    cdef enum hipTextureFilterMode:
        hipFilterModePoint
        hipFilterModeLinear
    cdef enum hipTextureReadMode:
        hipReadModeElementType
        hipReadModeNormalizedFloat
    cdef struct hipTextureDesc:
        enum hipTextureAddressMode[3] addressMode
        enum hipTextureFilterMode filterMode
        enum hipTextureReadMode readMode
        int sRGB
        float[4] borderColor
        int normalizedCoords
        unsigned int maxAnisotropy
        enum hipTextureFilterMode mipmapFilterMode
        float mipmapLevelBias
        float minMipmapLevelClamp
        float maxMipmapLevelClamp
    ctypedef hipTextureDesc hipTextureDesc
cdef extern from "hip/library_types.h":
    cdef int HIP_INCLUDE_HIP_LIBRARY_TYPES_H
    cdef int HIP_INCLUDE_HIP_HIP_COMMON_H
    cdef int HIP_PUBLIC_API
    cdef int HIP_INTERNAL_EXPORTED_API
    cdef enum hipDataType:
        HIP_R_16F
        HIP_R_32F
        HIP_R_64F
        HIP_C_16F
        HIP_C_32F
        HIP_C_64F
    ctypedef hipDataType hipDataType
    cdef enum hipLibraryPropertyType:
        HIP_LIBRARY_MAJOR_VERSION
        HIP_LIBRARY_MINOR_VERSION
        HIP_LIBRARY_PATCH_LEVEL
    ctypedef hipLibraryPropertyType hipLibraryPropertyType
cdef extern from "hip/hip_runtime_api.h":
    cdef int HIP_INCLUDE_HIP_HIP_RUNTIME_API_H
    cdef int HIP_VERSION_H
    cdef int HIP_VERSION_MAJOR
    cdef int HIP_VERSION_MINOR
    cdef int HIP_VERSION_PATCH
    cdef int HIP_VERSION_GITHASH
    cdef int HIP_VERSION_BUILD_ID
    cdef int HIP_VERSION_BUILD_NAME
    cdef int HIP_VERSION
    cdef int HIP_INCLUDE_HIP_HIP_COMMON_H
    cdef int HIP_PUBLIC_API
    cdef int HIP_INTERNAL_EXPORTED_API
    cdef int HIP_INCLUDE_HIP_AMD_DETAIL_HOST_DEFINES_H
    cdef int HIP_INCLUDE_HIP_HCC_DETAIL_HOST_DEFINES_H
    cdef int HIP_INCLUDE_HIP_DRIVER_TYPES_H
    cdef int HIP_INCLUDE_HIP_HCC_DETAIL_DRIVER_TYPES_H
    cdef int HIP_TRSA_OVERRIDE_FORMAT
    cdef int HIP_TRSF_READ_AS_INTEGER
    cdef int HIP_TRSF_NORMALIZED_COORDINATES
    cdef int HIP_TRSF_SRGB
    cdef int HIP_INCLUDE_HIP_TEXTURE_TYPES_H
    cdef int HIP_INCLUDE_HIP_CHANNEL_DESCRIPTOR_H
    cdef int HIP_INCLUDE_HIP_AMD_DETAIL_CHANNEL_DESCRIPTOR_H
    cdef int HIP_INCLUDE_HIP_AMD_DETAIL_HIP_VECTOR_TYPES_H
    cdef int hipTextureType1D
    cdef int hipTextureType2D
    cdef int hipTextureType3D
    cdef int hipTextureTypeCubemap
    cdef int hipTextureType1DLayered
    cdef int hipTextureType2DLayered
    cdef int hipTextureTypeCubemapLayered
    cdef int HIP_IMAGE_OBJECT_SIZE_DWORD
    cdef int HIP_SAMPLER_OBJECT_SIZE_DWORD
    cdef int HIP_SAMPLER_OBJECT_OFFSET_DWORD
    cdef int HIP_TEXTURE_OBJECT_SIZE_DWORD
    cdef int HIP_INCLUDE_HIP_SURFACE_TYPES_H
    cdef int HIP_LAUNCH_PARAM_BUFFER_POINTER
    cdef int HIP_LAUNCH_PARAM_BUFFER_SIZE
    cdef int HIP_LAUNCH_PARAM_END
    cdef int hipIpcMemLazyEnablePeerAccess
    cdef int HIP_IPC_HANDLE_SIZE
    cdef int hipStreamDefault
    cdef int hipStreamNonBlocking
    cdef int hipEventDefault
    cdef int hipEventBlockingSync
    cdef int hipEventDisableTiming
    cdef int hipEventInterprocess
    cdef int hipEventReleaseToDevice
    cdef int hipEventReleaseToSystem
    cdef int hipHostMallocDefault
    cdef int hipHostMallocPortable
    cdef int hipHostMallocMapped
    cdef int hipHostMallocWriteCombined
    cdef int hipHostMallocNumaUser
    cdef int hipHostMallocCoherent
    cdef int hipHostMallocNonCoherent
    cdef int hipMemAttachGlobal
    cdef int hipMemAttachHost
    cdef int hipMemAttachSingle
    cdef int hipDeviceMallocDefault
    cdef int hipDeviceMallocFinegrained
    cdef int hipMallocSignalMemory
    cdef int hipHostRegisterDefault
    cdef int hipHostRegisterPortable
    cdef int hipHostRegisterMapped
    cdef int hipHostRegisterIoMemory
    cdef int hipExtHostRegisterCoarseGrained
    cdef int hipDeviceScheduleAuto
    cdef int hipDeviceScheduleSpin
    cdef int hipDeviceScheduleYield
    cdef int hipDeviceScheduleBlockingSync
    cdef int hipDeviceScheduleMask
    cdef int hipDeviceMapHost
    cdef int hipDeviceLmemResizeToMax
    cdef int hipArrayDefault
    cdef int hipArrayLayered
    cdef int hipArraySurfaceLoadStore
    cdef int hipArrayCubemap
    cdef int hipArrayTextureGather
    cdef int hipOccupancyDefault
    cdef int hipCooperativeLaunchMultiDeviceNoPreSync
    cdef int hipCooperativeLaunchMultiDeviceNoPostSync
    cdef int hipCpuDeviceId
    cdef int hipInvalidDeviceId
    cdef int hipExtAnyOrderLaunch
    cdef int hipStreamWaitValueGte
    cdef int hipStreamWaitValueEq
    cdef int hipStreamWaitValueAnd
    cdef int hipStreamWaitValueNor
    cdef int hipStreamPerThread
    cdef int HIP_INCLUDE_HIP_HIP_RUNTIME_PT_API_H
    ctypedef __hip_python_helper_type_65 hipDeviceArch_t
    cdef struct hipUUID_t:
        char[16] bytes
    ctypedef hipUUID_t hipUUID
    cdef struct hipDeviceProp_t:
        pass
    ctypedef hipDeviceProp_t hipDeviceProp_t
    cdef enum hipMemoryType:
        hipMemoryTypeHost
        hipMemoryTypeDevice
        hipMemoryTypeArray
        hipMemoryTypeUnified
        hipMemoryTypeManaged
    ctypedef hipMemoryType hipMemoryType
    cdef struct hipPointerAttribute_t:
        enum hipMemoryType memoryType
        int device
        void * devicePointer
        void * hostPointer
        int isManaged
        unsigned int allocationFlags
    ctypedef hipPointerAttribute_t hipPointerAttribute_t
    cdef enum hipError_t:
        hipSuccess
        hipErrorInvalidValue
        hipErrorOutOfMemory
        hipErrorMemoryAllocation
        hipErrorNotInitialized
        hipErrorInitializationError
        hipErrorDeinitialized
        hipErrorProfilerDisabled
        hipErrorProfilerNotInitialized
        hipErrorProfilerAlreadyStarted
        hipErrorProfilerAlreadyStopped
        hipErrorInvalidConfiguration
        hipErrorInvalidPitchValue
        hipErrorInvalidSymbol
        hipErrorInvalidDevicePointer
        hipErrorInvalidMemcpyDirection
        hipErrorInsufficientDriver
        hipErrorMissingConfiguration
        hipErrorPriorLaunchFailure
        hipErrorInvalidDeviceFunction
        hipErrorNoDevice
        hipErrorInvalidDevice
        hipErrorInvalidImage
        hipErrorInvalidContext
        hipErrorContextAlreadyCurrent
        hipErrorMapFailed
        hipErrorMapBufferObjectFailed
        hipErrorUnmapFailed
        hipErrorArrayIsMapped
        hipErrorAlreadyMapped
        hipErrorNoBinaryForGpu
        hipErrorAlreadyAcquired
        hipErrorNotMapped
        hipErrorNotMappedAsArray
        hipErrorNotMappedAsPointer
        hipErrorECCNotCorrectable
        hipErrorUnsupportedLimit
        hipErrorContextAlreadyInUse
        hipErrorPeerAccessUnsupported
        hipErrorInvalidKernelFile
        hipErrorInvalidGraphicsContext
        hipErrorInvalidSource
        hipErrorFileNotFound
        hipErrorSharedObjectSymbolNotFound
        hipErrorSharedObjectInitFailed
        hipErrorOperatingSystem
        hipErrorInvalidHandle
        hipErrorInvalidResourceHandle
        hipErrorIllegalState
        hipErrorNotFound
        hipErrorNotReady
        hipErrorIllegalAddress
        hipErrorLaunchOutOfResources
        hipErrorLaunchTimeOut
        hipErrorPeerAccessAlreadyEnabled
        hipErrorPeerAccessNotEnabled
        hipErrorSetOnActiveProcess
        hipErrorContextIsDestroyed
        hipErrorAssert
        hipErrorHostMemoryAlreadyRegistered
        hipErrorHostMemoryNotRegistered
        hipErrorLaunchFailure
        hipErrorCooperativeLaunchTooLarge
        hipErrorNotSupported
        hipErrorStreamCaptureUnsupported
        hipErrorStreamCaptureInvalidated
        hipErrorStreamCaptureMerge
        hipErrorStreamCaptureUnmatched
        hipErrorStreamCaptureUnjoined
        hipErrorStreamCaptureIsolation
        hipErrorStreamCaptureImplicit
        hipErrorCapturedEvent
        hipErrorStreamCaptureWrongThread
        hipErrorGraphExecUpdateFailure
        hipErrorUnknown
        hipErrorRuntimeMemory
        hipErrorRuntimeOther
        hipErrorTbd
    ctypedef hipError_t hipError_t
    cdef enum hipDeviceAttribute_t:
        hipDeviceAttributeCudaCompatibleBegin
        hipDeviceAttributeEccEnabled
        hipDeviceAttributeAccessPolicyMaxWindowSize
        hipDeviceAttributeAsyncEngineCount
        hipDeviceAttributeCanMapHostMemory
        hipDeviceAttributeCanUseHostPointerForRegisteredMem
        hipDeviceAttributeClockRate
        hipDeviceAttributeComputeMode
        hipDeviceAttributeComputePreemptionSupported
        hipDeviceAttributeConcurrentKernels
        hipDeviceAttributeConcurrentManagedAccess
        hipDeviceAttributeCooperativeLaunch
        hipDeviceAttributeCooperativeMultiDeviceLaunch
        hipDeviceAttributeDeviceOverlap
        hipDeviceAttributeDirectManagedMemAccessFromHost
        hipDeviceAttributeGlobalL1CacheSupported
        hipDeviceAttributeHostNativeAtomicSupported
        hipDeviceAttributeIntegrated
        hipDeviceAttributeIsMultiGpuBoard
        hipDeviceAttributeKernelExecTimeout
        hipDeviceAttributeL2CacheSize
        hipDeviceAttributeLocalL1CacheSupported
        hipDeviceAttributeLuid
        hipDeviceAttributeLuidDeviceNodeMask
        hipDeviceAttributeComputeCapabilityMajor
        hipDeviceAttributeManagedMemory
        hipDeviceAttributeMaxBlocksPerMultiProcessor
        hipDeviceAttributeMaxBlockDimX
        hipDeviceAttributeMaxBlockDimY
        hipDeviceAttributeMaxBlockDimZ
        hipDeviceAttributeMaxGridDimX
        hipDeviceAttributeMaxGridDimY
        hipDeviceAttributeMaxGridDimZ
        hipDeviceAttributeMaxSurface1D
        hipDeviceAttributeMaxSurface1DLayered
        hipDeviceAttributeMaxSurface2D
        hipDeviceAttributeMaxSurface2DLayered
        hipDeviceAttributeMaxSurface3D
        hipDeviceAttributeMaxSurfaceCubemap
        hipDeviceAttributeMaxSurfaceCubemapLayered
        hipDeviceAttributeMaxTexture1DWidth
        hipDeviceAttributeMaxTexture1DLayered
        hipDeviceAttributeMaxTexture1DLinear
        hipDeviceAttributeMaxTexture1DMipmap
        hipDeviceAttributeMaxTexture2DWidth
        hipDeviceAttributeMaxTexture2DHeight
        hipDeviceAttributeMaxTexture2DGather
        hipDeviceAttributeMaxTexture2DLayered
        hipDeviceAttributeMaxTexture2DLinear
        hipDeviceAttributeMaxTexture2DMipmap
        hipDeviceAttributeMaxTexture3DWidth
        hipDeviceAttributeMaxTexture3DHeight
        hipDeviceAttributeMaxTexture3DDepth
        hipDeviceAttributeMaxTexture3DAlt
        hipDeviceAttributeMaxTextureCubemap
        hipDeviceAttributeMaxTextureCubemapLayered
        hipDeviceAttributeMaxThreadsDim
        hipDeviceAttributeMaxThreadsPerBlock
        hipDeviceAttributeMaxThreadsPerMultiProcessor
        hipDeviceAttributeMaxPitch
        hipDeviceAttributeMemoryBusWidth
        hipDeviceAttributeMemoryClockRate
        hipDeviceAttributeComputeCapabilityMinor
        hipDeviceAttributeMultiGpuBoardGroupID
        hipDeviceAttributeMultiprocessorCount
        hipDeviceAttributeName
        hipDeviceAttributePageableMemoryAccess
        hipDeviceAttributePageableMemoryAccessUsesHostPageTables
        hipDeviceAttributePciBusId
        hipDeviceAttributePciDeviceId
        hipDeviceAttributePciDomainID
        hipDeviceAttributePersistingL2CacheMaxSize
        hipDeviceAttributeMaxRegistersPerBlock
        hipDeviceAttributeMaxRegistersPerMultiprocessor
        hipDeviceAttributeReservedSharedMemPerBlock
        hipDeviceAttributeMaxSharedMemoryPerBlock
        hipDeviceAttributeSharedMemPerBlockOptin
        hipDeviceAttributeSharedMemPerMultiprocessor
        hipDeviceAttributeSingleToDoublePrecisionPerfRatio
        hipDeviceAttributeStreamPrioritiesSupported
        hipDeviceAttributeSurfaceAlignment
        hipDeviceAttributeTccDriver
        hipDeviceAttributeTextureAlignment
        hipDeviceAttributeTexturePitchAlignment
        hipDeviceAttributeTotalConstantMemory
        hipDeviceAttributeTotalGlobalMem
        hipDeviceAttributeUnifiedAddressing
        hipDeviceAttributeUuid
        hipDeviceAttributeWarpSize
        hipDeviceAttributeMemoryPoolsSupported
        hipDeviceAttributeVirtualMemoryManagementSupported
        hipDeviceAttributeCudaCompatibleEnd
        hipDeviceAttributeAmdSpecificBegin
        hipDeviceAttributeClockInstructionRate
        hipDeviceAttributeArch
        hipDeviceAttributeMaxSharedMemoryPerMultiprocessor
        hipDeviceAttributeGcnArch
        hipDeviceAttributeGcnArchName
        hipDeviceAttributeHdpMemFlushCntl
        hipDeviceAttributeHdpRegFlushCntl
        hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc
        hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim
        hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim
        hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem
        hipDeviceAttributeIsLargeBar
        hipDeviceAttributeAsicRevision
        hipDeviceAttributeCanUseStreamWaitValue
        hipDeviceAttributeImageSupport
        hipDeviceAttributePhysicalMultiProcessorCount
        hipDeviceAttributeFineGrainSupport
        hipDeviceAttributeWallClockRate
        hipDeviceAttributeAmdSpecificEnd
        hipDeviceAttributeVendorSpecificBegin
    ctypedef hipDeviceAttribute_t hipDeviceAttribute_t
    cdef enum hipComputeMode:
        hipComputeModeDefault
        hipComputeModeExclusive
        hipComputeModeProhibited
        hipComputeModeExclusiveProcess
    ctypedef hipDeviceptr_t hipDeviceptr_t
    cdef enum hipChannelFormatKind:
        hipChannelFormatKindSigned
        hipChannelFormatKindUnsigned
        hipChannelFormatKindFloat
        hipChannelFormatKindNone
    ctypedef hipChannelFormatKind hipChannelFormatKind
    cdef struct hipChannelFormatDesc:
        int x
        int y
        int z
        int w
        enum hipChannelFormatKind f
    ctypedef hipChannelFormatDesc hipChannelFormatDesc
    cdef enum hipArray_Format:
        HIP_AD_FORMAT_UNSIGNED_INT8
        HIP_AD_FORMAT_UNSIGNED_INT16
        HIP_AD_FORMAT_UNSIGNED_INT32
        HIP_AD_FORMAT_SIGNED_INT8
        HIP_AD_FORMAT_SIGNED_INT16
        HIP_AD_FORMAT_SIGNED_INT32
        HIP_AD_FORMAT_HALF
        HIP_AD_FORMAT_FLOAT
    ctypedef hipArray_Format hipArray_Format
    cdef struct HIP_ARRAY_DESCRIPTOR:
        pass
    ctypedef HIP_ARRAY_DESCRIPTOR HIP_ARRAY_DESCRIPTOR
    cdef struct HIP_ARRAY3D_DESCRIPTOR:
        pass
    ctypedef HIP_ARRAY3D_DESCRIPTOR HIP_ARRAY3D_DESCRIPTOR
    cdef struct hipArray:
        pass
    ctypedef hipArray hipArray
    cdef struct hip_Memcpy2D:
        pass
    ctypedef hip_Memcpy2D hip_Memcpy2D
    ctypedef hipArray_t hipArray_t
    ctypedef hiparray hiparray
    ctypedef hipArray_const_t hipArray_const_t
    cdef struct hipMipmappedArray:
        void * data
        struct hipChannelFormatDesc desc
        unsigned int type
        unsigned int width
        unsigned int height
        unsigned int depth
        unsigned int min_mipmap_level
        unsigned int max_mipmap_level
        unsigned int flags
        enum hipArray_Format format
    ctypedef hipMipmappedArray hipMipmappedArray
    ctypedef hipMipmappedArray_t hipMipmappedArray_t
    ctypedef hipMipmappedArray_const_t hipMipmappedArray_const_t
    cdef enum hipResourceType:
        hipResourceTypeArray
        hipResourceTypeMipmappedArray
        hipResourceTypeLinear
        hipResourceTypePitch2D
    ctypedef hipResourceType hipResourceType
    cdef enum HIPresourcetype_enum:
        HIP_RESOURCE_TYPE_ARRAY
        HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY
        HIP_RESOURCE_TYPE_LINEAR
        HIP_RESOURCE_TYPE_PITCH2D
    ctypedef HIPresourcetype_enum HIPresourcetype
    ctypedef HIPresourcetype_enum hipResourcetype
    cdef enum HIPaddress_mode_enum:
        HIP_TR_ADDRESS_MODE_WRAP
        HIP_TR_ADDRESS_MODE_CLAMP
        HIP_TR_ADDRESS_MODE_MIRROR
        HIP_TR_ADDRESS_MODE_BORDER
    ctypedef HIPaddress_mode_enum HIPaddress_mode
    cdef enum HIPfilter_mode_enum:
        HIP_TR_FILTER_MODE_POINT
        HIP_TR_FILTER_MODE_LINEAR
    ctypedef HIPfilter_mode_enum HIPfilter_mode
    cdef struct HIP_TEXTURE_DESC_st:
        HIPaddress_mode[3] addressMode
        HIPfilter_mode filterMode
        unsigned int flags
        unsigned int maxAnisotropy
        HIPfilter_mode mipmapFilterMode
        float mipmapLevelBias
        float minMipmapLevelClamp
        float maxMipmapLevelClamp
        float[4] borderColor
        int[12] reserved
    ctypedef HIP_TEXTURE_DESC_st HIP_TEXTURE_DESC
    cdef enum hipResourceViewFormat:
        hipResViewFormatNone
        hipResViewFormatUnsignedChar1
        hipResViewFormatUnsignedChar2
        hipResViewFormatUnsignedChar4
        hipResViewFormatSignedChar1
        hipResViewFormatSignedChar2
        hipResViewFormatSignedChar4
        hipResViewFormatUnsignedShort1
        hipResViewFormatUnsignedShort2
        hipResViewFormatUnsignedShort4
        hipResViewFormatSignedShort1
        hipResViewFormatSignedShort2
        hipResViewFormatSignedShort4
        hipResViewFormatUnsignedInt1
        hipResViewFormatUnsignedInt2
        hipResViewFormatUnsignedInt4
        hipResViewFormatSignedInt1
        hipResViewFormatSignedInt2
        hipResViewFormatSignedInt4
        hipResViewFormatHalf1
        hipResViewFormatHalf2
        hipResViewFormatHalf4
        hipResViewFormatFloat1
        hipResViewFormatFloat2
        hipResViewFormatFloat4
        hipResViewFormatUnsignedBlockCompressed1
        hipResViewFormatUnsignedBlockCompressed2
        hipResViewFormatUnsignedBlockCompressed3
        hipResViewFormatUnsignedBlockCompressed4
        hipResViewFormatSignedBlockCompressed4
        hipResViewFormatUnsignedBlockCompressed5
        hipResViewFormatSignedBlockCompressed5
        hipResViewFormatUnsignedBlockCompressed6H
        hipResViewFormatSignedBlockCompressed6H
        hipResViewFormatUnsignedBlockCompressed7
    ctypedef hipResourceViewFormat hipResourceViewFormat
    cdef enum HIPresourceViewFormat_enum:
        HIP_RES_VIEW_FORMAT_NONE
        HIP_RES_VIEW_FORMAT_UINT_1X8
        HIP_RES_VIEW_FORMAT_UINT_2X8
        HIP_RES_VIEW_FORMAT_UINT_4X8
        HIP_RES_VIEW_FORMAT_SINT_1X8
        HIP_RES_VIEW_FORMAT_SINT_2X8
        HIP_RES_VIEW_FORMAT_SINT_4X8
        HIP_RES_VIEW_FORMAT_UINT_1X16
        HIP_RES_VIEW_FORMAT_UINT_2X16
        HIP_RES_VIEW_FORMAT_UINT_4X16
        HIP_RES_VIEW_FORMAT_SINT_1X16
        HIP_RES_VIEW_FORMAT_SINT_2X16
        HIP_RES_VIEW_FORMAT_SINT_4X16
        HIP_RES_VIEW_FORMAT_UINT_1X32
        HIP_RES_VIEW_FORMAT_UINT_2X32
        HIP_RES_VIEW_FORMAT_UINT_4X32
        HIP_RES_VIEW_FORMAT_SINT_1X32
        HIP_RES_VIEW_FORMAT_SINT_2X32
        HIP_RES_VIEW_FORMAT_SINT_4X32
        HIP_RES_VIEW_FORMAT_FLOAT_1X16
        HIP_RES_VIEW_FORMAT_FLOAT_2X16
        HIP_RES_VIEW_FORMAT_FLOAT_4X16
        HIP_RES_VIEW_FORMAT_FLOAT_1X32
        HIP_RES_VIEW_FORMAT_FLOAT_2X32
        HIP_RES_VIEW_FORMAT_FLOAT_4X32
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC1
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC2
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC3
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC4
        HIP_RES_VIEW_FORMAT_SIGNED_BC4
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC5
        HIP_RES_VIEW_FORMAT_SIGNED_BC5
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H
        HIP_RES_VIEW_FORMAT_SIGNED_BC6H
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC7
    ctypedef HIPresourceViewFormat_enum HIPresourceViewFormat
    cdef struct hipResourceDesc:
        pass
    ctypedef hipResourceDesc hipResourceDesc
    cdef struct HIP_RESOURCE_DESC_st:
        pass
    ctypedef HIP_RESOURCE_DESC_st HIP_RESOURCE_DESC
    cdef struct hipResourceViewDesc:
        pass
    cdef struct HIP_RESOURCE_VIEW_DESC_st:
        pass
    ctypedef HIP_RESOURCE_VIEW_DESC_st HIP_RESOURCE_VIEW_DESC
    cdef enum hipMemcpyKind:
        hipMemcpyHostToHost
        hipMemcpyHostToDevice
        hipMemcpyDeviceToHost
        hipMemcpyDeviceToDevice
        hipMemcpyDefault
    ctypedef hipMemcpyKind hipMemcpyKind
    cdef struct hipPitchedPtr:
        pass
    ctypedef hipPitchedPtr hipPitchedPtr
    cdef struct hipExtent:
        pass
    ctypedef hipExtent hipExtent
    cdef struct hipPos:
        pass
    ctypedef hipPos hipPos
    cdef struct hipMemcpy3DParms:
        pass
    ctypedef hipMemcpy3DParms hipMemcpy3DParms
    cdef struct HIP_MEMCPY3D:
        unsigned int srcXInBytes
        unsigned int srcY
        unsigned int srcZ
        unsigned int srcLOD
        hipMemoryType srcMemoryType
        const void * srcHost
        hipDeviceptr_t srcDevice
        hipArray_t srcArray
        unsigned int srcPitch
        unsigned int srcHeight
        unsigned int dstXInBytes
        unsigned int dstY
        unsigned int dstZ
        unsigned int dstLOD
        hipMemoryType dstMemoryType
        void * dstHost
        hipDeviceptr_t dstDevice
        hipArray_t dstArray
        unsigned int dstPitch
        unsigned int dstHeight
        unsigned int WidthInBytes
        unsigned int Height
        unsigned int Depth
    ctypedef HIP_MEMCPY3D HIP_MEMCPY3D
    cdef enum hipFunction_attribute:
        HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
        HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
        HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
        HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
        HIP_FUNC_ATTRIBUTE_NUM_REGS
        HIP_FUNC_ATTRIBUTE_PTX_VERSION
        HIP_FUNC_ATTRIBUTE_BINARY_VERSION
        HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA
        HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
        HIP_FUNC_ATTRIBUTE_MAX
    ctypedef hipFunction_attribute hipFunction_attribute
    cdef enum hipPointer_attribute:
        HIP_POINTER_ATTRIBUTE_CONTEXT
        HIP_POINTER_ATTRIBUTE_MEMORY_TYPE
        HIP_POINTER_ATTRIBUTE_DEVICE_POINTER
        HIP_POINTER_ATTRIBUTE_HOST_POINTER
        HIP_POINTER_ATTRIBUTE_P2P_TOKENS
        HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS
        HIP_POINTER_ATTRIBUTE_BUFFER_ID
        HIP_POINTER_ATTRIBUTE_IS_MANAGED
        HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL
        HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE
        HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR
        HIP_POINTER_ATTRIBUTE_RANGE_SIZE
        HIP_POINTER_ATTRIBUTE_MAPPED
        HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES
        HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE
        HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS
        HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE
    ctypedef hipPointer_attribute hipPointer_attribute
    struct hipChannelFormatDesc hipCreateChannelDesc(int x,int y,int z,int w,enum hipChannelFormatKind f) nogil
    ctypedef hipTextureObject_t hipTextureObject_t
    cdef enum hipTextureAddressMode:
        hipAddressModeWrap
        hipAddressModeClamp
        hipAddressModeMirror
        hipAddressModeBorder
    cdef enum hipTextureFilterMode:
        hipFilterModePoint
        hipFilterModeLinear
    cdef enum hipTextureReadMode:
        hipReadModeElementType
        hipReadModeNormalizedFloat
    cdef struct hipTextureDesc:
        enum hipTextureAddressMode[3] addressMode
        enum hipTextureFilterMode filterMode
        enum hipTextureReadMode readMode
        int sRGB
        float[4] borderColor
        int normalizedCoords
        unsigned int maxAnisotropy
        enum hipTextureFilterMode mipmapFilterMode
        float mipmapLevelBias
        float minMipmapLevelClamp
        float maxMipmapLevelClamp
    ctypedef hipTextureDesc hipTextureDesc
    ctypedef hipSurfaceObject_t hipSurfaceObject_t
    cdef enum hipSurfaceBoundaryMode:
        hipBoundaryModeZero
        hipBoundaryModeTrap
        hipBoundaryModeClamp
    ctypedef hipCtx_t hipCtx_t
    ctypedef hipDevice_t hipDevice_t
    cdef enum hipDeviceP2PAttr:
        hipDevP2PAttrPerformanceRank
        hipDevP2PAttrAccessSupported
        hipDevP2PAttrNativeAtomicSupported
        hipDevP2PAttrHipArrayAccessSupported
    ctypedef hipDeviceP2PAttr hipDeviceP2PAttr
    ctypedef hipStream_t hipStream_t
    cdef struct hipIpcMemHandle_st:
        char[64] reserved
    ctypedef hipIpcMemHandle_st hipIpcMemHandle_t
    cdef struct hipIpcEventHandle_st:
        char[64] reserved
    ctypedef hipIpcEventHandle_st hipIpcEventHandle_t
    ctypedef hipModule_t hipModule_t
    ctypedef hipFunction_t hipFunction_t
    ctypedef hipMemPool_t hipMemPool_t
    cdef struct hipFuncAttributes:
        pass
    ctypedef hipFuncAttributes hipFuncAttributes
    ctypedef hipEvent_t hipEvent_t
    cdef enum hipLimit_t:
        hipLimitStackSize
        hipLimitPrintfFifoSize
        hipLimitMallocHeapSize
        hipLimitRange
    cdef enum hipMemoryAdvise:
        hipMemAdviseSetReadMostly
        hipMemAdviseUnsetReadMostly
        hipMemAdviseSetPreferredLocation
        hipMemAdviseUnsetPreferredLocation
        hipMemAdviseSetAccessedBy
        hipMemAdviseUnsetAccessedBy
        hipMemAdviseSetCoarseGrain
        hipMemAdviseUnsetCoarseGrain
    ctypedef hipMemoryAdvise hipMemoryAdvise
    cdef enum hipMemRangeCoherencyMode:
        hipMemRangeCoherencyModeFineGrain
        hipMemRangeCoherencyModeCoarseGrain
        hipMemRangeCoherencyModeIndeterminate
    ctypedef hipMemRangeCoherencyMode hipMemRangeCoherencyMode
    cdef enum hipMemRangeAttribute:
        hipMemRangeAttributeReadMostly
        hipMemRangeAttributePreferredLocation
        hipMemRangeAttributeAccessedBy
        hipMemRangeAttributeLastPrefetchLocation
        hipMemRangeAttributeCoherencyMode
    ctypedef hipMemRangeAttribute hipMemRangeAttribute
    cdef enum hipMemPoolAttr:
        hipMemPoolReuseFollowEventDependencies
        hipMemPoolReuseAllowOpportunistic
        hipMemPoolReuseAllowInternalDependencies
        hipMemPoolAttrReleaseThreshold
        hipMemPoolAttrReservedMemCurrent
        hipMemPoolAttrReservedMemHigh
        hipMemPoolAttrUsedMemCurrent
        hipMemPoolAttrUsedMemHigh
    ctypedef hipMemPoolAttr hipMemPoolAttr
    cdef enum hipMemLocationType:
        hipMemLocationTypeInvalid
        hipMemLocationTypeDevice
    ctypedef hipMemLocationType hipMemLocationType
    cdef struct hipMemLocation:
        hipMemLocationType type
        int id
    ctypedef hipMemLocation hipMemLocation
    cdef enum hipMemAccessFlags:
        hipMemAccessFlagsProtNone
        hipMemAccessFlagsProtRead
        hipMemAccessFlagsProtReadWrite
    ctypedef hipMemAccessFlags hipMemAccessFlags
    cdef struct hipMemAccessDesc:
        hipMemLocation location
        hipMemAccessFlags flags
    ctypedef hipMemAccessDesc hipMemAccessDesc
    cdef enum hipMemAllocationType:
        hipMemAllocationTypeInvalid
        hipMemAllocationTypePinned
        hipMemAllocationTypeMax
    ctypedef hipMemAllocationType hipMemAllocationType
    cdef enum hipMemAllocationHandleType:
        hipMemHandleTypeNone
        hipMemHandleTypePosixFileDescriptor
        hipMemHandleTypeWin32
        hipMemHandleTypeWin32Kmt
    ctypedef hipMemAllocationHandleType hipMemAllocationHandleType
    cdef struct hipMemPoolProps:
        hipMemAllocationType allocType
        hipMemAllocationHandleType handleTypes
        hipMemLocation location
        void * win32SecurityAttributes
        unsigned char[64] reserved
    ctypedef hipMemPoolProps hipMemPoolProps
    cdef struct hipMemPoolPtrExportData:
        unsigned char[64] reserved
    ctypedef hipMemPoolPtrExportData hipMemPoolPtrExportData
    cdef enum hipJitOption:
        hipJitOptionMaxRegisters
        hipJitOptionThreadsPerBlock
        hipJitOptionWallTime
        hipJitOptionInfoLogBuffer
        hipJitOptionInfoLogBufferSizeBytes
        hipJitOptionErrorLogBuffer
        hipJitOptionErrorLogBufferSizeBytes
        hipJitOptionOptimizationLevel
        hipJitOptionTargetFromContext
        hipJitOptionTarget
        hipJitOptionFallbackStrategy
        hipJitOptionGenerateDebugInfo
        hipJitOptionLogVerbose
        hipJitOptionGenerateLineInfo
        hipJitOptionCacheMode
        hipJitOptionSm3xOpt
        hipJitOptionFastCompile
        hipJitOptionNumOptions
    ctypedef hipJitOption hipJitOption
    cdef enum hipFuncAttribute:
        hipFuncAttributeMaxDynamicSharedMemorySize
        hipFuncAttributePreferredSharedMemoryCarveout
        hipFuncAttributeMax
    ctypedef hipFuncAttribute hipFuncAttribute
    cdef enum hipFuncCache_t:
        hipFuncCachePreferNone
        hipFuncCachePreferShared
        hipFuncCachePreferL1
        hipFuncCachePreferEqual
    ctypedef hipFuncCache_t hipFuncCache_t
    cdef enum hipSharedMemConfig:
        hipSharedMemBankSizeDefault
        hipSharedMemBankSizeFourByte
        hipSharedMemBankSizeEightByte
    ctypedef hipSharedMemConfig hipSharedMemConfig
    cdef struct hipLaunchParams_t:
        pass
    ctypedef hipLaunchParams_t hipLaunchParams
    cdef enum hipExternalMemoryHandleType_enum:
        hipExternalMemoryHandleTypeOpaqueFd
        hipExternalMemoryHandleTypeOpaqueWin32
        hipExternalMemoryHandleTypeOpaqueWin32Kmt
        hipExternalMemoryHandleTypeD3D12Heap
        hipExternalMemoryHandleTypeD3D12Resource
        hipExternalMemoryHandleTypeD3D11Resource
        hipExternalMemoryHandleTypeD3D11ResourceKmt
    ctypedef hipExternalMemoryHandleType_enum hipExternalMemoryHandleType
    cdef struct hipExternalMemoryHandleDesc_st:
        hipExternalMemoryHandleType type
        union (unnamed union at /opt/rocm/include/hip/hip_runtime_api.h:961:3) handle
        unsigned long long size
        unsigned int flags
    ctypedef hipExternalMemoryHandleDesc_st hipExternalMemoryHandleDesc
    cdef struct hipExternalMemoryBufferDesc_st:
        unsigned long long offset
        unsigned long long size
        unsigned int flags
    ctypedef hipExternalMemoryBufferDesc_st hipExternalMemoryBufferDesc
    ctypedef hipExternalMemory_t hipExternalMemory_t
    cdef enum hipExternalSemaphoreHandleType_enum:
        hipExternalSemaphoreHandleTypeOpaqueFd
        hipExternalSemaphoreHandleTypeOpaqueWin32
        hipExternalSemaphoreHandleTypeOpaqueWin32Kmt
        hipExternalSemaphoreHandleTypeD3D12Fence
    ctypedef hipExternalSemaphoreHandleType_enum hipExternalSemaphoreHandleType
    cdef struct hipExternalSemaphoreHandleDesc_st:
        hipExternalSemaphoreHandleType type
        union (unnamed union at /opt/rocm/include/hip/hip_runtime_api.h:985:3) handle
        unsigned int flags
    ctypedef hipExternalSemaphoreHandleDesc_st hipExternalSemaphoreHandleDesc
    ctypedef hipExternalSemaphore_t hipExternalSemaphore_t
    cdef struct hipExternalSemaphoreSignalParams_st:
        struct (unnamed struct at /opt/rocm/include/hip/hip_runtime_api.h:996:3) params
        unsigned int flags
        unsigned int[16] reserved
    ctypedef hipExternalSemaphoreSignalParams_st hipExternalSemaphoreSignalParams
    cdef struct hipExternalSemaphoreWaitParams_st:
        struct (unnamed struct at /opt/rocm/include/hip/hip_runtime_api.h:1012:3) params
        unsigned int flags
        unsigned int[16] reserved
    ctypedef hipExternalSemaphoreWaitParams_st hipExternalSemaphoreWaitParams
    cdef enum hipGLDeviceList:
        hipGLDeviceListAll
        hipGLDeviceListCurrentFrame
        hipGLDeviceListNextFrame
    ctypedef hipGLDeviceList hipGLDeviceList
    cdef enum hipGraphicsRegisterFlags:
        hipGraphicsRegisterFlagsNone
        hipGraphicsRegisterFlagsReadOnly
        hipGraphicsRegisterFlagsWriteDiscard
        hipGraphicsRegisterFlagsSurfaceLoadStore
        hipGraphicsRegisterFlagsTextureGather
    ctypedef hipGraphicsRegisterFlags hipGraphicsRegisterFlags
    ctypedef hipGraphicsResource hipGraphicsResource
    ctypedef hipGraphicsResource_t hipGraphicsResource_t
    ctypedef hipGraph_t hipGraph_t
    cdef struct hipGraphNode:
        pass
    ctypedef hipGraphNode_t hipGraphNode_t
    cdef struct hipGraphExec:
        pass
    ctypedef hipGraphExec_t hipGraphExec_t
    cdef struct hipUserObject:
        pass
    ctypedef hipUserObject_t hipUserObject_t
    cdef enum hipGraphNodeType:
        hipGraphNodeTypeKernel
        hipGraphNodeTypeMemcpy
        hipGraphNodeTypeMemset
        hipGraphNodeTypeHost
        hipGraphNodeTypeGraph
        hipGraphNodeTypeEmpty
        hipGraphNodeTypeWaitEvent
        hipGraphNodeTypeEventRecord
        hipGraphNodeTypeExtSemaphoreSignal
        hipGraphNodeTypeExtSemaphoreWait
        hipGraphNodeTypeMemcpyFromSymbol
        hipGraphNodeTypeMemcpyToSymbol
        hipGraphNodeTypeCount
    ctypedef hipGraphNodeType hipGraphNodeType
    ctypedef hipHostFn_t hipHostFn_t
    cdef struct hipHostNodeParams:
        hipHostFn_t fn
        void * userData
    ctypedef hipHostNodeParams hipHostNodeParams
    cdef struct hipKernelNodeParams:
        dim3 blockDim
        void ** extra
        void * func
        dim3 gridDim
        void ** kernelParams
        unsigned int sharedMemBytes
    ctypedef hipKernelNodeParams hipKernelNodeParams
    cdef struct hipMemsetParams:
        pass
    ctypedef hipMemsetParams hipMemsetParams
    cdef enum hipKernelNodeAttrID:
        hipKernelNodeAttributeAccessPolicyWindow
        hipKernelNodeAttributeCooperative
    ctypedef hipKernelNodeAttrID hipKernelNodeAttrID
    cdef enum hipAccessProperty:
        hipAccessPropertyNormal
        hipAccessPropertyStreaming
        hipAccessPropertyPersisting
    ctypedef hipAccessProperty hipAccessProperty
    cdef struct hipAccessPolicyWindow:
        pass
    ctypedef hipAccessPolicyWindow hipAccessPolicyWindow
    cdef union hipKernelNodeAttrValue:
        pass
    ctypedef hipKernelNodeAttrValue hipKernelNodeAttrValue
    cdef enum hipGraphExecUpdateResult:
        hipGraphExecUpdateSuccess
        hipGraphExecUpdateError
        hipGraphExecUpdateErrorTopologyChanged
        hipGraphExecUpdateErrorNodeTypeChanged
        hipGraphExecUpdateErrorFunctionChanged
        hipGraphExecUpdateErrorParametersChanged
        hipGraphExecUpdateErrorNotSupported
        hipGraphExecUpdateErrorUnsupportedFunctionChange
    ctypedef hipGraphExecUpdateResult hipGraphExecUpdateResult
    cdef enum hipStreamCaptureMode:
        hipStreamCaptureModeGlobal
        hipStreamCaptureModeThreadLocal
        hipStreamCaptureModeRelaxed
    ctypedef hipStreamCaptureMode hipStreamCaptureMode
    cdef enum hipStreamCaptureStatus:
        hipStreamCaptureStatusNone
        hipStreamCaptureStatusActive
        hipStreamCaptureStatusInvalidated
    ctypedef hipStreamCaptureStatus hipStreamCaptureStatus
    cdef enum hipStreamUpdateCaptureDependenciesFlags:
        hipStreamAddCaptureDependencies
        hipStreamSetCaptureDependencies
    ctypedef hipStreamUpdateCaptureDependenciesFlags hipStreamUpdateCaptureDependenciesFlags
    cdef enum hipGraphMemAttributeType:
        hipGraphMemAttrUsedMemCurrent
        hipGraphMemAttrUsedMemHigh
        hipGraphMemAttrReservedMemCurrent
        hipGraphMemAttrReservedMemHigh
    ctypedef hipGraphMemAttributeType hipGraphMemAttributeType
    cdef enum hipUserObjectFlags:
        hipUserObjectNoDestructorSync
    ctypedef hipUserObjectFlags hipUserObjectFlags
    cdef enum hipUserObjectRetainFlags:
        hipGraphUserObjectMove
    ctypedef hipUserObjectRetainFlags hipUserObjectRetainFlags
    cdef enum hipGraphInstantiateFlags:
        hipGraphInstantiateFlagAutoFreeOnLaunch
    ctypedef hipGraphInstantiateFlags hipGraphInstantiateFlags
    cdef struct hipMemAllocationProp:
        hipMemAllocationType type
        hipMemAllocationHandleType requestedHandleType
        hipMemLocation location
        void * win32HandleMetaData
        struct (unnamed struct at /opt/rocm/include/hip/hip_runtime_api.h:1219:5) allocFlags
    ctypedef hipMemAllocationProp hipMemAllocationProp
    ctypedef hipMemGenericAllocationHandle_t hipMemGenericAllocationHandle_t
    cdef enum hipMemAllocationGranularity_flags:
        hipMemAllocationGranularityMinimum
        hipMemAllocationGranularityRecommended
    ctypedef hipMemAllocationGranularity_flags hipMemAllocationGranularity_flags
    cdef enum hipMemHandleType:
        hipMemHandleTypeGeneric
    ctypedef hipMemHandleType hipMemHandleType
    cdef enum hipMemOperationType:
        hipMemOperationTypeMap
        hipMemOperationTypeUnmap
    ctypedef hipMemOperationType hipMemOperationType
    cdef enum hipArraySparseSubresourceType:
        hipArraySparseSubresourceTypeSparseLevel
        hipArraySparseSubresourceTypeMiptail
    ctypedef hipArraySparseSubresourceType hipArraySparseSubresourceType
    cdef struct hipArrayMapInfo:
        hipResourceType resourceType
        union (unnamed union at /opt/rocm/include/hip/hip_runtime_api.h:1275:6) resource
        hipArraySparseSubresourceType subresourceType
        union (unnamed union at /opt/rocm/include/hip/hip_runtime_api.h:1280:6) subresource
        hipMemOperationType memOperationType
        hipMemHandleType memHandleType
        union (unnamed union at /opt/rocm/include/hip/hip_runtime_api.h:1299:6) memHandle
        unsigned long long offset
        unsigned int deviceBitMask
        unsigned int flags
        unsigned int[2] reserved
    ctypedef hipArrayMapInfo hipArrayMapInfo
    hipError_t hipInit(unsigned int flags) nogil
    hipError_t hipDriverGetVersion(int * driverVersion) nogil
    hipError_t hipRuntimeGetVersion(int * runtimeVersion) nogil
    hipError_t hipDeviceGet(hipDevice_t * device,int ordinal) nogil
    hipError_t hipDeviceComputeCapability(int * major,int * minor,hipDevice_t device) nogil
    hipError_t hipDeviceGetName(char * name,int len,hipDevice_t device) nogil
    hipError_t hipDeviceGetUuid(hipUUID * uuid,hipDevice_t device) nogil
    hipError_t hipDeviceGetP2PAttribute(int * value,hipDeviceP2PAttr attr,int srcDevice,int dstDevice) nogil
    hipError_t hipDeviceGetPCIBusId(char * pciBusId,int len,int device) nogil
    hipError_t hipDeviceGetByPCIBusId(int * device,const char * pciBusId) nogil
    hipError_t hipDeviceTotalMem(int * bytes,hipDevice_t device) nogil
    hipError_t hipDeviceSynchronize() nogil
    hipError_t hipDeviceReset() nogil
    hipError_t hipSetDevice(int deviceId) nogil
    hipError_t hipGetDevice(int * deviceId) nogil
    hipError_t hipGetDeviceCount(int * count) nogil
    hipError_t hipDeviceGetAttribute(int * pi,hipDeviceAttribute_t attr,int deviceId) nogil
    hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t * mem_pool,int device) nogil
    hipError_t hipDeviceSetMemPool(int device,hipMemPool_t mem_pool) nogil
    hipError_t hipDeviceGetMemPool(hipMemPool_t * mem_pool,int device) nogil
    hipError_t hipGetDeviceProperties(hipDeviceProp_t * prop,int deviceId) nogil
    hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig) nogil
    hipError_t hipDeviceGetCacheConfig(hipFuncCache_t * cacheConfig) nogil
    hipError_t hipDeviceGetLimit(int * pValue,enum hipLimit_t limit) nogil
    hipError_t hipDeviceSetLimit(enum hipLimit_t limit,int value) nogil
    hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig * pConfig) nogil
    hipError_t hipGetDeviceFlags(unsigned int * flags) nogil
    hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig config) nogil
    hipError_t hipSetDeviceFlags(unsigned int flags) nogil
    hipError_t hipChooseDevice(int * device,const hipDeviceProp_t * prop) nogil
    hipError_t hipExtGetLinkTypeAndHopCount(int device1,int device2,uint32_t * linktype,uint32_t * hopcount) nogil
    hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t * handle,void * devPtr) nogil
    hipError_t hipIpcOpenMemHandle(void ** devPtr,hipIpcMemHandle_t handle,unsigned int flags) nogil
    hipError_t hipIpcCloseMemHandle(void * devPtr) nogil
    hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t * handle,hipEvent_t event) nogil
    hipError_t hipIpcOpenEventHandle(hipEvent_t * event,hipIpcEventHandle_t handle) nogil
    hipError_t hipFuncSetAttribute(const void * func,hipFuncAttribute attr,int value) nogil
    hipError_t hipFuncSetCacheConfig(const void * func,hipFuncCache_t config) nogil
    hipError_t hipFuncSetSharedMemConfig(const void * func,hipSharedMemConfig config) nogil
    hipError_t hipGetLastError() nogil
    hipError_t hipPeekAtLastError() nogil
    const char * hipGetErrorName(hipError_t hip_error) nogil
    const char * hipGetErrorString(hipError_t hipError) nogil
    hipError_t hipDrvGetErrorName(hipError_t hipError,const char ** errorString) nogil
    hipError_t hipDrvGetErrorString(hipError_t hipError,const char ** errorString) nogil
    hipError_t hipStreamCreate(hipStream_t * stream) nogil
    hipError_t hipStreamCreateWithFlags(hipStream_t * stream,unsigned int flags) nogil
    hipError_t hipStreamCreateWithPriority(hipStream_t * stream,unsigned int flags,int priority) nogil
    hipError_t hipDeviceGetStreamPriorityRange(int * leastPriority,int * greatestPriority) nogil
    hipError_t hipStreamDestroy(hipStream_t stream) nogil
    hipError_t hipStreamQuery(hipStream_t stream) nogil
    hipError_t hipStreamSynchronize(hipStream_t stream) nogil
    hipError_t hipStreamWaitEvent(hipStream_t stream,hipEvent_t event,unsigned int flags) nogil
    hipError_t hipStreamGetFlags(hipStream_t stream,unsigned int * flags) nogil
    hipError_t hipStreamGetPriority(hipStream_t stream,int * priority) nogil
    hipError_t hipExtStreamCreateWithCUMask(hipStream_t * stream,uint32_t cuMaskSize,const uint32_t * cuMask) nogil
    hipError_t hipExtStreamGetCUMask(hipStream_t stream,uint32_t cuMaskSize,uint32_t * cuMask) nogil
    ctypedef hipStreamCallback_t hipStreamCallback_t
    hipError_t hipStreamAddCallback(hipStream_t stream,hipStreamCallback_t callback,void * userData,unsigned int flags) nogil
    hipError_t hipStreamWaitValue32(hipStream_t stream,void * ptr,uint32_t value,unsigned int flags,uint32_t mask) nogil
    hipError_t hipStreamWaitValue64(hipStream_t stream,void * ptr,uint64_t value,unsigned int flags,uint64_t mask) nogil
    hipError_t hipStreamWriteValue32(hipStream_t stream,void * ptr,uint32_t value,unsigned int flags) nogil
    hipError_t hipStreamWriteValue64(hipStream_t stream,void * ptr,uint64_t value,unsigned int flags) nogil
    hipError_t hipEventCreateWithFlags(hipEvent_t * event,unsigned int flags) nogil
    hipError_t hipEventCreate(hipEvent_t * event) nogil
    hipError_t hipEventRecord(hipEvent_t event,hipStream_t stream) nogil
    hipError_t hipEventDestroy(hipEvent_t event) nogil
    hipError_t hipEventSynchronize(hipEvent_t event) nogil
    hipError_t hipEventElapsedTime(float * ms,hipEvent_t start,hipEvent_t stop) nogil
    hipError_t hipEventQuery(hipEvent_t event) nogil
    hipError_t hipPointerGetAttributes(hipPointerAttribute_t * attributes,const void * ptr) nogil
    hipError_t hipPointerGetAttribute(void * data,hipPointer_attribute attribute,hipDeviceptr_t ptr) nogil
    hipError_t hipDrvPointerGetAttributes(unsigned int numAttributes,hipPointer_attribute * attributes,void ** data,hipDeviceptr_t ptr) nogil
    hipError_t hipImportExternalSemaphore(hipExternalSemaphore_t * extSem_out,const hipExternalSemaphoreHandleDesc * semHandleDesc) nogil
    hipError_t hipSignalExternalSemaphoresAsync(const hipExternalSemaphore_t * extSemArray,const hipExternalSemaphoreSignalParams * paramsArray,unsigned int numExtSems,hipStream_t stream) nogil
    hipError_t hipWaitExternalSemaphoresAsync(const hipExternalSemaphore_t * extSemArray,const hipExternalSemaphoreWaitParams * paramsArray,unsigned int numExtSems,hipStream_t stream) nogil
    hipError_t hipDestroyExternalSemaphore(hipExternalSemaphore_t extSem) nogil
    hipError_t hipImportExternalMemory(hipExternalMemory_t * extMem_out,const hipExternalMemoryHandleDesc * memHandleDesc) nogil
    hipError_t hipExternalMemoryGetMappedBuffer(void ** devPtr,hipExternalMemory_t extMem,const hipExternalMemoryBufferDesc * bufferDesc) nogil
    hipError_t hipDestroyExternalMemory(hipExternalMemory_t extMem) nogil
    hipError_t hipMalloc(void ** ptr,int size) nogil
    hipError_t hipExtMallocWithFlags(void ** ptr,int sizeBytes,unsigned int flags) nogil
    hipError_t hipMallocHost(void ** ptr,int size) nogil
    hipError_t hipMemAllocHost(void ** ptr,int size) nogil
    hipError_t hipHostMalloc(void ** ptr,int size,unsigned int flags) nogil
    hipError_t hipMallocManaged(void ** dev_ptr,int size,unsigned int flags) nogil
    hipError_t hipMemPrefetchAsync(const void * dev_ptr,int count,int device,hipStream_t stream) nogil
    hipError_t hipMemAdvise(const void * dev_ptr,int count,hipMemoryAdvise advice,int device) nogil
    hipError_t hipMemRangeGetAttribute(void * data,int data_size,hipMemRangeAttribute attribute,const void * dev_ptr,int count) nogil
    hipError_t hipMemRangeGetAttributes(void ** data,int * data_sizes,hipMemRangeAttribute * attributes,int num_attributes,const void * dev_ptr,int count) nogil
    hipError_t hipStreamAttachMemAsync(hipStream_t stream,void * dev_ptr,int length,unsigned int flags) nogil
    hipError_t hipMallocAsync(void ** dev_ptr,int size,hipStream_t stream) nogil
    hipError_t hipFreeAsync(void * dev_ptr,hipStream_t stream) nogil
    hipError_t hipMemPoolTrimTo(hipMemPool_t mem_pool,int min_bytes_to_hold) nogil
    hipError_t hipMemPoolSetAttribute(hipMemPool_t mem_pool,hipMemPoolAttr attr,void * value) nogil
    hipError_t hipMemPoolGetAttribute(hipMemPool_t mem_pool,hipMemPoolAttr attr,void * value) nogil
    hipError_t hipMemPoolSetAccess(hipMemPool_t mem_pool,const hipMemAccessDesc * desc_list,int count) nogil
    hipError_t hipMemPoolGetAccess(hipMemAccessFlags * flags,hipMemPool_t mem_pool,hipMemLocation * location) nogil
    hipError_t hipMemPoolCreate(hipMemPool_t * mem_pool,const hipMemPoolProps * pool_props) nogil
    hipError_t hipMemPoolDestroy(hipMemPool_t mem_pool) nogil
    hipError_t hipMallocFromPoolAsync(void ** dev_ptr,int size,hipMemPool_t mem_pool,hipStream_t stream) nogil
    hipError_t hipMemPoolExportToShareableHandle(void * shared_handle,hipMemPool_t mem_pool,hipMemAllocationHandleType handle_type,unsigned int flags) nogil
    hipError_t hipMemPoolImportFromShareableHandle(hipMemPool_t * mem_pool,void * shared_handle,hipMemAllocationHandleType handle_type,unsigned int flags) nogil
    hipError_t hipMemPoolExportPointer(hipMemPoolPtrExportData * export_data,void * dev_ptr) nogil
    hipError_t hipMemPoolImportPointer(void ** dev_ptr,hipMemPool_t mem_pool,hipMemPoolPtrExportData * export_data) nogil
    hipError_t hipHostAlloc(void ** ptr,int size,unsigned int flags) nogil
    hipError_t hipHostGetDevicePointer(void ** devPtr,void * hstPtr,unsigned int flags) nogil
    hipError_t hipHostGetFlags(unsigned int * flagsPtr,void * hostPtr) nogil
    hipError_t hipHostRegister(void * hostPtr,int sizeBytes,unsigned int flags) nogil
    hipError_t hipHostUnregister(void * hostPtr) nogil
    hipError_t hipMallocPitch(void ** ptr,int * pitch,int width,int height) nogil
    hipError_t hipMemAllocPitch(hipDeviceptr_t * dptr,int * pitch,int widthInBytes,int height,unsigned int elementSizeBytes) nogil
    hipError_t hipFree(void * ptr) nogil
    hipError_t hipFreeHost(void * ptr) nogil
    hipError_t hipHostFree(void * ptr) nogil
    hipError_t hipMemcpy(void * dst,const void * src,int sizeBytes,hipMemcpyKind kind) nogil
    hipError_t hipMemcpyWithStream(void * dst,const void * src,int sizeBytes,hipMemcpyKind kind,hipStream_t stream) nogil
    hipError_t hipMemcpyHtoD(hipDeviceptr_t dst,void * src,int sizeBytes) nogil
    hipError_t hipMemcpyDtoH(void * dst,hipDeviceptr_t src,int sizeBytes) nogil
    hipError_t hipMemcpyDtoD(hipDeviceptr_t dst,hipDeviceptr_t src,int sizeBytes) nogil
    hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst,void * src,int sizeBytes,hipStream_t stream) nogil
    hipError_t hipMemcpyDtoHAsync(void * dst,hipDeviceptr_t src,int sizeBytes,hipStream_t stream) nogil
    hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst,hipDeviceptr_t src,int sizeBytes,hipStream_t stream) nogil
    hipError_t hipModuleGetGlobal(hipDeviceptr_t * dptr,int * bytes,hipModule_t hmod,const char * name) nogil
    hipError_t hipGetSymbolAddress(void ** devPtr,const void * symbol) nogil
    hipError_t hipGetSymbolSize(int * size,const void * symbol) nogil
    hipError_t hipMemcpyToSymbol(const void * symbol,const void * src,int sizeBytes,int offset,hipMemcpyKind kind) nogil
    hipError_t hipMemcpyToSymbolAsync(const void * symbol,const void * src,int sizeBytes,int offset,hipMemcpyKind kind,hipStream_t stream) nogil
    hipError_t hipMemcpyFromSymbol(void * dst,const void * symbol,int sizeBytes,int offset,hipMemcpyKind kind) nogil
    hipError_t hipMemcpyFromSymbolAsync(void * dst,const void * symbol,int sizeBytes,int offset,hipMemcpyKind kind,hipStream_t stream) nogil
    hipError_t hipMemcpyAsync(void * dst,const void * src,int sizeBytes,hipMemcpyKind kind,hipStream_t stream) nogil
    hipError_t hipMemset(void * dst,int value,int sizeBytes) nogil
    hipError_t hipMemsetD8(hipDeviceptr_t dest,unsigned char value,int count) nogil
    hipError_t hipMemsetD8Async(hipDeviceptr_t dest,unsigned char value,int count,hipStream_t stream) nogil
    hipError_t hipMemsetD16(hipDeviceptr_t dest,unsigned short value,int count) nogil
    hipError_t hipMemsetD16Async(hipDeviceptr_t dest,unsigned short value,int count,hipStream_t stream) nogil
    hipError_t hipMemsetD32(hipDeviceptr_t dest,int value,int count) nogil
    hipError_t hipMemsetAsync(void * dst,int value,int sizeBytes,hipStream_t stream) nogil
    hipError_t hipMemsetD32Async(hipDeviceptr_t dst,int value,int count,hipStream_t stream) nogil
    hipError_t hipMemset2D(void * dst,int pitch,int value,int width,int height) nogil
    hipError_t hipMemset2DAsync(void * dst,int pitch,int value,int width,int height,hipStream_t stream) nogil
    hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr,int value,hipExtent extent) nogil
    hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr,int value,hipExtent extent,hipStream_t stream) nogil
    hipError_t hipMemGetInfo(int * free,int * total) nogil
    hipError_t hipMemPtrGetInfo(void * ptr,int * size) nogil
    hipError_t hipMallocArray(hipArray ** array,const hipChannelFormatDesc * desc,int width,int height,unsigned int flags) nogil
    hipError_t hipArrayCreate(hipArray ** pHandle,const HIP_ARRAY_DESCRIPTOR * pAllocateArray) nogil
    hipError_t hipArrayDestroy(hipArray * array) nogil
    hipError_t hipArray3DCreate(hipArray ** array,const HIP_ARRAY3D_DESCRIPTOR * pAllocateArray) nogil
    hipError_t hipMalloc3D(hipPitchedPtr * pitchedDevPtr,hipExtent extent) nogil
    hipError_t hipFreeArray(hipArray * array) nogil
    hipError_t hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray) nogil
    hipError_t hipMalloc3DArray(hipArray ** array,const struct hipChannelFormatDesc * desc,struct hipExtent extent,unsigned int flags) nogil
    hipError_t hipMallocMipmappedArray(hipMipmappedArray_t * mipmappedArray,const struct hipChannelFormatDesc * desc,struct hipExtent extent,unsigned int numLevels,unsigned int flags) nogil
    hipError_t hipGetMipmappedArrayLevel(hipArray_t * levelArray,hipMipmappedArray_const_t mipmappedArray,unsigned int level) nogil
    hipError_t hipMemcpy2D(void * dst,int dpitch,const void * src,int spitch,int width,int height,hipMemcpyKind kind) nogil
    hipError_t hipMemcpyParam2D(const hip_Memcpy2D * pCopy) nogil
    hipError_t hipMemcpyParam2DAsync(const hip_Memcpy2D * pCopy,hipStream_t stream) nogil
    hipError_t hipMemcpy2DAsync(void * dst,int dpitch,const void * src,int spitch,int width,int height,hipMemcpyKind kind,hipStream_t stream) nogil
    hipError_t hipMemcpy2DToArray(hipArray * dst,int wOffset,int hOffset,const void * src,int spitch,int width,int height,hipMemcpyKind kind) nogil
    hipError_t hipMemcpy2DToArrayAsync(hipArray * dst,int wOffset,int hOffset,const void * src,int spitch,int width,int height,hipMemcpyKind kind,hipStream_t stream) nogil
    hipError_t hipMemcpyToArray(hipArray * dst,int wOffset,int hOffset,const void * src,int count,hipMemcpyKind kind) nogil
    hipError_t hipMemcpyFromArray(void * dst,hipArray_const_t srcArray,int wOffset,int hOffset,int count,hipMemcpyKind kind) nogil
    hipError_t hipMemcpy2DFromArray(void * dst,int dpitch,hipArray_const_t src,int wOffset,int hOffset,int width,int height,hipMemcpyKind kind) nogil
    hipError_t hipMemcpy2DFromArrayAsync(void * dst,int dpitch,hipArray_const_t src,int wOffset,int hOffset,int width,int height,hipMemcpyKind kind,hipStream_t stream) nogil
    hipError_t hipMemcpyAtoH(void * dst,hipArray * srcArray,int srcOffset,int count) nogil
    hipError_t hipMemcpyHtoA(hipArray * dstArray,int dstOffset,const void * srcHost,int count) nogil
    hipError_t hipMemcpy3D(const struct hipMemcpy3DParms * p) nogil
    hipError_t hipMemcpy3DAsync(const struct hipMemcpy3DParms * p,hipStream_t stream) nogil
    hipError_t hipDrvMemcpy3D(const HIP_MEMCPY3D * pCopy) nogil
    hipError_t hipDrvMemcpy3DAsync(const HIP_MEMCPY3D * pCopy,hipStream_t stream) nogil
    hipError_t hipDeviceCanAccessPeer(int * canAccessPeer,int deviceId,int peerDeviceId) nogil
    hipError_t hipDeviceEnablePeerAccess(int peerDeviceId,unsigned int flags) nogil
    hipError_t hipDeviceDisablePeerAccess(int peerDeviceId) nogil
    hipError_t hipMemGetAddressRange(hipDeviceptr_t * pbase,int * psize,hipDeviceptr_t dptr) nogil
    hipError_t hipMemcpyPeer(void * dst,int dstDeviceId,const void * src,int srcDeviceId,int sizeBytes) nogil
    hipError_t hipMemcpyPeerAsync(void * dst,int dstDeviceId,const void * src,int srcDevice,int sizeBytes,hipStream_t stream) nogil
    hipError_t hipCtxCreate(hipCtx_t * ctx,unsigned int flags,hipDevice_t device) nogil
    hipError_t hipCtxDestroy(hipCtx_t ctx) nogil
    hipError_t hipCtxPopCurrent(hipCtx_t * ctx) nogil
    hipError_t hipCtxPushCurrent(hipCtx_t ctx) nogil
    hipError_t hipCtxSetCurrent(hipCtx_t ctx) nogil
    hipError_t hipCtxGetCurrent(hipCtx_t * ctx) nogil
    hipError_t hipCtxGetDevice(hipDevice_t * device) nogil
    hipError_t hipCtxGetApiVersion(hipCtx_t ctx,int * apiVersion) nogil
    hipError_t hipCtxGetCacheConfig(hipFuncCache_t * cacheConfig) nogil
    hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig) nogil
    hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config) nogil
    hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig * pConfig) nogil
    hipError_t hipCtxSynchronize() nogil
    hipError_t hipCtxGetFlags(unsigned int * flags) nogil
    hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx,unsigned int flags) nogil
    hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx) nogil
    hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev,unsigned int * flags,int * active) nogil
    hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev) nogil
    hipError_t hipDevicePrimaryCtxRetain(hipCtx_t * pctx,hipDevice_t dev) nogil
    hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev) nogil
    hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev,unsigned int flags) nogil
    hipError_t hipModuleLoad(hipModule_t * module,const char * fname) nogil
    hipError_t hipModuleUnload(hipModule_t module) nogil
    hipError_t hipModuleGetFunction(hipFunction_t * function,hipModule_t module,const char * kname) nogil
    hipError_t hipFuncGetAttributes(struct hipFuncAttributes * attr,const void * func) nogil
    hipError_t hipFuncGetAttribute(int * value,hipFunction_attribute attrib,hipFunction_t hfunc) nogil
    hipError_t hipModuleGetTexRef(textureReference ** texRef,hipModule_t hmod,const char * name) nogil
    hipError_t hipModuleLoadData(hipModule_t * module,const void * image) nogil
    hipError_t hipModuleLoadDataEx(hipModule_t * module,const void * image,unsigned int numOptions,hipJitOption * options,void ** optionValues) nogil
    hipError_t hipModuleLaunchKernel(hipFunction_t f,unsigned int gridDimX,unsigned int gridDimY,unsigned int gridDimZ,unsigned int blockDimX,unsigned int blockDimY,unsigned int blockDimZ,unsigned int sharedMemBytes,hipStream_t stream,void ** kernelParams,void ** extra) nogil
    hipError_t hipLaunchCooperativeKernel(const void * f,dim3 gridDim,dim3 blockDimX,void ** kernelParams,unsigned int sharedMemBytes,hipStream_t stream) nogil
    hipError_t hipLaunchCooperativeKernelMultiDevice(hipLaunchParams * launchParamsList,int numDevices,unsigned int flags) nogil
    hipError_t hipExtLaunchMultiKernelMultiDevice(hipLaunchParams * launchParamsList,int numDevices,unsigned int flags) nogil
    hipError_t hipModuleOccupancyMaxPotentialBlockSize(int * gridSize,int * blockSize,hipFunction_t f,int dynSharedMemPerBlk,int blockSizeLimit) nogil
    hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(int * gridSize,int * blockSize,hipFunction_t f,int dynSharedMemPerBlk,int blockSizeLimit,unsigned int flags) nogil
    hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks,hipFunction_t f,int blockSize,int dynSharedMemPerBlk) nogil
    hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks,hipFunction_t f,int blockSize,int dynSharedMemPerBlk,unsigned int flags) nogil
    hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks,const void * f,int blockSize,int dynSharedMemPerBlk) nogil
    hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks,const void * f,int blockSize,int dynSharedMemPerBlk,unsigned int flags) nogil
    hipError_t hipOccupancyMaxPotentialBlockSize(int * gridSize,int * blockSize,const void * f,int dynSharedMemPerBlk,int blockSizeLimit) nogil
    hipError_t hipProfilerStart() nogil
    hipError_t hipProfilerStop() nogil
    hipError_t hipConfigureCall(dim3 gridDim,dim3 blockDim,int sharedMem,hipStream_t stream) nogil
    hipError_t hipSetupArgument(const void * arg,int size,int offset) nogil
    hipError_t hipLaunchByPtr(const void * func) nogil
    hipError_t hipLaunchKernel(const void * function_address,dim3 numBlocks,dim3 dimBlocks,void ** args,int sharedMemBytes,hipStream_t stream) nogil
    hipError_t hipLaunchHostFunc(hipStream_t stream,hipHostFn_t fn,void * userData) nogil
    hipError_t hipDrvMemcpy2DUnaligned(const hip_Memcpy2D * pCopy) nogil
    hipError_t hipExtLaunchKernel(const void * function_address,dim3 numBlocks,dim3 dimBlocks,void ** args,int sharedMemBytes,hipStream_t stream,hipEvent_t startEvent,hipEvent_t stopEvent,int flags) nogil
    hipError_t hipBindTextureToMipmappedArray(const textureReference * tex,hipMipmappedArray_const_t mipmappedArray,const hipChannelFormatDesc * desc) nogil
    hipError_t hipCreateTextureObject(hipTextureObject_t * pTexObject,const hipResourceDesc * pResDesc,const hipTextureDesc * pTexDesc,const struct hipResourceViewDesc * pResViewDesc) nogil
    hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject) nogil
    hipError_t hipGetChannelDesc(hipChannelFormatDesc * desc,hipArray_const_t array) nogil
    hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc * pResDesc,hipTextureObject_t textureObject) nogil
    hipError_t hipGetTextureObjectResourceViewDesc(struct hipResourceViewDesc * pResViewDesc,hipTextureObject_t textureObject) nogil
    hipError_t hipGetTextureObjectTextureDesc(hipTextureDesc * pTexDesc,hipTextureObject_t textureObject) nogil
    hipError_t hipTexObjectCreate(hipTextureObject_t * pTexObject,const HIP_RESOURCE_DESC * pResDesc,const HIP_TEXTURE_DESC * pTexDesc,const HIP_RESOURCE_VIEW_DESC * pResViewDesc) nogil
    hipError_t hipTexObjectDestroy(hipTextureObject_t texObject) nogil
    hipError_t hipTexObjectGetResourceDesc(HIP_RESOURCE_DESC * pResDesc,hipTextureObject_t texObject) nogil
    hipError_t hipTexObjectGetResourceViewDesc(HIP_RESOURCE_VIEW_DESC * pResViewDesc,hipTextureObject_t texObject) nogil
    hipError_t hipTexObjectGetTextureDesc(HIP_TEXTURE_DESC * pTexDesc,hipTextureObject_t texObject) nogil
    hipError_t hipGetTextureReference(const textureReference ** texref,const void * symbol) nogil
    hipError_t hipTexRefSetAddressMode(textureReference * texRef,int dim,enum hipTextureAddressMode am) nogil
    hipError_t hipTexRefSetArray(textureReference * tex,hipArray_const_t array,unsigned int flags) nogil
    hipError_t hipTexRefSetFilterMode(textureReference * texRef,enum hipTextureFilterMode fm) nogil
    hipError_t hipTexRefSetFlags(textureReference * texRef,unsigned int Flags) nogil
    hipError_t hipTexRefSetFormat(textureReference * texRef,hipArray_Format fmt,int NumPackedComponents) nogil
    hipError_t hipBindTexture(int * offset,const textureReference * tex,const void * devPtr,const hipChannelFormatDesc * desc,int size) nogil
    hipError_t hipBindTexture2D(int * offset,const textureReference * tex,const void * devPtr,const hipChannelFormatDesc * desc,int width,int height,int pitch) nogil
    hipError_t hipBindTextureToArray(const textureReference * tex,hipArray_const_t array,const hipChannelFormatDesc * desc) nogil
    hipError_t hipGetTextureAlignmentOffset(int * offset,const textureReference * texref) nogil
    hipError_t hipUnbindTexture(const textureReference * tex) nogil
    hipError_t hipTexRefGetAddress(hipDeviceptr_t * dev_ptr,const textureReference * texRef) nogil
    hipError_t hipTexRefGetAddressMode(enum hipTextureAddressMode * pam,const textureReference * texRef,int dim) nogil
    hipError_t hipTexRefGetFilterMode(enum hipTextureFilterMode * pfm,const textureReference * texRef) nogil
    hipError_t hipTexRefGetFlags(unsigned int * pFlags,const textureReference * texRef) nogil
    hipError_t hipTexRefGetFormat(hipArray_Format * pFormat,int * pNumChannels,const textureReference * texRef) nogil
    hipError_t hipTexRefGetMaxAnisotropy(int * pmaxAnsio,const textureReference * texRef) nogil
    hipError_t hipTexRefGetMipmapFilterMode(enum hipTextureFilterMode * pfm,const textureReference * texRef) nogil
    hipError_t hipTexRefGetMipmapLevelBias(float * pbias,const textureReference * texRef) nogil
    hipError_t hipTexRefGetMipmapLevelClamp(float * pminMipmapLevelClamp,float * pmaxMipmapLevelClamp,const textureReference * texRef) nogil
    hipError_t hipTexRefGetMipMappedArray(hipMipmappedArray_t * pArray,const textureReference * texRef) nogil
    hipError_t hipTexRefSetAddress(int * ByteOffset,textureReference * texRef,hipDeviceptr_t dptr,int bytes) nogil
    hipError_t hipTexRefSetAddress2D(textureReference * texRef,const HIP_ARRAY_DESCRIPTOR * desc,hipDeviceptr_t dptr,int Pitch) nogil
    hipError_t hipTexRefSetMaxAnisotropy(textureReference * texRef,unsigned int maxAniso) nogil
    hipError_t hipTexRefSetBorderColor(textureReference * texRef,float * pBorderColor) nogil
    hipError_t hipTexRefSetMipmapFilterMode(textureReference * texRef,enum hipTextureFilterMode fm) nogil
    hipError_t hipTexRefSetMipmapLevelBias(textureReference * texRef,float bias) nogil
    hipError_t hipTexRefSetMipmapLevelClamp(textureReference * texRef,float minMipMapLevelClamp,float maxMipMapLevelClamp) nogil
    hipError_t hipTexRefSetMipmappedArray(textureReference * texRef,struct hipMipmappedArray * mipmappedArray,unsigned int Flags) nogil
    hipError_t hipMipmappedArrayCreate(hipMipmappedArray_t * pHandle,HIP_ARRAY3D_DESCRIPTOR * pMipmappedArrayDesc,unsigned int numMipmapLevels) nogil
    hipError_t hipMipmappedArrayDestroy(hipMipmappedArray_t hMipmappedArray) nogil
    hipError_t hipMipmappedArrayGetLevel(hipArray_t * pLevelArray,hipMipmappedArray_t hMipMappedArray,unsigned int level) nogil
    const char * hipApiName(uint32_t id) nogil
    const char * hipKernelNameRef(const hipFunction_t f) nogil
    const char * hipKernelNameRefByPtr(const void * hostFunction,hipStream_t stream) nogil
    int hipGetStreamDeviceId(hipStream_t stream) nogil
    hipError_t hipStreamBeginCapture(hipStream_t stream,hipStreamCaptureMode mode) nogil
    hipError_t hipStreamEndCapture(hipStream_t stream,hipGraph_t * pGraph) nogil
    hipError_t hipStreamGetCaptureInfo(hipStream_t stream,hipStreamCaptureStatus * pCaptureStatus,unsigned long long * pId) nogil
    hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream,hipStreamCaptureStatus * captureStatus_out,unsigned long long * id_out,hipGraph_t * graph_out,const hipGraphNode_t ** dependencies_out,int * numDependencies_out) nogil
    hipError_t hipStreamIsCapturing(hipStream_t stream,hipStreamCaptureStatus * pCaptureStatus) nogil
    hipError_t hipStreamUpdateCaptureDependencies(hipStream_t stream,hipGraphNode_t * dependencies,int numDependencies,unsigned int flags) nogil
    hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode * mode) nogil
    hipError_t hipGraphCreate(hipGraph_t * pGraph,unsigned int flags) nogil
    hipError_t hipGraphDestroy(hipGraph_t graph) nogil
    hipError_t hipGraphAddDependencies(hipGraph_t graph,const hipGraphNode_t * from,const hipGraphNode_t * to,int numDependencies) nogil
    hipError_t hipGraphRemoveDependencies(hipGraph_t graph,const hipGraphNode_t * from,const hipGraphNode_t * to,int numDependencies) nogil
    hipError_t hipGraphGetEdges(hipGraph_t graph,hipGraphNode_t * from,hipGraphNode_t * to,int * numEdges) nogil
    hipError_t hipGraphGetNodes(hipGraph_t graph,hipGraphNode_t * nodes,int * numNodes) nogil
    hipError_t hipGraphGetRootNodes(hipGraph_t graph,hipGraphNode_t * pRootNodes,int * pNumRootNodes) nogil
    hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node,hipGraphNode_t * pDependencies,int * pNumDependencies) nogil
    hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node,hipGraphNode_t * pDependentNodes,int * pNumDependentNodes) nogil
    hipError_t hipGraphNodeGetType(hipGraphNode_t node,hipGraphNodeType * pType) nogil
    hipError_t hipGraphDestroyNode(hipGraphNode_t node) nogil
    hipError_t hipGraphClone(hipGraph_t * pGraphClone,hipGraph_t originalGraph) nogil
    hipError_t hipGraphNodeFindInClone(hipGraphNode_t * pNode,hipGraphNode_t originalNode,hipGraph_t clonedGraph) nogil
    hipError_t hipGraphInstantiate(hipGraphExec_t * pGraphExec,hipGraph_t graph,hipGraphNode_t * pErrorNode,char * pLogBuffer,int bufferSize) nogil
    hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t * pGraphExec,hipGraph_t graph,unsigned long long flags) nogil
    hipError_t hipGraphLaunch(hipGraphExec_t graphExec,hipStream_t stream) nogil
    hipError_t hipGraphUpload(hipGraphExec_t graphExec,hipStream_t stream) nogil
    hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec) nogil
    hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec,hipGraph_t hGraph,hipGraphNode_t * hErrorNode_out,hipGraphExecUpdateResult * updateResult_out) nogil
    hipError_t hipGraphAddKernelNode(hipGraphNode_t * pGraphNode,hipGraph_t graph,const hipGraphNode_t * pDependencies,int numDependencies,const hipKernelNodeParams * pNodeParams) nogil
    hipError_t hipGraphKernelNodeGetParams(hipGraphNode_t node,hipKernelNodeParams * pNodeParams) nogil
    hipError_t hipGraphKernelNodeSetParams(hipGraphNode_t node,const hipKernelNodeParams * pNodeParams) nogil
    hipError_t hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec,hipGraphNode_t node,const hipKernelNodeParams * pNodeParams) nogil
    hipError_t hipGraphAddMemcpyNode(hipGraphNode_t * pGraphNode,hipGraph_t graph,const hipGraphNode_t * pDependencies,int numDependencies,const hipMemcpy3DParms * pCopyParams) nogil
    hipError_t hipGraphMemcpyNodeGetParams(hipGraphNode_t node,hipMemcpy3DParms * pNodeParams) nogil
    hipError_t hipGraphMemcpyNodeSetParams(hipGraphNode_t node,const hipMemcpy3DParms * pNodeParams) nogil
    hipError_t hipGraphKernelNodeSetAttribute(hipGraphNode_t hNode,hipKernelNodeAttrID attr,const hipKernelNodeAttrValue * value) nogil
    hipError_t hipGraphKernelNodeGetAttribute(hipGraphNode_t hNode,hipKernelNodeAttrID attr,hipKernelNodeAttrValue * value) nogil
    hipError_t hipGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec,hipGraphNode_t node,hipMemcpy3DParms * pNodeParams) nogil
    hipError_t hipGraphAddMemcpyNode1D(hipGraphNode_t * pGraphNode,hipGraph_t graph,const hipGraphNode_t * pDependencies,int numDependencies,void * dst,const void * src,int count,hipMemcpyKind kind) nogil
    hipError_t hipGraphMemcpyNodeSetParams1D(hipGraphNode_t node,void * dst,const void * src,int count,hipMemcpyKind kind) nogil
    hipError_t hipGraphExecMemcpyNodeSetParams1D(hipGraphExec_t hGraphExec,hipGraphNode_t node,void * dst,const void * src,int count,hipMemcpyKind kind) nogil
    hipError_t hipGraphAddMemcpyNodeFromSymbol(hipGraphNode_t * pGraphNode,hipGraph_t graph,const hipGraphNode_t * pDependencies,int numDependencies,void * dst,const void * symbol,int count,int offset,hipMemcpyKind kind) nogil
    hipError_t hipGraphMemcpyNodeSetParamsFromSymbol(hipGraphNode_t node,void * dst,const void * symbol,int count,int offset,hipMemcpyKind kind) nogil
    hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(hipGraphExec_t hGraphExec,hipGraphNode_t node,void * dst,const void * symbol,int count,int offset,hipMemcpyKind kind) nogil
    hipError_t hipGraphAddMemcpyNodeToSymbol(hipGraphNode_t * pGraphNode,hipGraph_t graph,const hipGraphNode_t * pDependencies,int numDependencies,const void * symbol,const void * src,int count,int offset,hipMemcpyKind kind) nogil
    hipError_t hipGraphMemcpyNodeSetParamsToSymbol(hipGraphNode_t node,const void * symbol,const void * src,int count,int offset,hipMemcpyKind kind) nogil
    hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol(hipGraphExec_t hGraphExec,hipGraphNode_t node,const void * symbol,const void * src,int count,int offset,hipMemcpyKind kind) nogil
    hipError_t hipGraphAddMemsetNode(hipGraphNode_t * pGraphNode,hipGraph_t graph,const hipGraphNode_t * pDependencies,int numDependencies,const hipMemsetParams * pMemsetParams) nogil
    hipError_t hipGraphMemsetNodeGetParams(hipGraphNode_t node,hipMemsetParams * pNodeParams) nogil
    hipError_t hipGraphMemsetNodeSetParams(hipGraphNode_t node,const hipMemsetParams * pNodeParams) nogil
    hipError_t hipGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec,hipGraphNode_t node,const hipMemsetParams * pNodeParams) nogil
    hipError_t hipGraphAddHostNode(hipGraphNode_t * pGraphNode,hipGraph_t graph,const hipGraphNode_t * pDependencies,int numDependencies,const hipHostNodeParams * pNodeParams) nogil
    hipError_t hipGraphHostNodeGetParams(hipGraphNode_t node,hipHostNodeParams * pNodeParams) nogil
    hipError_t hipGraphHostNodeSetParams(hipGraphNode_t node,const hipHostNodeParams * pNodeParams) nogil
    hipError_t hipGraphExecHostNodeSetParams(hipGraphExec_t hGraphExec,hipGraphNode_t node,const hipHostNodeParams * pNodeParams) nogil
    hipError_t hipGraphAddChildGraphNode(hipGraphNode_t * pGraphNode,hipGraph_t graph,const hipGraphNode_t * pDependencies,int numDependencies,hipGraph_t childGraph) nogil
    hipError_t hipGraphChildGraphNodeGetGraph(hipGraphNode_t node,hipGraph_t * pGraph) nogil
    hipError_t hipGraphExecChildGraphNodeSetParams(hipGraphExec_t hGraphExec,hipGraphNode_t node,hipGraph_t childGraph) nogil
    hipError_t hipGraphAddEmptyNode(hipGraphNode_t * pGraphNode,hipGraph_t graph,const hipGraphNode_t * pDependencies,int numDependencies) nogil
    hipError_t hipGraphAddEventRecordNode(hipGraphNode_t * pGraphNode,hipGraph_t graph,const hipGraphNode_t * pDependencies,int numDependencies,hipEvent_t event) nogil
    hipError_t hipGraphEventRecordNodeGetEvent(hipGraphNode_t node,hipEvent_t * event_out) nogil
    hipError_t hipGraphEventRecordNodeSetEvent(hipGraphNode_t node,hipEvent_t event) nogil
    hipError_t hipGraphExecEventRecordNodeSetEvent(hipGraphExec_t hGraphExec,hipGraphNode_t hNode,hipEvent_t event) nogil
    hipError_t hipGraphAddEventWaitNode(hipGraphNode_t * pGraphNode,hipGraph_t graph,const hipGraphNode_t * pDependencies,int numDependencies,hipEvent_t event) nogil
    hipError_t hipGraphEventWaitNodeGetEvent(hipGraphNode_t node,hipEvent_t * event_out) nogil
    hipError_t hipGraphEventWaitNodeSetEvent(hipGraphNode_t node,hipEvent_t event) nogil
    hipError_t hipGraphExecEventWaitNodeSetEvent(hipGraphExec_t hGraphExec,hipGraphNode_t hNode,hipEvent_t event) nogil
    hipError_t hipDeviceGetGraphMemAttribute(int device,hipGraphMemAttributeType attr,void * value) nogil
    hipError_t hipDeviceSetGraphMemAttribute(int device,hipGraphMemAttributeType attr,void * value) nogil
    hipError_t hipDeviceGraphMemTrim(int device) nogil
    hipError_t hipUserObjectCreate(hipUserObject_t * object_out,void * ptr,hipHostFn_t destroy,unsigned int initialRefcount,unsigned int flags) nogil
    hipError_t hipUserObjectRelease(hipUserObject_t object,unsigned int count) nogil
    hipError_t hipUserObjectRetain(hipUserObject_t object,unsigned int count) nogil
    hipError_t hipGraphRetainUserObject(hipGraph_t graph,hipUserObject_t object,unsigned int count,unsigned int flags) nogil
    hipError_t hipGraphReleaseUserObject(hipGraph_t graph,hipUserObject_t object,unsigned int count) nogil
    hipError_t hipMemAddressFree(void * devPtr,int size) nogil
    hipError_t hipMemAddressReserve(void ** ptr,int size,int alignment,void * addr,unsigned long long flags) nogil
    hipError_t hipMemCreate(hipMemGenericAllocationHandle_t * handle,int size,const hipMemAllocationProp * prop,unsigned long long flags) nogil
    hipError_t hipMemExportToShareableHandle(void * shareableHandle,hipMemGenericAllocationHandle_t handle,hipMemAllocationHandleType handleType,unsigned long long flags) nogil
    hipError_t hipMemGetAccess(unsigned long long * flags,const hipMemLocation * location,void * ptr) nogil
    hipError_t hipMemGetAllocationGranularity(int * granularity,const hipMemAllocationProp * prop,hipMemAllocationGranularity_flags option) nogil
    hipError_t hipMemGetAllocationPropertiesFromHandle(hipMemAllocationProp * prop,hipMemGenericAllocationHandle_t handle) nogil
    hipError_t hipMemImportFromShareableHandle(hipMemGenericAllocationHandle_t * handle,void * osHandle,hipMemAllocationHandleType shHandleType) nogil
    hipError_t hipMemMap(void * ptr,int size,int offset,hipMemGenericAllocationHandle_t handle,unsigned long long flags) nogil
    hipError_t hipMemMapArrayAsync(hipArrayMapInfo * mapInfoList,unsigned int count,hipStream_t stream) nogil
    hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle) nogil
    hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t * handle,void * addr) nogil
    hipError_t hipMemSetAccess(void * ptr,int size,const hipMemAccessDesc * desc,int count) nogil
    hipError_t hipMemUnmap(void * ptr,int size) nogil
    hipError_t hipGLGetDevices(unsigned int * pHipDeviceCount,int * pHipDevices,unsigned int hipDeviceCount,hipGLDeviceList deviceList) nogil
    hipError_t hipGraphicsGLRegisterBuffer(hipGraphicsResource ** resource,GLuint buffer,unsigned int flags) nogil
    hipError_t hipGraphicsGLRegisterImage(hipGraphicsResource ** resource,GLuint image,GLenum target,unsigned int flags) nogil
    hipError_t hipGraphicsMapResources(int count,hipGraphicsResource_t * resources,hipStream_t stream) nogil
    hipError_t hipGraphicsSubResourceGetMappedArray(hipArray_t * array,hipGraphicsResource_t resource,unsigned int arrayIndex,unsigned int mipLevel) nogil
    hipError_t hipGraphicsResourceGetMappedPointer(void ** devPtr,int * size,hipGraphicsResource_t resource) nogil
    hipError_t hipGraphicsUnmapResources(int count,hipGraphicsResource_t * resources,hipStream_t stream) nogil
    hipError_t hipGraphicsUnregisterResource(hipGraphicsResource_t resource) nogil
    hipError_t hipMemcpy_spt(void * dst,const void * src,int sizeBytes,hipMemcpyKind kind) nogil
    hipError_t hipMemcpyToSymbol_spt(const void * symbol,const void * src,int sizeBytes,int offset,hipMemcpyKind kind) nogil
    hipError_t hipMemcpyFromSymbol_spt(void * dst,const void * symbol,int sizeBytes,int offset,hipMemcpyKind kind) nogil
    hipError_t hipMemcpy2D_spt(void * dst,int dpitch,const void * src,int spitch,int width,int height,hipMemcpyKind kind) nogil
    hipError_t hipMemcpy2DFromArray_spt(void * dst,int dpitch,hipArray_const_t src,int wOffset,int hOffset,int width,int height,hipMemcpyKind kind) nogil
    hipError_t hipMemcpy3D_spt(const struct hipMemcpy3DParms * p) nogil
    hipError_t hipMemset_spt(void * dst,int value,int sizeBytes) nogil
    hipError_t hipMemsetAsync_spt(void * dst,int value,int sizeBytes,hipStream_t stream) nogil
    hipError_t hipMemset2D_spt(void * dst,int pitch,int value,int width,int height) nogil
    hipError_t hipMemset2DAsync_spt(void * dst,int pitch,int value,int width,int height,hipStream_t stream) nogil
    hipError_t hipMemset3DAsync_spt(hipPitchedPtr pitchedDevPtr,int value,hipExtent extent,hipStream_t stream) nogil
    hipError_t hipMemset3D_spt(hipPitchedPtr pitchedDevPtr,int value,hipExtent extent) nogil
    hipError_t hipMemcpyAsync_spt(void * dst,const void * src,int sizeBytes,hipMemcpyKind kind,hipStream_t stream) nogil
    hipError_t hipMemcpy3DAsync_spt(const hipMemcpy3DParms * p,hipStream_t stream) nogil
    hipError_t hipMemcpy2DAsync_spt(void * dst,int dpitch,const void * src,int spitch,int width,int height,hipMemcpyKind kind,hipStream_t stream) nogil
    hipError_t hipMemcpyFromSymbolAsync_spt(void * dst,const void * symbol,int sizeBytes,int offset,hipMemcpyKind kind,hipStream_t stream) nogil
    hipError_t hipMemcpyToSymbolAsync_spt(const void * symbol,const void * src,int sizeBytes,int offset,hipMemcpyKind kind,hipStream_t stream) nogil
    hipError_t hipMemcpyFromArray_spt(void * dst,hipArray_const_t src,int wOffsetSrc,int hOffset,int count,hipMemcpyKind kind) nogil
    hipError_t hipMemcpy2DToArray_spt(hipArray * dst,int wOffset,int hOffset,const void * src,int spitch,int width,int height,hipMemcpyKind kind) nogil
    hipError_t hipMemcpy2DFromArrayAsync_spt(void * dst,int dpitch,hipArray_const_t src,int wOffsetSrc,int hOffsetSrc,int width,int height,hipMemcpyKind kind,hipStream_t stream) nogil
    hipError_t hipMemcpy2DToArrayAsync_spt(hipArray * dst,int wOffset,int hOffset,const void * src,int spitch,int width,int height,hipMemcpyKind kind,hipStream_t stream) nogil
    hipError_t hipStreamQuery_spt(hipStream_t stream) nogil
    hipError_t hipStreamSynchronize_spt(hipStream_t stream) nogil
    hipError_t hipStreamGetPriority_spt(hipStream_t stream,int * priority) nogil
    hipError_t hipStreamWaitEvent_spt(hipStream_t stream,hipEvent_t event,unsigned int flags) nogil
    hipError_t hipStreamGetFlags_spt(hipStream_t stream,unsigned int * flags) nogil
    hipError_t hipStreamAddCallback_spt(hipStream_t stream,hipStreamCallback_t callback,void * userData,unsigned int flags) nogil
    hipError_t hipEventRecord_spt(hipEvent_t event,hipStream_t stream) nogil
    hipError_t hipLaunchCooperativeKernel_spt(const void * f,dim3 gridDim,dim3 blockDim,void ** kernelParams,uint32_t sharedMemBytes,hipStream_t hStream) nogil
    hipError_t hipLaunchKernel_spt(const void * function_address,dim3 numBlocks,dim3 dimBlocks,void ** args,int sharedMemBytes,hipStream_t stream) nogil
    hipError_t hipGraphLaunch_spt(hipGraphExec_t graphExec,hipStream_t stream) nogil
    hipError_t hipStreamBeginCapture_spt(hipStream_t stream,hipStreamCaptureMode mode) nogil
    hipError_t hipStreamEndCapture_spt(hipStream_t stream,hipGraph_t * pGraph) nogil
    hipError_t hipStreamIsCapturing_spt(hipStream_t stream,hipStreamCaptureStatus * pCaptureStatus) nogil
    hipError_t hipStreamGetCaptureInfo_spt(hipStream_t stream,hipStreamCaptureStatus * pCaptureStatus,unsigned long long * pId) nogil
    hipError_t hipStreamGetCaptureInfo_v2_spt(hipStream_t stream,hipStreamCaptureStatus * captureStatus_out,unsigned long long * id_out,hipGraph_t * graph_out,const hipGraphNode_t ** dependencies_out,int * numDependencies_out) nogil
    hipError_t hipLaunchHostFunc_spt(hipStream_t stream,hipHostFn_t fn,void * userData) nogil
