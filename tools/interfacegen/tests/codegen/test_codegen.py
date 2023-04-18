# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

import tempfile

import addtoplevelpath
from _codegen import CParser, PackageGenerator

file_content = """\
typedef unsigned int GLuint;

typedef struct hipResourceType {
  int a;
}

typedef struct hipMipmappedArray {
  int a;
}

typedef struct hipArray_t {
  int a;
}

typedef struct hipArrayMapInfo {
     hipResourceType resourceType;                   ///< Resource type
     union {
         hipMipmappedArray mipmap;
         hipArray_t array;
     } resource;
     hipArraySparseSubresourceType subresourceType;  ///< Sparse subresource type
     union {
         struct {
             unsigned int level;   ///< For mipmapped arrays must be a valid mipmap level. For arrays must be zero
             unsigned int layer;   ///< For layered arrays must be a valid layer index. Otherwise, must be zero
             unsigned int offsetX;                   ///< X offset in elements
             unsigned int offsetY;                   ///< Y offset in elements
             unsigned int offsetZ;                   ///< Z offset in elements
             unsigned int extentWidth;               ///< Width in elements
             unsigned int extentHeight;              ///< Height in elements
             unsigned int extentDepth;               ///< Depth in elements
         } sparseLevel;
         struct {
             unsigned int layer;   ///< For layered arrays must be a valid layer index. Otherwise, must be zero
             unsigned long long offset;              ///< Offset within mip tail
             unsigned long long size;                ///< Extent in bytes
         } miptail;
     } subresource;
     hipMemOperationType memOperationType;           ///< Memory operation type
     hipMemHandleType memHandleType;                 ///< Memory handle type
     union {
         hipMemGenericAllocationHandle_t memHandle;
     } memHandle;
     unsigned long long offset;                      ///< Offset within the memory
     unsigned int deviceBitMask;                     ///< Device ordinal bit mask
     unsigned int flags;                             ///< flags for future use, must be zero now.
     unsigned int reserved[2];                       ///< Reserved for future use, must be zero now.
} hipArrayMapInfo;\
"""
file_name = "input.h"
parser = CParser(file_name, unsaved_files=[(file_name, file_content)])
parser.parse()
print(parser.render_cursors())

pkg_gen_hip = PackageGenerator("hip", None, [file_name], "libhipamd64.so")
print(pkg_gen_hip.render_cython_c_bindings())
print(pkg_gen_hip.render_python_interfaces("hip"))
