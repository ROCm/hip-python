# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

"""This is the project's setup script.

After pointing this setup script to an HIP installation,
it generates Cython files and compiles them to Python
modules. The generated Cython declaration files can be used
by Cython users of this project.
"""

import codegen

# entry point
if __name__ == "__main__":
    include_dir = "/opt/rocm/include"

    # hiprtc
    def hiprtc_node_filter(node: codegen.Node):
        return node.name.startswith("hiprtc")

    pkg_gen_hiprtc = codegen.PackageGenerator(
        "hiprtc", include_dir, ["hip/hiprtc.h"], "libhiprtc.so", hiprtc_node_filter
    )

    with open("hip/chiprtc.pxd", "w") as outfile:
        outfile.write(pkg_gen_hiprtc.render_cython_bindings())

    # hip
    def hip_node_filter(node: codegen.Node):
        return node.name.startswith("hip") or node.name.startswith("HIP")

    pkg_gen_hip = codegen.PackageGenerator(
        "hip",
        include_dir,
        [
            # "hip/vector_types.h",
            "hip/driver_types.h",
            "hip/surface_types.h",
            "hip/texture_types.h",
            "hip/library_types.h",
            "hip/hip_runtime_api.h",
            # "hip/device_types.h",
        ],
        "libhipamd64.so",
        hip_node_filter,
    )

    with open("hip/chip.pxd", "w") as outfile:
        outfile.write(pkg_gen_hip.render_cython_bindings())
