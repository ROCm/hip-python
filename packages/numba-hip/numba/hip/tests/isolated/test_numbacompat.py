import importlib.util
import os


def test_postprocessing():
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    module_name = "numbacompat"
    file_path = f"{parent_dir}/../../util/{module_name}.py"

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    numbacompat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(numbacompat)

    input_llvm_ir = """\
%0= alloca double, align 8
%1 = alloca double, align 8
%a = alloca double, align 8
%"a"= alloca double, align 8
%"a" = alloca double, align 8
%tmp.2 = alloca double, align 8
%.99 = alloca { ptr, ptr, i64, i64, ptr, [2 x i64], [2 x i64] }, align 8
%"$phi62.0" = alloca { ptr, i64, i64, ptr }, align 8
%"$phi64.1" = alloca i64, align 8
%val1 = sext ptr null to i32
%val2 = sext ptr null to i64
"""

    expected_llvm_ir = """\
%0__numba_hip_tmp = alloca double, align 8, addrspace(5)
%0 = addrspacecast ptr addrspace(5) %0__numba_hip_tmp to ptr addrspace(0)
%1__numba_hip_tmp = alloca double, align 8, addrspace(5)
%1 = addrspacecast ptr addrspace(5) %1__numba_hip_tmp to ptr addrspace(0)
%a__numba_hip_tmp = alloca double, align 8, addrspace(5)
%a = addrspacecast ptr addrspace(5) %a__numba_hip_tmp to ptr addrspace(0)
%"a__numba_hip_tmp" = alloca double, align 8, addrspace(5)
%"a" = addrspacecast ptr addrspace(5) %"a__numba_hip_tmp" to ptr addrspace(0)
%"a__numba_hip_tmp" = alloca double, align 8, addrspace(5)
%"a" = addrspacecast ptr addrspace(5) %"a__numba_hip_tmp" to ptr addrspace(0)
%tmp.2__numba_hip_tmp = alloca double, align 8, addrspace(5)
%tmp.2 = addrspacecast ptr addrspace(5) %tmp.2__numba_hip_tmp to ptr addrspace(0)
%.99__numba_hip_tmp = alloca { ptr, ptr, i64, i64, ptr, [2 x i64], [2 x i64] }, align 8, addrspace(5)
%.99 = addrspacecast ptr addrspace(5) %.99__numba_hip_tmp to ptr addrspace(0)
%"$phi62.0__numba_hip_tmp" = alloca { ptr, i64, i64, ptr }, align 8, addrspace(5)
%"$phi62.0" = addrspacecast ptr addrspace(5) %"$phi62.0__numba_hip_tmp" to ptr addrspace(0)
%"$phi64.1__numba_hip_tmp" = alloca i64, align 8, addrspace(5)
%"$phi64.1" = addrspacecast ptr addrspace(5) %"$phi64.1__numba_hip_tmp" to ptr addrspace(0)
%val1 = ptrtoint ptr null to i32
%val2 = ptrtoint ptr null to i64
"""

    post_processed_llvm_ir = numbacompat.postprocess_numba_llvm_ir(
        input_llvm_ir
    )
    # print(post_processed_llvm_ir)
    assert post_processed_llvm_ir == expected_llvm_ir


if __name__ == "__main__":
    test_postprocessing()
