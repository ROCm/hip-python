"""HIP-based ROC Driver

- Driver API binding
- Device array implementation

"""

from numba.core import config
assert not config.ENABLE_CUDASIM, "Cannot use real driver API with simulator"

# ^ based on original code

#-----------------------
# Now follow the modules
#-----------------------

import numba.roc._modulerepl as _modulerepl
import os
import re

mr = _modulerepl.ModuleReplicator(
    "numba.roc.cudadrv", os.path.join(os.path.dirname(__file__), "..", "..", "cuda", "cudadrv"),
    base_context=globals(),
    preprocess_all=lambda content: re.sub(r"\bnumba.cuda\b","numba.roc",content),
)

def _preprocess_driver(content: str):
    """Applied after `preprocess_all`.
    """

    import ast
    class Transformer(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign):
            # print(ast.dump(node))
            for lhs in node.targets:
                # USE_NV_BINDING = True always
                if isinstance(lhs,ast.Name):
                    if lhs.id == "USE_NV_BINDING":
                        node.value = mr.to_ast_node("True")
                    # TODO(workaround,remove after hip-python-as-cuda fixes it):
                    elif lhs.id == "jitty":
                        # 'jitty = binding.CUjitInputType' -> 'jitty = binding2.CUjitInputType'
                        node.value = mr.to_ast_node("binding2.CUjitInputType",lineno=node.lineno)
            return node

        def visit_FunctionDef(self, node: ast.FunctionDef):
            if node.name in (
                "_stream_callback",
            ):
                return None # kick that one out, not used with USE_NV_BINDING
            return node # keep the others

        def visit_ImportFrom(self, node: ast.AST):
            """AST dump of typical nodes that we handle:
            ImportFrom(
                module='dep',
                names=[
                    alias(name='var1', asname='aliased_var1'),
                    alias(name='var2')],
                level=0),

            Assign(
                targets=[
                    Name(id='orig_var1', ctx=Store())],
                value=Constant(value='orig_var1')),
            """
            #  print(ast.dump(node))
            if node.module != None: 
                loc = mr.get_loc(node)
                if node.module == "cuda":
                    return mr.to_ast_node("from cuda import cuda as binding, nvrtc as binding2",**loc)
                elif node.module.endswith("drvapi"):
                    # 'from .drvapi import <VARS>' -> constants <VARS> 
                    result = []
                    for alias in node.names:
                        if alias.asname != None:
                            local_name = alias.asname
                        else:
                            local_name = alias.name
                        result.append(
                            ast.Assign(
                                targets=[
                                    ast.Name(id=local_name,ctx=ast.Store(),**loc),
                                ],
                                value=ast.Constant(value=None,**loc),
                                **loc
                            )
                        )
                    return result
            return node

    result = Transformer().visit(
        compile(content, "<string>", "exec", ast.PyCF_ONLY_AST)
    )
    #print(ast.unparse(result))
    return result
 
# order is important here!

_extras = mr.create_and_register_derived_module(
    "_extras",
    from_file=False,
    module_content="CUDA_IPC_HANDLE_SIZE=HIP_IPC_HANDLE_SIZE=64\n"
)  # make this a submodule of the package

enums = mr.create_and_register_derived_module(
    "enums"
)  # make this a submodule of the package

error = mr.create_and_register_derived_module(
    "error"
)  # make this a submodule of the package

drvapi = mr.create_and_register_derived_module(
    "drvapi",
)  # make this a submodule of the package
for k,v in list(drvapi.__dict__.items()):
    if k.startswith("cu_"):
        drvapi.__dict__["hip_"+k[3:]] = v
    elif k.startswith("CU_"):
        drvapi.__dict__["HIP_"+k[3:]] = v
    elif k == "API_PROTOTYPES":
        for api,args in list(v.items()):
            # FIXME some of the API are not supported by HIP
            assert api.startswith("cu")
            v["hip"+api[2:]] = args

rtapi = mr.create_and_register_derived_module(
    "rtapi",
)  # make this a submodule of the package
for api,args in list(rtapi.__dict__["API_PROTOTYPES"].items()):
    # FIXME some of the API are not supported by HIP
    assert api.startswith("cuda")
    v["hip"+api[2:]] = args

nvrtc = mr.create_and_register_derived_module(
    "nvrtc",
)  # make this a submodule of the package

driver = mr.create_and_register_derived_module(
    "driver", preprocess=_preprocess_driver
)  # make this a submodule of the package

devices = mr.create_and_register_derived_module(
    "devices"
)  # make this a submodule of the package

devicearray = mr.create_and_register_derived_module(
    "devicearray"
)  # make this a submodule of the package

ndarray = mr.create_and_register_derived_module(
    "ndarray"
)  # make this a submodule of the package

from . import nvvm

# clean up
del _preprocess_driver
del mr
del _modulerepl
del os
del re