#!/usr/bin/env python3
import os
import glob
import argparse
import re

def _remove_unwanted(filepaths):
    """Removes files that do not have a useful include.
    """
    result = []
    for f in filepaths:
        if f.endswith("ExternC.h"):
            continue
        if "llvm" not in f:
            continue
        result.append(f)
    return result

def _collect_files(llvm_inc_dir: str):
    glob_expr = os.path.join(llvm_inc_dir,"llvm-c","**","*.h")
    result = _remove_unwanted(glob.glob(glob_expr, recursive=True))
    llvm_config_path = os.path.join(llvm_inc_dir,"llvm","Config","llvm-config.h")
    result.insert(0,llvm_config_path)
    return result

def _collect_includes_per_file(filepath: str):
    includes = []
    include_expr = re.compile(r'\s*#\s*include\s+["<](?P<file>[A-Za-z0-9_\-/]+(\.h\w*)?)[">]')
    with open(filepath,"r") as infile:
        for line in infile.readlines():
            match = include_expr.match(line)
            if match:
                includes.append(match.group("file"))
    return includes

def build_include_graph(llvm_inc_dir: str):
    include_graph_nodes = dict()
    for abspath in _collect_files(llvm_inc_dir):
        relpath = abspath.replace(llvm_inc_dir,"").lstrip("/")
        include_graph_nodes[relpath] = _remove_unwanted(
            _collect_includes_per_file(abspath)
        )
    return include_graph_nodes

if __name__ == "__main__":
    import pprint
    def build_cli():
        parser = argparse.ArgumentParser("_include_graph.py")
        parser.add_argument("rocm_dir",type=str)#
        return parser

    args = build_cli().parse_args()
    ROCM_DIR = args.rocm_dir
    ROCM_LLVM_INC_DIR = os.path.join(ROCM_DIR,"llvm","include")
    include_graph_nodes = build_include_graph(ROCM_LLVM_INC_DIR)
    pprint.pprint(include_graph_nodes,indent=2)