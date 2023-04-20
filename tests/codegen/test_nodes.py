# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

import tempfile

import addtoplevelpath
from _codegen.cparser import CParser
from _codegen import nodes

file_content = """
#define macro 0

struct simple  {
  int field1;
};

typedef struct typedefed_simple  {
  int field1;
} simple_t;
  
struct nested_outer {
  struct nested_inner1 {
    int field1;
  } field1;

  simple_t field2;
};

int test(struct nested_outer n, simple_t s);

typedef unsigned int GLuint;
"""

file_name = "input.h"
parser = CParser(file_name, unsaved_files=[(file_name, file_content)])
parser.parse()


def node_filter(node: nodes.Node):
    if node.name == "macro":
        return True
    elif node.file == file_name:
        return True
    return False


for node in nodes.create_nodes(parser, node_filter):
    print(node.name)
# print(parser.render_cursor())
