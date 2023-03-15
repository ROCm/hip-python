#AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

import tempfile

import addtoplevelpath
from _codegen.cparser import CParser
    
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
};

int test(struct nested_outer n, simple_t s);

typedef unsigned int GLuint;
"""

with tempfile.NamedTemporaryFile("w",suffix=".c") as tmpfile_handle:
    with tmpfile_handle.file as tmpfile:
        tmpfile.write(file_content)
    parser = CParser(tmpfile.name)
    parser.parse()
    print(parser.render_cursors())
