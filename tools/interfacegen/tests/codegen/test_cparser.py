# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

import tempfile

import addtoplevelpath
from _codegen.cparser import CParser

file_content = """
#define macro 0

#define macro2 0.2

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
  struct simple field2;
  simple_t field3;
};

typedef simple_t* simple_t_ptr;

int test(simple_t **out1, simple_t_ptr *out2, int a, const int* b, struct nested_outer c, const nested_outer* d, simple_t e, const simple_t* f, const simple_t** const g, const int h[10], const int i[], const int* j[], const simple_t k[], const int* m = 0);

int test2(void);

typedef unsigned int GLuint;

typedef struct incomplete_t* pointer_t_1;

typedef const struct incomplete_t*[20] pointer_t_2;

typedef const struct incomplete_t*[] pointer_t_3;

typedef int* int_ptr;

typedef const int*[20] int_ptr2;

typedef const int*[] int_ptr3;
"""

file_name = "input.h"
parser = CParser(file_name, unsaved_files=[(file_name, file_content)])
parser.parse()
print(parser.render_cursors())
