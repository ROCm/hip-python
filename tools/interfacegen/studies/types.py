import tempfile

from interfacegen.cparser import CParser, Analysis

import clang.cindex

types = """
struct mystruct {
  int a;
  char* b;
  char c[5];
};

union myunion { 
  int a[2]; 
  double f; 
};

enum myenum { 
  a = 0 
};

enum { 
  b = 1
}; // unnamed enum

struct { int a; } anon_struct_field;
"""

nested_types = """
struct outer {
  struct {
    int a;
  }; // no type and field name
  struct {
    int a;
  } b; // no type name
  struct inner {
    int a;
  } c; // type and field name
  struct {
    int a;
  }* d;
  struct inner2 {
    int a;
  }* e;
};
"""

typedefs = """
typedef struct {
  int a;
} anon_inner_t; // no inner type name, legal

typedef struct same_name { 
  int a;
} same_name; // same name, legal

typedef struct name {
  int a;
} different_name; // different name, legal

// pointers
struct defined {
  int a;
}
typedef struct defined* myptr1; // defined struct
typedef struct undefined* myptr2; // undefined struct, legal
typedef struct myptr3* mytptr3; //  (defined/undefined) struct with same name, legal
//typedef struct* myptr4; // no inner name and no definition, illegal
typedef struct { int a; }* myptr4; // inner definition, legal
typedef struct {  }* myptr5; // empty inner definition, legal
typedef struct mytype6 { int a; }* myptr6;
typedef struct mytype7 { int a; }*  mytype7;

// enums
typedef enum Value { a = 0 } Value_t;
typedef union Union { int a[2]; double f; } Union_t;

typedef const int cint;
cint a;

typedef enum { 
  HSA_STATUS_SUCCESS = 0x0,
} hsa_status_t;

// function pointers
typedef int (*foo)(const int* event, void* data);
typedef hsa_status_t (*hsa_amd_system_event_callback_t)(const hsa_status_t* event, void* data);
"""

functions = """\
typedef int (*fptr)(int a, int b);

void foo(float a,
         fptr b,
         void (*c)(void *),
         void (*d) (void (*d1)(int,int)),
         void (*e) (void (*)(int,int)) // unnamed funptr
                                       // in parm list
         );

struct mystruct {
  fptr a;
  void (*c)(void *);
  void (*d) (void (*d1)(int,int));
};
"""

file_content = functions
file_content = types
file_content = typedefs

parser = CParser("input.h",unsaved_files=[("input.h",file_content)])
parser.parse()

print(Analysis.subtree_as_csv(parser.cursor,None,5))