/*===-- llvm-c/Target.h - Target Lib C Iface --------------------*- C++ -*-===*/
/*                                                                            */
/* Part of the LLVM Project, under the Apache License v2.0 with LLVM          */
/* Exceptions.                                                                */
/* See https://llvm.org/LICENSE.txt for license information.                  */
/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    */
/*                                                                            */
// 
/*===----------------------------------------------------------------------===*/
/*                                                                            */
/* This header declares the C interface to libLLVMTarget.a, which             */
/* implements target information.                                             */
/*                                                                            */
/* Many exotic languages can interoperate with C code but have a harder time  */
/* with C++ due to name mangling. So in addition to C, this interface enables */
/* tools written in such languages.                                           */
/*                                                                            */
/*===----------------------------------------------------------------------===*/

// MIT License
// 
// Modifications Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef LLVM_C_TARGET_H
#define LLVM_C_TARGET_H

#include "llvm-c/ExternC.h"
#include "llvm-c/Types.h"
#include "llvm/Config/llvm-config.h"

LLVM_C_EXTERN_C_BEGIN

/**
 * @defgroup LLVMCTarget Target information
 * @ingroup LLVMC
 *
 * @{
 */

enum LLVMByteOrdering { LLVMBigEndian, LLVMLittleEndian };

typedef struct LLVMOpaqueTargetData *LLVMTargetDataRef;
typedef struct LLVMOpaqueTargetLibraryInfotData *LLVMTargetLibraryInfoRef;

/* Declare all of the target-initialization functions that are available. */
#define LLVM_TARGET(TargetName) \
  void LLVMInitialize##TargetName##TargetInfo(void);
#include "llvm/Config/Targets.def"
#undef LLVM_TARGET  /* Explicit undef to make SWIG happier */

#define LLVM_TARGET(TargetName) void LLVMInitialize##TargetName##Target(void);
#include "llvm/Config/Targets.def"
#undef LLVM_TARGET  /* Explicit undef to make SWIG happier */

#define LLVM_TARGET(TargetName) \
  void LLVMInitialize##TargetName##TargetMC(void);
#include "llvm/Config/Targets.def"
#undef LLVM_TARGET  /* Explicit undef to make SWIG happier */

/* Declare all of the available assembly printer initialization functions. */
#define LLVM_ASM_PRINTER(TargetName) \
  void LLVMInitialize##TargetName##AsmPrinter(void);
#include "llvm/Config/AsmPrinters.def"
#undef LLVM_ASM_PRINTER  /* Explicit undef to make SWIG happier */

/* Declare all of the available assembly parser initialization functions. */
#define LLVM_ASM_PARSER(TargetName) \
  void LLVMInitialize##TargetName##AsmParser(void);
#include "llvm/Config/AsmParsers.def"
#undef LLVM_ASM_PARSER  /* Explicit undef to make SWIG happier */

/* Declare all of the available disassembler initialization functions. */
#define LLVM_DISASSEMBLER(TargetName) \
  void LLVMInitialize##TargetName##Disassembler(void);
#include "llvm/Config/Disassemblers.def"
#undef LLVM_DISASSEMBLER  /* Explicit undef to make SWIG happier */

/** LLVMInitializeAllTargetInfos - The main program should call this function if
    it wants access to all available targets that LLVM is configured to
    support. */
void LLVMInitializeAllTargetInfos(void) {
#define LLVM_TARGET(TargetName) LLVMInitialize##TargetName##TargetInfo();
#include "llvm/Config/Targets.def"
#undef LLVM_TARGET  /* Explicit undef to make SWIG happier */
}

/** LLVMInitializeAllTargets - The main program should call this function if it
    wants to link in all available targets that LLVM is configured to
    support. */
void LLVMInitializeAllTargets(void) {
#define LLVM_TARGET(TargetName) LLVMInitialize##TargetName##Target();
#include "llvm/Config/Targets.def"
#undef LLVM_TARGET  /* Explicit undef to make SWIG happier */
}

/** LLVMInitializeAllTargetMCs - The main program should call this function if
    it wants access to all available target MC that LLVM is configured to
    support. */
void LLVMInitializeAllTargetMCs(void) {
#define LLVM_TARGET(TargetName) LLVMInitialize##TargetName##TargetMC();
#include "llvm/Config/Targets.def"
#undef LLVM_TARGET  /* Explicit undef to make SWIG happier */
}

/** LLVMInitializeAllAsmPrinters - The main program should call this function if
    it wants all asm printers that LLVM is configured to support, to make them
    available via the TargetRegistry. */
void LLVMInitializeAllAsmPrinters(void) {
#define LLVM_ASM_PRINTER(TargetName) LLVMInitialize##TargetName##AsmPrinter();
#include "llvm/Config/AsmPrinters.def"
#undef LLVM_ASM_PRINTER  /* Explicit undef to make SWIG happier */
}

/** LLVMInitializeAllAsmParsers - The main program should call this function if
    it wants all asm parsers that LLVM is configured to support, to make them
    available via the TargetRegistry. */
void LLVMInitializeAllAsmParsers(void) {
#define LLVM_ASM_PARSER(TargetName) LLVMInitialize##TargetName##AsmParser();
#include "llvm/Config/AsmParsers.def"
#undef LLVM_ASM_PARSER  /* Explicit undef to make SWIG happier */
}

/** LLVMInitializeAllDisassemblers - The main program should call this function
    if it wants all disassemblers that LLVM is configured to support, to make
    them available via the TargetRegistry. */
void LLVMInitializeAllDisassemblers(void) {
#define LLVM_DISASSEMBLER(TargetName) \
  LLVMInitialize##TargetName##Disassembler();
#include "llvm/Config/Disassemblers.def"
#undef LLVM_DISASSEMBLER  /* Explicit undef to make SWIG happier */
}

/** LLVMInitializeNativeTarget - The main program should call this function to
    initialize the native target corresponding to the host.  This is useful
    for JIT applications to ensure that the target gets linked in correctly. */
LLVMBool LLVMInitializeNativeTarget(void) {
  /* If we have a native target, initialize it to ensure it is linked in. */
#ifdef LLVM_NATIVE_TARGET
  LLVM_NATIVE_TARGETINFO();
  LLVM_NATIVE_TARGET();
  LLVM_NATIVE_TARGETMC();
  return 0;
#else
  return 1;
#endif
}

/** LLVMInitializeNativeTargetAsmParser - The main program should call this
    function to initialize the parser for the native target corresponding to the
    host. */
LLVMBool LLVMInitializeNativeAsmParser(void) {
#ifdef LLVM_NATIVE_ASMPARSER
  LLVM_NATIVE_ASMPARSER();
  return 0;
#else
  return 1;
#endif
}

/** LLVMInitializeNativeTargetAsmPrinter - The main program should call this
    function to initialize the printer for the native target corresponding to
    the host. */
LLVMBool LLVMInitializeNativeAsmPrinter(void) {
#ifdef LLVM_NATIVE_ASMPRINTER
  LLVM_NATIVE_ASMPRINTER();
  return 0;
#else
  return 1;
#endif
}

/** LLVMInitializeNativeTargetDisassembler - The main program should call this
    function to initialize the disassembler for the native target corresponding
    to the host. */
LLVMBool LLVMInitializeNativeDisassembler(void) {
#ifdef LLVM_NATIVE_DISASSEMBLER
  LLVM_NATIVE_DISASSEMBLER();
  return 0;
#else
  return 1;
#endif
}

/*===-- Target Data -------------------------------------------------------===*/
LLVM_C_EXTERN_C_END

#endif