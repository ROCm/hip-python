/* MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Freestanding replacement for the compiler-provided <stddef.h>.
 *
 * See the package README for why these headers exist. Every type is spelled
 * through a clang predefine rather than a concrete C type, so a parse for a
 * Windows target gets Windows' widths and a parse for an LP64 target gets
 * LP64's -- naming `unsigned long` here would reintroduce exactly the
 * host-dependence the generator exists to avoid.
 */

#ifndef __INTERFACEGEN_STDDEF_H
#define __INTERFACEGEN_STDDEF_H

typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;
#ifndef __cplusplus
typedef __WCHAR_TYPE__ wchar_t;
#endif

/* Library headers guard their own fallback definitions on these rather than
 * on an include guard, and define the type themselves when they are absent. */
#define __size_t_defined 1
#define _SIZE_T_DEFINED 1
#define __DEFINED_size_t 1
#define _PTRDIFF_T_DEFINED 1
#define __DEFINED_ptrdiff_t 1
#define _WCHAR_T_DEFINED 1
#define __DEFINED_wchar_t 1

#ifndef NULL
#ifdef __cplusplus
#define NULL __null
#else
#define NULL ((void *)0)
#endif
#endif

#define offsetof(t, d) __builtin_offsetof(t, d)

typedef struct {
  long long __ll;
  long double __ld;
} max_align_t;

#endif /* __INTERFACEGEN_STDDEF_H */
