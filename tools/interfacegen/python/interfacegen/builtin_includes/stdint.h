/* MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Freestanding replacement for the compiler-provided <stdint.h>.
 *
 * See the package README for why these headers exist. Every type is spelled
 * through a clang predefine rather than a concrete C type, so a parse for a
 * Windows target gets Windows' widths and a parse for an LP64 target gets
 * LP64's -- naming `unsigned long` here would reintroduce exactly the
 * host-dependence the generator exists to avoid.
 *
 * The whole C99 surface is present, limit and constructor macros included,
 * because this file shadows the platform's <stdint.h> wherever it is on the
 * search path. A header that reaches for UINT32_MAX has to find it here.
 */

#ifndef __INTERFACEGEN_STDINT_H
#define __INTERFACEGEN_STDINT_H

/* Exact-width */

typedef __INT8_TYPE__ int8_t;
typedef __INT16_TYPE__ int16_t;
typedef __INT32_TYPE__ int32_t;
typedef __INT64_TYPE__ int64_t;
typedef __UINT8_TYPE__ uint8_t;
typedef __UINT16_TYPE__ uint16_t;
typedef __UINT32_TYPE__ uint32_t;
typedef __UINT64_TYPE__ uint64_t;

/* Minimum-width */

typedef __INT_LEAST8_TYPE__ int_least8_t;
typedef __INT_LEAST16_TYPE__ int_least16_t;
typedef __INT_LEAST32_TYPE__ int_least32_t;
typedef __INT_LEAST64_TYPE__ int_least64_t;
typedef __UINT_LEAST8_TYPE__ uint_least8_t;
typedef __UINT_LEAST16_TYPE__ uint_least16_t;
typedef __UINT_LEAST32_TYPE__ uint_least32_t;
typedef __UINT_LEAST64_TYPE__ uint_least64_t;

/* Fastest minimum-width */

typedef __INT_FAST8_TYPE__ int_fast8_t;
typedef __INT_FAST16_TYPE__ int_fast16_t;
typedef __INT_FAST32_TYPE__ int_fast32_t;
typedef __INT_FAST64_TYPE__ int_fast64_t;
typedef __UINT_FAST8_TYPE__ uint_fast8_t;
typedef __UINT_FAST16_TYPE__ uint_fast16_t;
typedef __UINT_FAST32_TYPE__ uint_fast32_t;
typedef __UINT_FAST64_TYPE__ uint_fast64_t;

/* Pointer-sized and greatest-width */

typedef __INTPTR_TYPE__ intptr_t;
typedef __UINTPTR_TYPE__ uintptr_t;
typedef __INTMAX_TYPE__ intmax_t;
typedef __UINTMAX_TYPE__ uintmax_t;

#define __intptr_t_defined 1
#define _INTPTR_T_DEFINED 1
#define __DEFINED_intptr_t 1
#define _UINTPTR_T_DEFINED 1
#define __DEFINED_uintptr_t 1

/* Limits.
 *
 * Clang predefines only the maxima. A signed minimum is one below the
 * negation of its maximum, written as `-MAX - 1` so the literal itself stays
 * representable in the type. */

#define INT8_MAX __INT8_MAX__
#define INT16_MAX __INT16_MAX__
#define INT32_MAX __INT32_MAX__
#define INT64_MAX __INT64_MAX__
#define INT8_MIN (-__INT8_MAX__ - 1)
#define INT16_MIN (-__INT16_MAX__ - 1)
#define INT32_MIN (-__INT32_MAX__ - 1)
#define INT64_MIN (-__INT64_MAX__ - 1)
#define UINT8_MAX __UINT8_MAX__
#define UINT16_MAX __UINT16_MAX__
#define UINT32_MAX __UINT32_MAX__
#define UINT64_MAX __UINT64_MAX__

#define INT_LEAST8_MAX __INT_LEAST8_MAX__
#define INT_LEAST16_MAX __INT_LEAST16_MAX__
#define INT_LEAST32_MAX __INT_LEAST32_MAX__
#define INT_LEAST64_MAX __INT_LEAST64_MAX__
#define INT_LEAST8_MIN (-__INT_LEAST8_MAX__ - 1)
#define INT_LEAST16_MIN (-__INT_LEAST16_MAX__ - 1)
#define INT_LEAST32_MIN (-__INT_LEAST32_MAX__ - 1)
#define INT_LEAST64_MIN (-__INT_LEAST64_MAX__ - 1)
#define UINT_LEAST8_MAX __UINT_LEAST8_MAX__
#define UINT_LEAST16_MAX __UINT_LEAST16_MAX__
#define UINT_LEAST32_MAX __UINT_LEAST32_MAX__
#define UINT_LEAST64_MAX __UINT_LEAST64_MAX__

#define INT_FAST8_MAX __INT_FAST8_MAX__
#define INT_FAST16_MAX __INT_FAST16_MAX__
#define INT_FAST32_MAX __INT_FAST32_MAX__
#define INT_FAST64_MAX __INT_FAST64_MAX__
#define INT_FAST8_MIN (-__INT_FAST8_MAX__ - 1)
#define INT_FAST16_MIN (-__INT_FAST16_MAX__ - 1)
#define INT_FAST32_MIN (-__INT_FAST32_MAX__ - 1)
#define INT_FAST64_MIN (-__INT_FAST64_MAX__ - 1)
#define UINT_FAST8_MAX __UINT_FAST8_MAX__
#define UINT_FAST16_MAX __UINT_FAST16_MAX__
#define UINT_FAST32_MAX __UINT_FAST32_MAX__
#define UINT_FAST64_MAX __UINT_FAST64_MAX__

#define INTPTR_MAX __INTPTR_MAX__
#define INTPTR_MIN (-__INTPTR_MAX__ - 1)
#define UINTPTR_MAX __UINTPTR_MAX__
#define INTMAX_MAX __INTMAX_MAX__
#define INTMAX_MIN (-__INTMAX_MAX__ - 1)
#define UINTMAX_MAX __UINTMAX_MAX__

#define PTRDIFF_MAX __PTRDIFF_MAX__
#define PTRDIFF_MIN (-__PTRDIFF_MAX__ - 1)
#define SIZE_MAX __SIZE_MAX__
#define SIG_ATOMIC_MAX __SIG_ATOMIC_MAX__
#define SIG_ATOMIC_MIN (-__SIG_ATOMIC_MAX__ - 1)

#define WINT_MAX __WINT_MAX__
#ifdef __WINT_UNSIGNED__
#define WINT_MIN 0
#else
#define WINT_MIN (-__WINT_MAX__ - 1)
#endif

#define WCHAR_MAX __WCHAR_MAX__
#ifdef __WCHAR_UNSIGNED__
#define WCHAR_MIN 0
#else
#define WCHAR_MIN (-__WCHAR_MAX__ - 1)
#endif

/* Constant constructors.
 *
 * The suffix predefines expand to nothing for the narrow types, which token
 * pasting turns into a placemarker, leaving the value unchanged. */

#define __interfacegen_paste(a, b) a##b
#define __interfacegen_int_c(v, suffix) __interfacegen_paste(v, suffix)

#define INT8_C(v) __interfacegen_int_c(v, __INT8_C_SUFFIX__)
#define INT16_C(v) __interfacegen_int_c(v, __INT16_C_SUFFIX__)
#define INT32_C(v) __interfacegen_int_c(v, __INT32_C_SUFFIX__)
#define INT64_C(v) __interfacegen_int_c(v, __INT64_C_SUFFIX__)
#define UINT8_C(v) __interfacegen_int_c(v, __UINT8_C_SUFFIX__)
#define UINT16_C(v) __interfacegen_int_c(v, __UINT16_C_SUFFIX__)
#define UINT32_C(v) __interfacegen_int_c(v, __UINT32_C_SUFFIX__)
#define UINT64_C(v) __interfacegen_int_c(v, __UINT64_C_SUFFIX__)
#define INTMAX_C(v) __interfacegen_int_c(v, __INTMAX_C_SUFFIX__)
#define UINTMAX_C(v) __interfacegen_int_c(v, __UINTMAX_C_SUFFIX__)

#endif /* __INTERFACEGEN_STDINT_H */
