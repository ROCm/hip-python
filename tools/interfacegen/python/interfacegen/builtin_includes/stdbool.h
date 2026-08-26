/* MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Freestanding replacement for the compiler-provided <stdbool.h>.
 *
 * See the package README for why these headers exist.
 */

#ifndef __INTERFACEGEN_STDBOOL_H
#define __INTERFACEGEN_STDBOOL_H

#ifndef __cplusplus
#define bool _Bool
#define true 1
#define false 0
#endif

#define __bool_true_false_are_defined 1

#endif /* __INTERFACEGEN_STDBOOL_H */
