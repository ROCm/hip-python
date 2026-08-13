# Freestanding builtin headers

`stddef.h`, `stdint.h` and `stdbool.h` are supplied by the *compiler*, not by
the C library, and live in clang's resource directory. The `libclang` wheel on
PyPI ships `libclang.so` on its own and no resource directory, so a parse
driven by it cannot resolve them.

Clang reports the miss as a fatal diagnostic and then keeps going, which is the
dangerous part: `size_t` and `ptrdiff_t` are not simply absent, they recover to
`int`. A generator reading that AST emits a 32-bit binding for a 64-bit API
without anything having failed. On an LP64 host the callee then writes eight
bytes into the four-byte slot behind every `size_t *` out-parameter.

`CParser` adds this directory to the include path when no `-resource-dir` was
supplied, so the fallback is invisible to any caller that has a real toolchain
-- the production codegen passes `--clang-resource-dir` and never sees these
files. `CParser` also raises on fatal diagnostics now, so a gap this set does
not cover fails loudly rather than silently changing a width.

Two notes for a Windows host. The fallback still wins there, since `-isystem`
is searched ahead of the platform's own headers: `stdint.h` resolves here
rather than in the UCRT. What cannot be demonstrated on such a host is the
*absence* of these headers, because clang targeting MSVC finds `stddef.h` in
the Windows SDK by detecting the Visual Studio installation, and predeclares
`size_t` in MS-compatibility mode. Neither `-nostdsysteminc` nor an empty
`INCLUDE` suppresses that, so the tests that assert the miss name
`--target=x86_64-unknown-linux-gnu` rather than leaving the target to the
host.

Every type here is spelled through a clang predefine (`__SIZE_TYPE__`,
`__INT64_TYPE__`, ...) rather than a concrete C type. The predefines follow the
parse target, so a Windows target gets Windows' widths; writing
`unsigned long` would bake in the host's data model, which is the failure these
headers exist to prevent.

The `stdint.h` surface is complete -- limits and constant constructors as well
as the typedefs -- because the directory is on the system include path and so
shadows the platform's own `stdint.h` where one exists.
