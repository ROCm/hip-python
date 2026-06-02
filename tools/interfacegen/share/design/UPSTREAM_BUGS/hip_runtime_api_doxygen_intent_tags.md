# Bug report: incorrect `@param[in|out|in,out]` tags in `hip_runtime_api.h` (and `amd_smi/amdsmi.h`)

**Filed against:**
- [`ROCm/HIP`](https://github.com/ROCm/HIP) for `hip_runtime_api.h`
- [`ROCm/amdsmi`](https://github.com/ROCm/amdsmi) for `amd_smi/amdsmi.h`

**Encountered on:** 2026-05-08 (Families 1–3); 2026-06-02 (Family 4,
caller-provided input pointers tagged `@param[out]`)

**Affected ROCm version:** 7.13.0 (`/opt/rocm/.info/version` reports
`7.13.0`; line numbers below are from this checkout). The same
mistagging patterns are present in older releases — none of the
affected `@param[…]` annotations have been touched in years.

**Affected files:** `/opt/rocm/include/hip/hip_runtime_api.h` and
`/opt/rocm/include/amd_smi/amdsmi.h`.

## Title

`hip_runtime_api.h` — `@param[out] dst` for `hipMemcpy*`,
`@param[in, out]` for opaque-handle creators (`hipStreamCreate`,
`hipEventCreate`, `hipModuleLoad*`, `hipMalloc*`, `hipMemPool*`,
`hipGraph*`, `hipImport*`), and `@param[out]` for caller-provided
**input** pointers (`hipHostRegister`, `hipMemcpyToSymbol*`)
misclassify parameter intent.

## Summary

Several families of public HIP runtime API functions carry
doxygen `@param[…]` tags whose intent does not match the C semantics:

1. **`hipMemcpy*` family** tags the destination buffer parameter as
   `@param[out]`, but it is in fact `@param[in,out]`: the caller
   pre-allocates the destination buffer (any rank: scalar, 1-D, 2-D,
   3-D, host- or device-resident), and `hipMemcpy*` writes into it.
   `[out]` semantics in the doxygen vocabulary mean callee-allocated /
   caller-readback (a fresh resource is produced); that is not what
   `hipMemcpy*` does.

2. **Opaque-handle creators** (`hipStreamCreate*`, `hipEventCreate*`,
   `hipModuleLoad*`, `hipMalloc*`, `hipMemPool*`, `hipGraph*`,
   `hipImport*`, `hipCtxCreate`, `hipDevicePrimaryCtxRetain`,
   `hipExternalMemoryGetMappedBuffer`, `hipGetSymbol*`, `hipModuleGet*`,
   …) tag the handle output as `@param[in, out]`. The C contract is
   pure `[out]`: the caller does not pre-populate the destination; the
   function writes the new handle and the prior contents (if any) are
   discarded.

3. **Caller-provided input pointers** (`hipHostRegister`,
   `hipMemcpyToSymbol`, `hipMemcpyToSymbolAsync`) tag a read-only / only-read
   input pointer as `@param[out]`. The C contract is `[in]`: the caller
   allocates/owns the memory and the function merely registers it
   (`hipHostRegister`) or reads the destination symbol address
   (`hipMemcpyToSymbol*`, where the parameter is even `const`-qualified —
   `const void* symbol` — which directly contradicts `[out]`).

These mistags break any tooling that derives parameter intent from
doxygen tags (binding generators, swagger-style documentation walkers,
static analyzers). Specifically, the AMD `hip-python` codegen
(`/src/interfacegen`) recently added a generic
`documented_param_intent` rule at the head of every per-library intent
chain. Trusting these tags causes the regenerated python bindings to:
- drop `dst` from `hipMemcpy*` and present it as a callee-allocated
  return value (wrong — there is no allocation function called); and
- drag the OUT handle pointer of `hipStreamCreate` and friends into the
  argument list (wrong — the handle is conceptually returned, not
  passed in); and
- drop the input pointer of `hipHostRegister` / force `hipMemcpyToSymbol*`'s
  `symbol` to `NULL` (wrong — these are required inputs; see Family 4 for
  the broken generated signatures).

We have had to special-case all three families in the codegen rule chain;
fixing the tags upstream lets us drop the workaround.

## Reproducer (text inspection)

```text
$ awk 'NR==4701' /opt/rocm/include/hip/hip_runtime_api.h
 *  @param[out]  dst Data being copy to

$ awk 'NR==4714' /opt/rocm/include/hip/hip_runtime_api.h
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);
```

```text
$ awk 'NR==2819' /opt/rocm/include/hip/hip_runtime_api.h
 * @param[in, out] stream  Valid pointer to hipStream_t.  This function writes the memory with the

$ awk 'NR==2834' /opt/rocm/include/hip/hip_runtime_api.h
hipError_t hipStreamCreate(hipStream_t* stream);
```

## Suggested fix

### Family 1 — `hipMemcpy*` `dst` (and the parallel `dstHost`, `dstDevice`, `dstArray`, `dsts`)

`@param[out]` → `@param[in,out]`.

Affected entry points (function names; the same mismatch exists on the
`dst`-style parameter of each):

```
hipMemcpy
hipMemcpyAsync
hipMemcpyWithStream
hipMemcpyHtoD
hipMemcpyDtoH
hipMemcpyDtoD
hipMemcpyHtoDAsync
hipMemcpyDtoHAsync
hipMemcpyDtoDAsync
hipMemcpyAtoD
hipMemcpyDtoA
hipMemcpyAtoA
hipMemcpyAtoH
hipMemcpyHtoA
hipMemcpyAtoHAsync
hipMemcpyHtoAAsync
hipMemcpyToSymbol
hipMemcpyToSymbolAsync
hipMemcpyFromSymbol
hipMemcpyFromSymbolAsync
hipMemcpy2D
hipMemcpy2DAsync
hipMemcpy2DToArray
hipMemcpy2DToArrayAsync
hipMemcpy2DArrayToArray
hipMemcpy2DFromArray
hipMemcpy2DFromArrayAsync
hipMemcpyToArray
hipMemcpyFromArray
hipMemcpyPeer
hipMemcpyPeerAsync
hipMemcpyBatchAsync
```

(Plus any future `hipMemcpy*` variant — the rule is uniform.)

The **`hipMemset*` family** shares the identical mismatch on its
destination buffer (`@param[out]` → `@param[in,out]`): the caller
allocates the device buffer and the function fills it.

```
hipMemset
hipMemsetAsync
hipMemsetD8
hipMemsetD8Async
hipMemsetD16
hipMemsetD16Async
hipMemsetD32
hipMemsetD32Async
hipMemset2D
hipMemset2DAsync
hipMemsetD2D8
hipMemsetD2D8Async
hipMemsetD2D16
hipMemsetD2D16Async
hipMemsetD2D32
hipMemsetD2D32Async
```

**Destination-name inconsistency (important for any name-based tooling).**
The destination parameter is *not* named uniformly across this family.
Most use `dst`, but **`hipMemsetD8`, `hipMemsetD8Async`, `hipMemsetD16`,
`hipMemsetD16Async`, and `hipMemsetD32` name it `dest`** — while the
otherwise-parallel `hipMemsetD32Async` uses `dst`. A binding generator
that keys its INOUT override on the parameter name (as `hip-python` does
via `_HIPMEMCPY_INOUT_DST_NAMES`) must list **both** `dst` and `dest`, or
the five `dest` variants fall through to the `@param[out]` tag and the
generated binding drops the buffer entirely (it memsets `NULL`). Aligning
the upstream parameter name to `dst` everywhere would also remove this
foot-gun.

### Family 2 — opaque-handle creators

`@param[in, out]` (or `@param[in,out]`) → `@param[out]` for the leading
handle-pointer parameter.

Sample (not exhaustive) entry points and the line numbers of the
mistagged `@param[in, out] stream`-style parameters in the current
header:

| line | function | parameter |
|------|----------|-----------|
| 2819 | `hipStreamCreate` | `stream` |
| 2838 | `hipStreamCreateWithFlags` | `stream` |
| 2859 | `hipStreamCreateWithPriority` | `stream` |

Other functions in the same family (intent fix is identical — first
output parameter of an opaque-handle creator should be `[out]`):

```
hipEventCreate
hipEventCreateWithFlags
hipExtStreamCreateWithCUMask
hipMalloc
hipExtMallocWithFlags
hipMallocHost
hipMemAllocHost
hipHostMalloc
hipHostAlloc
hipMallocManaged
hipMallocAsync
hipMallocFromPoolAsync
hipMallocArray
hipMalloc3DArray
hipMallocPitch
hipMemAllocPitch
hipMemPoolCreate
hipMemPoolImportFromShareableHandle
hipMemPoolImportPointer
hipModuleLoad
hipModuleLoadData
hipModuleLoadDataEx
hipModuleLoadFatBinary
hipModuleGetFunction
hipModuleGetGlobal
hipModuleGetTexRef
hipGetSymbolAddress
hipGetSymbolSize
hipGraphCreate
hipGraphClone
hipGraphInstantiate
hipGraphInstantiateWithFlags
hipGraphInstantiateWithParams
hipGraphAddNode
hipGraphAddKernelNode
hipGraphAddMemcpyNode
hipGraphAddMemcpyNode1D
hipGraphAddMemcpyNodeFromSymbol
hipGraphAddMemcpyNodeToSymbol
hipGraphAddMemsetNode
hipGraphAddHostNode
hipGraphAddChildGraphNode
hipGraphAddEmptyNode
hipGraphAddEventRecordNode
hipGraphAddEventWaitNode
hipGraphAddMemAllocNode
hipGraphAddMemFreeNode
hipGraphAddBatchMemOpNode
hipGraphAddExternalSemaphoresWaitNode
hipGraphAddExternalSemaphoresSignalNode
hipImportExternalMemory
hipImportExternalSemaphore
hipExternalMemoryGetMappedBuffer
hipCtxCreate
hipDevicePrimaryCtxRetain
hipCreateTextureObject
hipCreateSurfaceObject
hipUserObjectCreate
```

For `hipMallocPitch` and `hipMemAllocPitch`, the first **two**
parameters (devPtr + pitch) are pure `[out]` for the same reason.

### Family 3 — `amdsmi.h` overloaded `@param[in,out]`

`/opt/rocm/include/amd_smi/amdsmi.h` uses `@param[in,out]` for two
distinct semantic cases without distinguishing them in the tag:

1. **Genuine INOUT** — the two-call count-then-fill pattern, where
   the caller pre-populates a buffer-size value and the function
   consumes/updates it. Example: `socket_count` in
   `amdsmi_get_socket_handles`. The tag is correct here.

2. **Pure OUT mistagged as `[in,out]`** — a fixed-size output buffer
   or pointer-to-scalar that the caller does NOT pre-populate; the
   function just writes. Tools that trust the tag drag the OUT
   pointer back into the python args (or equivalent), forcing every
   caller to construct an empty placeholder.

Confirmed instances of the second case (non-exhaustive — a sweep of
the header is in flight):

| function | parameter | actual semantics |
|---|---|---|
| `amdsmi_get_lib_version` | `version` | OUT (writes into a fresh `amdsmi_version_t`) |
| `amdsmi_get_gpu_id` | `id` | OUT (writes into a fresh `uint16_t*`) |
| `amdsmi_get_gpu_subsystem_id` | `id` | OUT (same shape) |
| `amdsmi_get_gpu_bdf_id` | `bdfid` | OUT (writes into a fresh `uint64_t*`) |
| `amdsmi_get_gpu_topo_numa_affinity` | `numa_node` | OUT |
| `amdsmi_get_gpu_pci_replay_counter` | `counter` | OUT |
| `amdsmi_get_gpu_virtualization_mode` | `mode` | OUT (writes into an enum slot) |
| `amdsmi_get_gpu_pci_throughput` | `sent`, `received`, `max_pkt_sz` | OUT (3 separate fixed-size scalar outs) |
| `amdsmi_get_energy_count` | `energy_accumulator`, `counter_resolution`, `timestamp` | OUT (3 separate fixed-size scalar outs) |

Suggested fix: split `[in,out]` into the correct `[out]` for these,
keeping `[in,out]` only for the genuine count-then-fill pattern. A
clean separation makes the codegen and any documentation tooling
deterministic.

### Family 4 — `hip_runtime_api.h` caller-provided input pointers tagged `@param[out]`

*(Confirmed 2026-06-02 on ROCm 7.13.0; line numbers from this checkout.)*

The mirror image of Family 1: instead of an `[out]` that should be
`[in,out]`, these are an `@param[out]` that should be a plain `@param[in]`.
The parameter is a caller-provided **input** pointer — the caller owns the
memory and the function only reads it (or just reads the pointer value).
`[out]` semantics (callee-produces-a-fresh-result) are simply wrong here.

| line | function | parameter | C type | actual semantics |
|------|----------|-----------|--------|------------------|
| 4596 | `hipHostRegister` | `hostPtr` | `void *` | IN — caller allocates the host memory; the call only registers it for device access. (`hipHostUnregister` correctly tags the same pointer `@param[in]`.) |
| 5068 | `hipMemcpyToSymbol` | `symbol` | `const void *` | IN — the destination device symbol *address* is an input; the `const` qualifier already proves read-only, directly contradicting `[out]`. |
| 5084 | `hipMemcpyToSymbolAsync` | `symbol` | `const void *` | IN — same as `hipMemcpyToSymbol`. |

Suggested fix: `@param[out]` → `@param[in]` for each of the three
parameters above.

**Why this one is not merely cosmetic — it produces broken bindings.**
Because `documented_param_intent` trusts the `[out]` tag, the generated
`hip-python` bindings before the workaround were:

```text
# hostPtr is dropped from the signature entirely — the buffer to register
# can never be passed:
def hipHostRegister(unsigned long sizeBytes, unsigned int flags): ...

# symbol is initialized to NULL, passed to the C call as NULL, and returned
# instead of accepted — hipMemcpyToSymbol can never target a real symbol:
def hipMemcpyToSymbol(object src, unsigned long sizeBytes, unsigned long offset, object kind):
    cdef ...Pointer symbol = ...Pointer.fromPtr(NULL)
    ...
    return (hipError_t(...), None if symbol._ptr == NULL else symbol)
```

## Why this is more than cosmetic

Doxygen `@param[…]` tags are increasingly consumed by automated
binding generators that derive a Pythonic / TypeScript / Rust-style
signature where IN params are arguments and OUT params are returned.
Mistagging an INOUT buffer as OUT, or an OUT handle as INOUT, leads
those generators to silently produce APIs that don't compile against
the real C contract — and the divergence is hard to spot in review
because the doxygen reads as authoritative.

A one-time pass over `hip_runtime_api.h` to align tags with the
documented C contract (which is itself unambiguous about which
parameters are caller-allocated vs callee-written) closes the loop.

## Workaround in `hip-python` until the fix lands

`/src/interfacegen/python/interfacegen/support/recipes/rocm.py`:

- **`class hip.ptr_parm_intent`** hardcodes the correct intent for the
  hip-runtime families (Memcpy/Memset INOUT, opaque-handle creators
  OUT, and the Family-4 input pointers `hipHostRegister` /
  `hipMemcpyToSymbol*` forced to IN) and short-circuits the generic
  `documented_param_intent` rule via the `@fallback` decorator. See the
  comment block immediately above the `_HIPMEMCPY_INOUT_DST_NAMES` /
  `_HIP_HANDLE_CREATOR_OUT_PARM0` definitions for the full rationale, and
  the explicit `(func_name, parm_idx) -> ParmIntent.IN` tuple for Family 4.

- **`class hipfft.ptr_parm_intent`** hardcodes `odata` in the
  `hipfftExec*` family as INOUT (the upstream tag `@param[out]` is
  wrong — `odata` is a caller-allocated device buffer that the
  function fills).

- **`class amdsmi.ptr_parm_intent`** carries an explicit
  `_MISTAGGED_OUT` set of `(function_name, parm_index)` tuples that
  the recipe forces to OUT before consulting the doxygen rule. New
  entries are appended as additional Family-3 cases are confirmed.
  The audit test `test_real_amdsmi_doxygen_audit_zero_unintended_mismatches`
  in `python/interfacegen/test/test_amdsmi_doxygen_intent.py`
  enforces that the override set is the only deviation from the
  doxygen tags.

When the upstream tags are fixed, each override block can be removed
and the generic rule will produce the correct intent automatically.
