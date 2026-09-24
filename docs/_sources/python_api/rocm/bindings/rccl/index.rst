rocm.bindings.rccl
==================

.. py:module:: rocm.bindings.rccl


Attributes
----------

.. autoapisummary::

   rocm.bindings.rccl.NCCL_MAJOR
   rocm.bindings.rccl.NCCL_MINOR
   rocm.bindings.rccl.NCCL_PATCH
   rocm.bindings.rccl.NCCL_SUFFIX
   rocm.bindings.rccl.NCCL_VERSION_CODE
   rocm.bindings.rccl.RCCL_BFLOAT16
   rocm.bindings.rccl.RCCL_GATHER_SCATTER
   rocm.bindings.rccl.RCCL_ALLTOALLV
   rocm.bindings.rccl.NCCL_UNIQUE_ID_BYTES
   rocm.bindings.rccl.ncclComm_t
   rocm.bindings.rccl.ncclWindow_t
   rocm.bindings.rccl.ncclConfig_t
   rocm.bindings.rccl.ncclSimInfo_t
   rocm.bindings.rccl.ncclParamHandle_t


Classes
-------

.. autoapisummary::

   rocm.bindings.rccl.ncclComm
   rocm.bindings.rccl.ncclWindow_vidmem
   rocm.bindings.rccl.ncclUniqueId
   rocm.bindings.rccl.ncclResult_t
   rocm.bindings.rccl.ncclConfig_v22800
   rocm.bindings.rccl.ncclSimInfo_v22200
   rocm.bindings.rccl.ncclCommMemStat_t
   rocm.bindings.rccl.ncclRedOp_dummy_t
   rocm.bindings.rccl.ncclRedOp_t
   rocm.bindings.rccl.ncclDataType_t
   rocm.bindings.rccl.ncclScalarResidence_t
   rocm.bindings.rccl.ncclWaitSignalDesc_t
   rocm.bindings.rccl.ncclParamHandle


Functions
---------

.. autoapisummary::

   rocm.bindings.rccl.has_symbol
   rocm.bindings.rccl.ncclMemAlloc
   rocm.bindings.rccl.ncclMemFree
   rocm.bindings.rccl.ncclGetVersion
   rocm.bindings.rccl.ncclGetUniqueId
   rocm.bindings.rccl.ncclCommInitRankConfig
   rocm.bindings.rccl.ncclCommInitRank
   rocm.bindings.rccl.ncclCommInitAll
   rocm.bindings.rccl.ncclCommFinalize
   rocm.bindings.rccl.ncclCommDestroy
   rocm.bindings.rccl.ncclCommAbort
   rocm.bindings.rccl.ncclCommRevoke
   rocm.bindings.rccl.ncclCommSplit
   rocm.bindings.rccl.ncclCommShrink
   rocm.bindings.rccl.ncclCommGetUniqueId
   rocm.bindings.rccl.ncclCommGrow
   rocm.bindings.rccl.ncclCommInitRankScalable
   rocm.bindings.rccl.ncclGetErrorString
   rocm.bindings.rccl.ncclGetLastError
   rocm.bindings.rccl.ncclResetDebugInit
   rocm.bindings.rccl.ncclCommGetAsyncError
   rocm.bindings.rccl.ncclCommCount
   rocm.bindings.rccl.ncclCommCuDevice
   rocm.bindings.rccl.ncclCommUserRank
   rocm.bindings.rccl.ncclCommRegister
   rocm.bindings.rccl.ncclCommDeregister
   rocm.bindings.rccl.ncclCommSuspend
   rocm.bindings.rccl.ncclCommResume
   rocm.bindings.rccl.ncclCommMemStats
   rocm.bindings.rccl.ncclCommWindowRegister
   rocm.bindings.rccl.ncclCommWindowDeregister
   rocm.bindings.rccl.ncclWinGetUserPtr
   rocm.bindings.rccl.ncclRedOpCreatePreMulSum
   rocm.bindings.rccl.ncclRedOpDestroy
   rocm.bindings.rccl.ncclReduce
   rocm.bindings.rccl.ncclBcast
   rocm.bindings.rccl.ncclBroadcast
   rocm.bindings.rccl.ncclAllReduce
   rocm.bindings.rccl.ncclAllReduceWithBias
   rocm.bindings.rccl.ncclReduceScatter
   rocm.bindings.rccl.ncclAllGather
   rocm.bindings.rccl.ncclAlltoAll
   rocm.bindings.rccl.ncclAlltoAllv
   rocm.bindings.rccl.ncclGather
   rocm.bindings.rccl.ncclScatter
   rocm.bindings.rccl.ncclAllToAll
   rocm.bindings.rccl.ncclAllToAllv
   rocm.bindings.rccl.ncclSend
   rocm.bindings.rccl.ncclRecv
   rocm.bindings.rccl.ncclPutSignal
   rocm.bindings.rccl.ncclSignal
   rocm.bindings.rccl.ncclWaitSignal
   rocm.bindings.rccl.ncclGroupStart
   rocm.bindings.rccl.ncclGroupEnd
   rocm.bindings.rccl.ncclGroupSimulateEnd
   rocm.bindings.rccl.ncclParamBind
   rocm.bindings.rccl.ncclParamGetI8
   rocm.bindings.rccl.ncclParamGetI16
   rocm.bindings.rccl.ncclParamGetI32
   rocm.bindings.rccl.ncclParamGetI64
   rocm.bindings.rccl.ncclParamGetU8
   rocm.bindings.rccl.ncclParamGetU16
   rocm.bindings.rccl.ncclParamGetU32
   rocm.bindings.rccl.ncclParamGetU64
   rocm.bindings.rccl.ncclParamGetStr
   rocm.bindings.rccl.ncclParamGet
   rocm.bindings.rccl.ncclParamGetParameter
   rocm.bindings.rccl.ncclParamGetAllParameterKeys
   rocm.bindings.rccl.ncclParamDumpAll


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:data:: NCCL_MAJOR
   :type:  Any

.. py:data:: NCCL_MINOR
   :type:  Any

.. py:data:: NCCL_PATCH
   :type:  Any

.. py:data:: NCCL_SUFFIX
   :type:  Any

.. py:data:: NCCL_VERSION_CODE
   :type:  Any

.. py:data:: RCCL_BFLOAT16
   :type:  Any

.. py:data:: RCCL_GATHER_SCATTER
   :type:  Any

.. py:data:: RCCL_ALLTOALLV
   :type:  Any

.. py:data:: NCCL_UNIQUE_ID_BYTES
   :type:  Any

.. py:class:: ncclComm(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: ncclComm_t

.. py:class:: ncclWindow_vidmem(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: ncclWindow_t

.. py:class:: ncclUniqueId(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Opaque unique id used to initialize communicators

   The ncclUniqueId must be passed to all participating ranks


   .. py:attribute:: internal
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: ncclResult_t

   Bases: :py:obj:`enum.IntEnum`


   Result type

   Return codes aside from ncclSuccess indicate that a call has failed


   .. py:attribute:: ncclSuccess
      :type:  int


   .. py:attribute:: ncclUnhandledCudaError
      :type:  int


   .. py:attribute:: ncclSystemError
      :type:  int


   .. py:attribute:: ncclInternalError
      :type:  int


   .. py:attribute:: ncclInvalidArgument
      :type:  int


   .. py:attribute:: ncclInvalidUsage
      :type:  int


   .. py:attribute:: ncclRemoteError
      :type:  int


   .. py:attribute:: ncclInProgress
      :type:  int


   .. py:attribute:: ncclTimeout
      :type:  int


   .. py:attribute:: ncclNumResults
      :type:  int


.. py:class:: ncclConfig_v22800(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Communicator configuration

   Users can assign value to attributes to specify the behavior of a communicator


   .. py:attribute:: size
      :type:  Any


   .. py:attribute:: magic
      :type:  Any


   .. py:attribute:: version
      :type:  Any


   .. py:attribute:: blocking
      :type:  Any


   .. py:attribute:: cgaClusterSize
      :type:  Any


   .. py:attribute:: minCTAs
      :type:  Any


   .. py:attribute:: maxCTAs
      :type:  Any


   .. py:attribute:: netName
      :type:  Any


   .. py:attribute:: splitShare
      :type:  Any


   .. py:attribute:: trafficClass
      :type:  Any


   .. py:attribute:: commName
      :type:  Any


   .. py:attribute:: collnetEnable
      :type:  Any


   .. py:attribute:: CTAPolicy
      :type:  Any


   .. py:attribute:: shrinkShare
      :type:  Any


   .. py:attribute:: nvlsCTAs
      :type:  Any


   .. py:attribute:: nChannelsPerNetPeer
      :type:  Any


   .. py:attribute:: nvlinkCentricSched
      :type:  Any


   .. py:attribute:: graphUsageMode
      :type:  Any


   .. py:attribute:: numRmaCtx
      :type:  Any


   .. py:attribute:: maxP2pPeers
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: ncclConfig_t

.. py:class:: ncclSimInfo_v22200(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: size
      :type:  Any


   .. py:attribute:: magic
      :type:  Any


   .. py:attribute:: version
      :type:  Any


   .. py:attribute:: estimatedTime
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: ncclSimInfo_t

.. py:function:: ncclMemAlloc(size)

   NCCL malloc and free function for all types of NCCL optimizations
   (e.g.

   user buffer registration). The actual allocated size might
   be larger than requested due to granularity requirement.

   Args:
       size (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: (undocumented)
       * ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclMemAlloc(void ** ptr, size_t size)


.. py:function:: ncclMemFree(ptr)

   (No short description, might be part of a group.)

   Args:
       ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclMemFree(void * ptr)


.. py:function:: ncclGetVersion()

   Return the RCCL_VERSION_CODE of RCCL in the supplied integer.

   This integer is coded with the MAJOR, MINOR and PATCH level of RCCL.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.
       * :py:obj:`~.int`:
               Pointer to where version will be stored

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclGetVersion(int * version)


.. py:function:: ncclGetUniqueId()

   Generates an ID for ncclCommInitRank.

   Generates an ID to be used in ncclCommInitRank.
   ncclGetUniqueId should be called once by a single rank and the
   ID should be distributed to all ranks in the communicator before
   using it as a parameter for ncclCommInitRank.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.
       * :py:obj:`~.ncclUniqueId`:
               Pointer to where uniqueId will be stored

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclGetUniqueId(ncclUniqueId * uniqueId)


.. py:function:: ncclCommInitRankConfig(nranks, commId, rank, config)

   Create a new communicator with config.

   Create a new communicator (multi thread/process version) with a configuration
   set by users. See ``rccl_config_type`` for more details.
   Each rank is associated to a CUDA device, which has to be set before calling
   ncclCommInitRank.

   Args:
       nranks (:py:obj:`~.int`) -- *IN*:
           Total number of ranks participating in this communicator

       commId (:py:obj:`~.ncclUniqueId`) -- *IN*:
           UniqueId required for initialization

       rank (:py:obj:`~.int`) -- *IN*:
           Current rank to create communicator for. [0 to nranks-1]

       config (:py:obj:`~.ncclConfig_v22800`/:py:obj:`~.object`) -- *IN*:
           Pointer to communicator configuration

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.
       * :py:obj:`~.ncclComm`:
               Pointer to created communicator

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclCommInitRankConfig(ncclComm_t * comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t * config)


.. py:function:: ncclCommInitRank(nranks, commId, rank)

   Creates a new communicator (multi thread/process version).

   Rank must be between 0 and nranks-1 and unique within a communicator clique.
   Each rank is associated to a CUDA device, which has to be set before calling
   ncclCommInitRank.  ncclCommInitRank implicitly syncronizes with other ranks,
   so it must be called by different threads/processes or use ncclGroupStart/ncclGroupEnd.

   Args:
       nranks (:py:obj:`~.int`) -- *IN*:
           Total number of ranks participating in this communicator

       commId (:py:obj:`~.ncclUniqueId`) -- *IN*:
           UniqueId required for initialization

       rank (:py:obj:`~.int`) -- *IN*:
           Current rank to create communicator for

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.
       * :py:obj:`~.ncclComm`:
               Pointer to created communicator

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclCommInitRank(ncclComm_t * comm, int nranks, ncclUniqueId commId, int rank)


.. py:function:: ncclCommInitAll(comm, ndev, devlist)

   Creates a clique of communicators (single process version).

   This is a convenience function to create a single-process communicator clique.
   Returns an array of ndev newly initialized communicators in comm.
   comm should be pre-allocated with size at least ndev*sizeof(ncclComm_t).
   If devlist is NULL, the first ndev HIP devices are used.
   Order of devlist defines user-order of processors within the communicator.

   Args:
       comm (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Pointer to array of created communicators

       ndev (:py:obj:`~.int`) -- *IN*:
           Total number of ranks participating in this communicator

       devlist (:py:obj:`~.rocm.bindings.util.types.ListOfInt`/:py:obj:`~.object`) -- *IN*:
           Array of GPU device indices to create for

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclCommInitAll(ncclComm_t * comm, int ndev, const int * devlist)


.. py:function:: ncclCommFinalize(comm)

   Finalize a communicator.

   ncclCommFinalize flushes all issued communications
   and marks communicator state as ncclInProgress. The state will change to ncclSuccess
   when the communicator is globally quiescent and related resources are freed; then,
   calling ncclCommDestroy can locally free the rest of the resources (e.g. communicator
   itself) without blocking.

   Args:
       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator to finalize

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclCommFinalize(ncclComm_t comm)


.. py:function:: ncclCommDestroy(comm)

   Frees local resources associated with communicator object.

   Destroy all local resources associated with the passed in communicator object

   Args:
       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator to destroy

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclCommDestroy(ncclComm_t comm)


.. py:function:: ncclCommAbort(comm)

   Abort any in-progress calls and destroy the communicator object.

   Frees resources associated with communicator object and aborts any operations
   that might still be running on the device.

   Args:
       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator to abort and destroy

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclCommAbort(ncclComm_t comm)


.. py:function:: ncclCommRevoke(comm, revokeFlags)

   Revoke a communicator without destroying it.

   Aborts in-flight collectives by raising the comm's abort flag,
   stops the proxy service, and rejects subsequently enqueued
   collectives with ncclInvalidUsage. The abort flag is cleared
   once the asynchronous revoke job completes, so the communicator
   remains valid as a parent for ncclCommSplit / ncclCommShrink /
   ncclCommGrow, and may be torn down via ncclCommDestroy /
   ncclCommAbort.
   Because in-flight collectives are aborted (not drained), revoke
   can recover from peers that have failed mid-collective; output
   buffers of an aborted collective contain undefined data.
   ncclCommFinalize on a revoked communicator is invalid and returns
   ncclInvalidUsage.
   Pass NCCL_REVOKE_DEFAULT for revokeFlags; other values are rejected
   with ncclInvalidArgument.

   Args:
       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator to revoke

       revokeFlags (:py:obj:`~.int`) -- *IN*:
           Reserved; must be NCCL_REVOKE_DEFAULT

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclCommRevoke(ncclComm_t comm, int revokeFlags)


.. py:function:: ncclCommSplit(comm, color, key, config)

   Create one or more communicators from an existing one.

   Creates one or more communicators from an existing one.
   Ranks with the same color will end up in the same communicator.
   Within the new communicator, key will be used to order ranks.
   NCCL_SPLIT_NOCOLOR as color will indicate the rank will not be part of any group
   and will therefore return a NULL communicator.
   If config is NULL, the new communicator will inherit the original communicator's configuration

   Args:
       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Original communicator object for this rank

       color (:py:obj:`~.int`) -- *IN*:
           Color to assign this rank

       key (:py:obj:`~.int`) -- *IN*:
           Key used to order ranks within the same new communicator

       config (:py:obj:`~.ncclConfig_v22800`/:py:obj:`~.object`) -- *IN*:
           Config file for new communicator. May be NULL to inherit from comm

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.
       * :py:obj:`~.ncclComm`:
               Pointer to new communicator

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t * newcomm, ncclConfig_t * config)


.. py:function:: ncclCommShrink(comm, excludeRanksCount, config, shrinkFlags)

   Shrink existing communicator.

   Ranks in excludeRanksList will be removed form the existing communicator.
   Within the new communicator, ranks will be re-ordered to fill the gap of removed ones.
   If config is NULL, the new communicator will inherit the original communicator's configuration.
   The flag enables NCCL to adapt to various states of the parent communicator, see NCCL_SHRINK flags.

   Args:
       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Original communicator object for this rank

       excludeRanksCount (:py:obj:`~.int`) -- *IN*:
           Number of ranks to be excluded

       config (:py:obj:`~.ncclConfig_v22800`/:py:obj:`~.object`) -- *IN*:
           Config file for new communicator. May be NULL to inherit from comm

       shrinkFlags (:py:obj:`~.int`) -- *IN*:
           Flag to adapt to various states of the parent communicator (see NCCL_SHRINK flags)

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.
       * :py:obj:`~.int`:
               List of ranks to be exluded
       * :py:obj:`~.ncclComm`:
               Pointer to new communicator

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclCommShrink(ncclComm_t comm, int * excludeRanksList, int excludeRanksCount, ncclComm_t * newcomm, ncclConfig_t * config, int shrinkFlags)


.. py:function:: ncclCommGetUniqueId(comm)

   Generate a per-communicator unique ID for growing a communicator.

   Generates a unique ID on an existing communicator. The ID must be
   distributed to the ranks joining through ncclCommGrow.
   Constraints:
   - A new ID cannot be generated while a previous ID is unconsumed.
   - Each ID can only be used once (no reuse after consumption).
   - The grow operation must complete before calling this again.

   Args:
       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Existing communicator that coordinates the grow

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.
       * :py:obj:`~.ncclUniqueId`:
               Pointer to the generated unique ID

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclCommGetUniqueId(ncclComm_t comm, ncclUniqueId * uniqueId)


.. py:function:: ncclCommGrow(comm, nRanks, uniqueId, rank, config)

   Grow a communicator by adding new ranks.

   Creates a larger communicator from an existing one plus newly
   joining ranks. The unique ID obtained from ncclCommGetUniqueId
   must be distributed to the new ranks. Parameter usage:
   - Existing non-root ranks: comm set, uniqueId = NULL, rank = -1
   - Existing root rank: comm set, uniqueId = &id, rank = -1
   - New ranks: comm = NULL, uniqueId = &id, rank = assigned
   The unique ID is consumed upon a successful grow and cannot be reused.
   If config is NULL, the new communicator inherits the original
   communicator's configuration.

   Args:
       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Existing communicator, or NULL for newly joining ranks

       nRanks (:py:obj:`~.int`) -- *IN*:
           Total number of ranks in the new communicator

       uniqueId (:py:obj:`~.ncclUniqueId`/:py:obj:`~.object`) -- *IN*:
           Unique ID from ncclCommGetUniqueId; NULL on existing non-root ranks

       rank (:py:obj:`~.int`) -- *IN*:
           Rank in the new communicator for joining ranks; -1 for existing ranks

       config (:py:obj:`~.ncclConfig_v22800`/:py:obj:`~.object`) -- *IN*:
           Config for the new communicator. May be NULL to inherit from comm

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.
       * :py:obj:`~.ncclComm`:
               Pointer to the new communicator

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclCommGrow(ncclComm_t comm, int nRanks, const ncclUniqueId * uniqueId, int rank, ncclComm_t * newcomm, ncclConfig_t * config)


.. py:function:: ncclCommInitRankScalable(nranks, myrank, nId, commIds, config)

   Creates a new communicator (multi thread/process version), similar to ncclCommInitRankConfig.

   Allows to use more than one ncclUniqueId (up to one per rank),
   indicated by nId, to accelerate the init operation.
   The number of ncclUniqueIds and their order must be the same for every rank.

   Args:
       nranks (:py:obj:`~.int`) -- *IN*:
           Total number of ranks participating in this communicator

       myrank (:py:obj:`~.int`) -- *IN*:
           Current rank

       nId (:py:obj:`~.int`) -- *IN*:
           Number of unique IDs

       commIds (:py:obj:`~.ncclUniqueId`/:py:obj:`~.object`) -- *IN*:
           List of unique IDs

       config (:py:obj:`~.ncclConfig_v22800`/:py:obj:`~.object`) -- *IN*:
           Config file for new communicator. May be NULL to inherit from comm

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.
       * :py:obj:`~.ncclComm`:
               Pointer to new communicator

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclCommInitRankScalable(ncclComm_t * newcomm, int nranks, int myrank, int nId, ncclUniqueId * commIds, ncclConfig_t * config)


.. py:function:: ncclGetErrorString(result)

   Returns a string for each result code.

   Returns a human-readable string describing the given result code.

   Args:
       result (:py:obj:`~.ncclResult_t`) -- *IN*:
           Result code to get description for

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`:
               Always returns `~.ncclResult_t.ncclSuccess`.
       * :py:obj:`~.bytes`: String containing description of result code.

   .. rubric:: C signature

   .. code-block:: c

       const char * ncclGetErrorString(ncclResult_t result)


.. py:function:: ncclGetLastError(comm)

   Returns a human-readable message of the last error that occurred.

   Args:
       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`:
               Always returns `~.ncclResult_t.ncclSuccess`.
       * :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * ncclGetLastError(ncclComm_t comm)


.. py:function:: ncclResetDebugInit()

   Reload environment variables that determine logging.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`:
               Always returns `~.ncclResult_t.ncclSuccess`.

   .. rubric:: C signature

   .. code-block:: c

       void ncclResetDebugInit()


.. py:function:: ncclCommGetAsyncError(comm, asyncError)

   Checks whether the comm has encountered any asynchronous errors

   Query whether the provided communicator has encountered any asynchronous errors

   Args:
       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator to query

       asyncError (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Pointer to where result code will be stored

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t * asyncError)


.. py:function:: ncclCommCount(comm)

   Gets the number of ranks in the communicator clique.

   Returns the number of ranks in the communicator clique (as set during initialization)

   Args:
       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.
       * :py:obj:`~.int`:
               Pointer to where number of ranks will be stored

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclCommCount(const ncclComm_t comm, int * count)


.. py:function:: ncclCommCuDevice(comm)

   Get the ROCm device index associated with a communicator

   Returns the ROCm device number associated with the provided communicator.

   Args:
       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.
       * :py:obj:`~.int`:
               Pointer to where the associated ROCm device index will be stored

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int * device)


.. py:function:: ncclCommUserRank(comm)

   Get the rank associated with a communicator

   Returns the user-ordered "rank" associated with the provided communicator.

   Args:
       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.
       * :py:obj:`~.int`:
               Pointer to where the associated rank will be stored

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclCommUserRank(const ncclComm_t comm, int * rank)


.. py:function:: ncclCommRegister(comm, buff, size)

   Register CUDA buffer for zero-copy operation

   Args:
       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`):
           (undocumented)

       buff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       size (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: (undocumented)
       * handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclCommRegister(const ncclComm_t comm, void * buff, size_t size, void ** handle)


.. py:function:: ncclCommDeregister(comm, handle)

   Deregister CUDA buffer

   Args:
       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`):
           (undocumented)

       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclCommDeregister(const ncclComm_t comm, void * handle)


.. py:function:: ncclCommSuspend(comm, flags)

   Suspend communicator operations to free resources.

   Releases the resources selected by ``flags.`` The communicator
   cannot be used until ``ncclCommResume`` is called.

   Args:
       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator to suspend

       flags (:py:obj:`~.int`) -- *IN*:
           Bitmask of NCCL_SUSPEND_* flags (e.g. ``NCCL_SUSPEND_MEM`` )

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclCommSuspend(ncclComm_t comm, int flags)


.. py:function:: ncclCommResume(comm)

   Resume a previously suspended communicator.

   Reacquires every resource that was released by the matching
   ``ncclCommSuspend`` call.

   Args:
       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator to resume

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclCommResume(ncclComm_t comm)


.. py:class:: ncclCommMemStat_t

   Bases: :py:obj:`enum.IntEnum`


   Communicator memory statistic selector

   Identifier passed to ``ncclCommMemStats`` to choose which
   memory counter to read.


   .. py:attribute:: ncclStatGpuMemSuspend
      :type:  int


   .. py:attribute:: ncclStatGpuMemSuspended
      :type:  int


   .. py:attribute:: ncclStatGpuMemPersist
      :type:  int


   .. py:attribute:: ncclStatGpuMemTotal
      :type:  int


.. py:function:: ncclCommMemStats(comm, stat)

   Query communicator memory statistics.

   Args:
       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator to query

       stat (:py:obj:`~.ncclCommMemStat_t`) -- *IN*:
           One of ``ncclCommMemStat_t`` values

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.
       * :py:obj:`~.int`:
               Pointer to receive the memory statistic value

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclCommMemStats(ncclComm_t comm, ncclCommMemStat_t stat, uint64_t * value)


.. py:function:: ncclCommWindowRegister(comm, buff, size, winFlags)

   Register memory window

   Args:
       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`):
           (undocumented)

       buff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       size (:py:obj:`~.int`):
           (undocumented)

       winFlags (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: (undocumented)
       * win (:py:obj:`~.ncclWindow_vidmem`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void * buff, size_t size, ncclWindow_t * win, int winFlags)


.. py:function:: ncclCommWindowDeregister(comm, win)

   Deregister symmetric memory

   Args:
       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`):
           (undocumented)

       win (:py:obj:`~.ncclWindow_vidmem`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win)


.. py:function:: ncclWinGetUserPtr(comm, win)

   Get the user pointer from the window

   Args:
       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`):
           (undocumented)

       win (:py:obj:`~.ncclWindow_vidmem`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: (undocumented)
       * outUserPtr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclWinGetUserPtr(ncclComm_t comm, ncclWindow_t win, void ** outUserPtr)


.. py:class:: ncclRedOp_dummy_t

   Bases: :py:obj:`enum.IntEnum`


   Dummy reduction enumeration

   Dummy reduction enumeration used to determine value for ncclMaxRedOp


   .. py:attribute:: ncclNumOps_dummy
      :type:  int


.. py:class:: ncclRedOp_t

   Bases: :py:obj:`enum.IntEnum`


   Reduction operation selector

   Enumeration used to specify the various reduction operations
   ncclNumOps is the number of built-in ncclRedOp_t values and serves as
   the least possible value for dynamic ncclRedOp_t values constructed by
   ncclRedOpCreate functions.

   ncclMaxRedOp is the largest valid value for ncclRedOp_t and is defined
   to be the largest signed value (since compilers are permitted to use
   signed enums) that won't grow sizeof(ncclRedOp_t) when compared to previous
   RCCL versions to maintain ABI compatibility.


   .. py:attribute:: ncclSum
      :type:  int


   .. py:attribute:: ncclProd
      :type:  int


   .. py:attribute:: ncclMax
      :type:  int


   .. py:attribute:: ncclMin
      :type:  int


   .. py:attribute:: ncclAvg
      :type:  int


   .. py:attribute:: ncclNumOps
      :type:  int


   .. py:attribute:: ncclMaxRedOp
      :type:  int


.. py:class:: ncclDataType_t

   Bases: :py:obj:`enum.IntEnum`


   Data types

   Enumeration of the various supported datatype


   .. py:attribute:: ncclInt8
      :type:  int


   .. py:attribute:: ncclChar
      :type:  int


   .. py:attribute:: ncclUint8
      :type:  int


   .. py:attribute:: ncclInt32
      :type:  int


   .. py:attribute:: ncclInt
      :type:  int


   .. py:attribute:: ncclUint32
      :type:  int


   .. py:attribute:: ncclInt64
      :type:  int


   .. py:attribute:: ncclUint64
      :type:  int


   .. py:attribute:: ncclFloat16
      :type:  int


   .. py:attribute:: ncclHalf
      :type:  int


   .. py:attribute:: ncclFloat32
      :type:  int


   .. py:attribute:: ncclFloat
      :type:  int


   .. py:attribute:: ncclFloat64
      :type:  int


   .. py:attribute:: ncclDouble
      :type:  int


   .. py:attribute:: ncclBfloat16
      :type:  int


   .. py:attribute:: ncclFloat8e4m3
      :type:  int


   .. py:attribute:: ncclFloat8e5m2
      :type:  int


   .. py:attribute:: ncclNumTypes
      :type:  int


.. py:class:: ncclScalarResidence_t

   Bases: :py:obj:`enum.IntEnum`


   Location and dereferencing logic for scalar arguments.

   Enumeration specifying memory location of the scalar argument.
   Based on where the value is stored, the argument will be dereferenced either
   while the collective is running (if in device memory), or before the ncclRedOpCreate()
   function returns (if in host memory).


   .. py:attribute:: ncclScalarDevice
      :type:  int


   .. py:attribute:: ncclScalarHostImmediate
      :type:  int


.. py:function:: ncclRedOpCreatePreMulSum(op, scalar, datatype, residence, comm)

   Create a custom pre-multiplier reduction operator

   Creates a new reduction operator which pre-multiplies input values by a given
   scalar locally before reducing them with peer values via summation. For use
   only with collectives launched against *comm* and *datatype*. The
    residence* argument indicates how/when the memory pointed to by *scalar*
   will be dereferenced. Upon return, the newly created operator's handle
   is stored in *op*.

   Args:
       op (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Pointer to where newly created custom reduction operator is to be stored

       scalar (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to scalar value.

       datatype (:py:obj:`~.ncclDataType_t`) -- *IN*:
           Scalar value datatype

       residence (:py:obj:`~.ncclScalarResidence_t`) -- *IN*:
           Memory type of the scalar value

       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator to associate with this custom reduction operator

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t * op, void * scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm)


.. py:function:: ncclRedOpDestroy(op, comm)

   Destroy custom reduction operator

   Destroys the reduction operator *op*. The operator must have been created by
   ncclRedOpCreatePreMul with the matching communicator *comm*. An operator may be
   destroyed as soon as the last RCCL function which is given that operator returns.

   Args:
       op (:py:obj:`~.ncclRedOp_t`) -- *IN*:
           Custom reduction operator is to be destroyed

       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator associated with this reduction operator

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm)


.. py:function:: ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream)

   Reduce

   Reduces data arrays of length *count* in *sendbuff* into *recvbuff* using *op*
   operation.
    recvbuff* may be NULL on all calls except for root device.
    root* is the rank (not the HIP device) where data will reside after the
    operation is complete.
   In-place operation will happen if sendbuff == recvbuff.

   Args:
       sendbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Local device data buffer to be reduced

       recvbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           Data buffer where result is stored (only for *root* rank).  May be null for other ranks.

       count (:py:obj:`~.int`) -- *IN*:
           Number of elements in every send buffer

       datatype (:py:obj:`~.ncclDataType_t`) -- *IN*:
           Data buffer element datatype

       op (:py:obj:`~.ncclRedOp_t`) -- *IN*:
           Reduction operator type

       root (:py:obj:`~.int`) -- *IN*:
           Rank where result data array will be stored

       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator group object to execute on

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           HIP stream to execute collective on

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclReduce(const void * sendbuff, void * recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, hipStream_t stream)


.. py:function:: ncclBcast(buff, count, datatype, root, comm, stream)

   (Deprecated) Broadcast (in-place)

   Copies *count* values from *root* to all other devices.
   root is the rank (not the CUDA device) where data resides before the
   operation is started.
   This operation is implicitly in-place.

   Args:
       buff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           Input array on *root* to be copied to other ranks.  Output array for all ranks.

       count (:py:obj:`~.int`) -- *IN*:
           Number of elements in data buffer

       datatype (:py:obj:`~.ncclDataType_t`) -- *IN*:
           Data buffer element datatype

       root (:py:obj:`~.int`) -- *IN*:
           Rank owning buffer to be copied to others

       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator group object to execute on

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           HIP stream to execute collective on

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclBcast(void * buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream)


.. py:function:: ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream)

   Broadcast

   Copies *count* values from *sendbuff* on *root* to *recvbuff* on all devices.
    root* is the rank (not the HIP device) where data resides before the operation is started.
    sendbuff* may be NULL on ranks other than *root*.
   In-place operation will happen if *sendbuff* == *recvbuff*.

   Args:
       sendbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Data array to copy (if *root*).  May be NULL for other ranks

       recvbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           Data array to store received array

       count (:py:obj:`~.int`) -- *IN*:
           Number of elements in data buffer

       datatype (:py:obj:`~.ncclDataType_t`) -- *IN*:
           Data buffer element datatype

       root (:py:obj:`~.int`) -- *IN*:
           Rank of broadcast root

       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator group object to execute on

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           HIP stream to execute collective on

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclBroadcast(const void * sendbuff, void * recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream)


.. py:function:: ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream)

   All-Reduce

   Reduces data arrays of length *count* in *sendbuff* using *op* operation, and
   leaves identical copies of result on each *recvbuff*.
   In-place operation will happen if sendbuff == recvbuff.

   Args:
       sendbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Input data array to reduce

       recvbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           Data array to store reduced result array

       count (:py:obj:`~.int`) -- *IN*:
           Number of elements in data buffer

       datatype (:py:obj:`~.ncclDataType_t`) -- *IN*:
           Data buffer element datatype

       op (:py:obj:`~.ncclRedOp_t`) -- *IN*:
           Reduction operator

       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator group object to execute on

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           HIP stream to execute collective on

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclAllReduce(const void * sendbuff, void * recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, hipStream_t stream)


.. py:function:: ncclAllReduceWithBias(sendbuff, recvbuff, count, datatype, op, comm, stream, acc)

   All-Reduce-with-Bias

   Reduces data arrays of length *count* in *sendbuff* using *op* operation, and
   leaves identical copies of result on each *recvbuff*.
   In-place operation will happen if sendbuff == recvbuff.

   Args:
       sendbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Input data array to reduce

       recvbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           Data array to store reduced result array

       count (:py:obj:`~.int`) -- *IN*:
           Number of elements in data buffer

       datatype (:py:obj:`~.ncclDataType_t`) -- *IN*:
           Data buffer element datatype

       op (:py:obj:`~.ncclRedOp_t`) -- *IN*:
           Reduction operator

       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator group object to execute on

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           HIP stream to execute collective on

       acc (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Bias data array to reduce

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclAllReduceWithBias(const void * sendbuff, void * recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, hipStream_t stream, const void * acc)


.. py:function:: ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream)

   Reduce-Scatter

   Reduces data in *sendbuff* using *op* operation and leaves reduced result
   scattered over the devices so that *recvbuff* on rank i will contain the i-th
   block of the result.
   Assumes sendcount is equal to nranks*recvcount, which means that *sendbuff*
   should have a size of at least nranks*recvcount elements.
   In-place operations will happen if recvbuff == sendbuff + rank * recvcount.

   Args:
       sendbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Input data array to reduce

       recvbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           Data array to store reduced result subarray

       recvcount (:py:obj:`~.int`) -- *IN*:
           Number of elements each rank receives

       datatype (:py:obj:`~.ncclDataType_t`) -- *IN*:
           Data buffer element datatype

       op (:py:obj:`~.ncclRedOp_t`) -- *IN*:
           Reduction operator

       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator group object to execute on

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           HIP stream to execute collective on

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclReduceScatter(const void * sendbuff, void * recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, hipStream_t stream)


.. py:function:: ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream)

   All-Gather

   Each device gathers *sendcount* values from other GPUs into *recvbuff*,
   receiving data from rank i at offset i*sendcount.
   Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
   should have a size of at least nranks*sendcount elements.
   In-place operations will happen if sendbuff == recvbuff + rank * sendcount.

   Args:
       sendbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Input data array to send

       recvbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           Data array to store the gathered result

       sendcount (:py:obj:`~.int`) -- *IN*:
           Number of elements each rank sends

       datatype (:py:obj:`~.ncclDataType_t`) -- *IN*:
           Data buffer element datatype

       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator group object to execute on

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           HIP stream to execute collective on

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclAllGather(const void * sendbuff, void * recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream)


.. py:function:: ncclAlltoAll(sendbuff, recvbuff, count, datatype, comm, stream)

   All-to-All

   Each device sends count values to all other devices and receives count values
   from all other devices. Data to send to destination rank j is taken from
   sendbuff+j*count and data received from source rank i is placed at
   recvbuff+i*count.

   Args:
       sendbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Data array to send (contains blocks for each other rank)

       recvbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           Data array to receive (contains blocks from each other rank)

       count (:py:obj:`~.int`) -- *IN*:
           Number of elements to send between each pair of ranks

       datatype (:py:obj:`~.ncclDataType_t`) -- *IN*:
           Data buffer element datatype

       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator group object to execute on

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           HIP stream to execute collective on

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclAlltoAll(const void * sendbuff, void * recvbuff, size_t count, ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream)


.. py:function:: ncclAlltoAllv(sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls, datatype, comm, stream)

   All-To-Allv

   Device (i) sends sendcounts[j] of data from offset sdispls[j]
   to device (j). At the same time, device (i) receives recvcounts[j] of data
   from device (j) to be placed at rdispls[j].
   sendcounts, sdispls, recvcounts and rdispls are all measured in the units
   of datatype, not bytes.
   In-place operation will happen if sendbuff == recvbuff.

   Args:
       sendbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Data array to send (contains blocks for each other rank)

       sendcounts (:py:obj:`~.rocm.bindings.util.types.ListOfUInt64`/:py:obj:`~.object`) -- *IN*:
           Array containing number of elements to send to each participating rank

       sdispls (:py:obj:`~.rocm.bindings.util.types.ListOfUInt64`/:py:obj:`~.object`) -- *IN*:
           Array of offsets into *sendbuff* for each participating rank

       recvbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           Data array to receive (contains blocks from each other rank)

       recvcounts (:py:obj:`~.rocm.bindings.util.types.ListOfUInt64`/:py:obj:`~.object`) -- *IN*:
           Array containing number of elements to receive from each participating rank

       rdispls (:py:obj:`~.rocm.bindings.util.types.ListOfUInt64`/:py:obj:`~.object`) -- *IN*:
           Array of offsets into *recvbuff* for each participating rank

       datatype (:py:obj:`~.ncclDataType_t`) -- *IN*:
           Data buffer element datatype

       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator group object to execute on

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           HIP stream to execute collective on

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclAlltoAllv(const void * sendbuff, const size_t[] sendcounts, const size_t[] sdispls, void * recvbuff, const size_t[] recvcounts, const size_t[] rdispls, ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream)


.. py:function:: ncclGather(sendbuff, recvbuff, count, datatype, root, comm, stream)

   Gather

   Each rank sends count elements from sendbuff to the root rank.
   On the root rank, data from rank i is placed at recvbuff + i*count.
   On non-root ranks, recvbuff is not used.
   root is the rank where data will be gathered.

   In-place operations will happen if sendbuff == recvbuff + root * count.

   Args:
       sendbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Data array to send

       recvbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           Data array to recv

       count (:py:obj:`~.int`) -- *IN*:
           Number of elements

       datatype (:py:obj:`~.ncclDataType_t`) -- *IN*:
           Data buffer element datatype

       root (:py:obj:`~.int`) -- *IN*:
           Rank of gather root

       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator group object to execute on

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           HIP stream to execute collective on

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclGather(const void * sendbuff, void * recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream)


.. py:function:: ncclScatter(sendbuff, recvbuff, count, datatype, root, comm, stream)

   Scatter

   On the root rank, count elements from sendbuff+i*count are sent to rank i.
   On non-root ranks, sendbuff is not used.
   Each rank receives count elements into recvbuff.
   root is the rank that will distribute the data.

   In-place operations will happen if recvbuff == sendbuff + root * count.

   Args:
       sendbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Data array to send

       recvbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           Data array to recv

       count (:py:obj:`~.int`) -- *IN*:
           Number of elements

       datatype (:py:obj:`~.ncclDataType_t`) -- *IN*:
           Data buffer element datatype

       root (:py:obj:`~.int`) -- *IN*:
           Rank of scatter root

       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator group object to execute on

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           HIP stream to execute collective on

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclScatter(const void * sendbuff, void * recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream)


.. py:function:: ncclAllToAll(sendbuff, recvbuff, count, datatype, comm, stream)

   All-To-All

   Device (i) send (j)th block of data to device (j) and be placed as (i)th
   block. Each block for sending/receiving has *count* elements, which means
   that *recvbuff* and *sendbuff* should have a size of nranks*count elements.
   In-place operation is NOT supported. It is the user's responsibility
   to ensure that sendbuff and recvbuff are distinct.

   Deprecated:
       ncclAllToAll is replaced with ncclAlltoAll and will be removed in the future.

   Args:
       sendbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Data array to send (contains blocks for each other rank)

       recvbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           Data array to receive (contains blocks from each other rank)

       count (:py:obj:`~.int`) -- *IN*:
           Number of elements to send between each pair of ranks

       datatype (:py:obj:`~.ncclDataType_t`) -- *IN*:
           Data buffer element datatype

       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator group object to execute on

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           HIP stream to execute collective on

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclAllToAll(const void * sendbuff, void * recvbuff, size_t count, ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream)


.. py:function:: ncclAllToAllv(sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls, datatype, comm, stream)

   All-To-Allv

   Device (i) sends sendcounts[j] of data from offset sdispls[j]
   to device (j). At the same time, device (i) receives recvcounts[j] of data
   from device (j) to be placed at rdispls[j].
   sendcounts, sdispls, recvcounts and rdispls are all measured in the units
   of datatype, not bytes.
   In-place operation will happen if sendbuff == recvbuff.

   Deprecated:
       ncclAllToAllv is replaced with ncclAlltoAllv and will be removed in the future.

   Args:
       sendbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Data array to send (contains blocks for each other rank)

       sendcounts (:py:obj:`~.rocm.bindings.util.types.ListOfUInt64`/:py:obj:`~.object`) -- *IN*:
           Array containing number of elements to send to each participating rank

       sdispls (:py:obj:`~.rocm.bindings.util.types.ListOfUInt64`/:py:obj:`~.object`) -- *IN*:
           Array of offsets into *sendbuff* for each participating rank

       recvbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           Data array to receive (contains blocks from each other rank)

       recvcounts (:py:obj:`~.rocm.bindings.util.types.ListOfUInt64`/:py:obj:`~.object`) -- *IN*:
           Array containing number of elements to receive from each participating rank

       rdispls (:py:obj:`~.rocm.bindings.util.types.ListOfUInt64`/:py:obj:`~.object`) -- *IN*:
           Array of offsets into *recvbuff* for each participating rank

       datatype (:py:obj:`~.ncclDataType_t`) -- *IN*:
           Data buffer element datatype

       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator group object to execute on

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           HIP stream to execute collective on

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclAllToAllv(const void * sendbuff, const size_t[] sendcounts, const size_t[] sdispls, void * recvbuff, const size_t[] recvcounts, const size_t[] rdispls, ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream)


.. py:function:: ncclSend(sendbuff, count, datatype, peer, comm, stream)

   Send

   Send data from *sendbuff* to rank *peer*.
   Rank *peer* needs to call ncclRecv with the same *datatype* and the same *count*
   as this rank.
   This operation is blocking for the GPU. If multiple ncclSend and ncclRecv operations
   need to progress concurrently to complete, they must be fused within a ncclGroupStart /
   ncclGroupEnd section.

   Args:
       sendbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Data array to send

       count (:py:obj:`~.int`) -- *IN*:
           Number of elements to send

       datatype (:py:obj:`~.ncclDataType_t`) -- *IN*:
           Data buffer element datatype

       peer (:py:obj:`~.int`) -- *IN*:
           Peer rank to send to

       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator group object to execute on

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           HIP stream to execute collective on

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclSend(const void * sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, hipStream_t stream)


.. py:function:: ncclRecv(recvbuff, count, datatype, peer, comm, stream)

   Receive

   Receive data from rank *peer* into *recvbuff*.
   Rank *peer* needs to call ncclSend with the same datatype and the same count
   as this rank.
   This operation is blocking for the GPU. If multiple ncclSend and ncclRecv operations
   need to progress concurrently to complete, they must be fused within a ncclGroupStart/
   ncclGroupEnd section.

   Args:
       recvbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           Data array to receive

       count (:py:obj:`~.int`) -- *IN*:
           Number of elements to receive

       datatype (:py:obj:`~.ncclDataType_t`) -- *IN*:
           Data buffer element datatype

       peer (:py:obj:`~.int`) -- *IN*:
           Peer rank to send to

       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`) -- *IN*:
           Communicator group object to execute on

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           HIP stream to execute collective on

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclRecv(void * recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, hipStream_t stream)


.. py:function:: ncclPutSignal(localbuff, count, datatype, peer, peerWin, peerWinOffset, sigIdx, ctx, flags, comm, stream)

   Put

   One-sided communication operation that writes data from the local buffer to a
   remote peer's registered memory window without explicit participation from the
   target process.

   Args:
       localbuff (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       count (:py:obj:`~.int`):
           (undocumented)

       datatype (:py:obj:`~.ncclDataType_t`):
           (undocumented)

       peer (:py:obj:`~.int`):
           (undocumented)

       peerWin (:py:obj:`~.ncclWindow_vidmem`/:py:obj:`~.object`):
           (undocumented)

       peerWinOffset (:py:obj:`~.int`):
           (undocumented)

       sigIdx (:py:obj:`~.int`):
           (undocumented)

       ctx (:py:obj:`~.int`):
           (undocumented)

       flags (:py:obj:`~.int`):
           (undocumented)

       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`):
           (undocumented)

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclPutSignal(const void * localbuff, size_t count, ncclDataType_t datatype, int peer, ncclWindow_t peerWin, size_t peerWinOffset, int sigIdx, int ctx, unsigned int flags, ncclComm_t comm, hipStream_t stream)


.. py:function:: ncclSignal(peer, sigIdx, ctx, flags, comm, stream)

   Signal

   Sends a signal to the specified peer without transferring data.

   Args:
       peer (:py:obj:`~.int`):
           (undocumented)

       sigIdx (:py:obj:`~.int`):
           (undocumented)

       ctx (:py:obj:`~.int`):
           (undocumented)

       flags (:py:obj:`~.int`):
           (undocumented)

       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`):
           (undocumented)

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclSignal(int peer, int sigIdx, int ctx, unsigned int flags, ncclComm_t comm, hipStream_t stream)


.. py:class:: ncclWaitSignalDesc_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: opCnt
      :type:  Any


   .. py:attribute:: peer
      :type:  Any


   .. py:attribute:: sigIdx
      :type:  Any


   .. py:attribute:: ctx
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:function:: ncclWaitSignal(nDesc, signalDescs, comm, stream)

   Wait Signal

   Waits for signals as described in the signal descriptor array.

   Args:
       nDesc (:py:obj:`~.int`):
           (undocumented)

       signalDescs (:py:obj:`~.ncclWaitSignalDesc_t`/:py:obj:`~.object`):
           (undocumented)

       comm (:py:obj:`~.ncclComm`/:py:obj:`~.object`):
           (undocumented)

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclWaitSignal(int nDesc, ncclWaitSignalDesc_t * signalDescs, ncclComm_t comm, hipStream_t stream)


.. py:function:: ncclGroupStart()

   Group Start

   Start a group call. All calls to RCCL until ncclGroupEnd will be fused into
   a single RCCL operation. Nothing will be started on the HIP stream until
   ncclGroupEnd.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclGroupStart()


.. py:function:: ncclGroupEnd()

   Group End

   End a group call. Start a fused RCCL operation consisting of all calls since
   ncclGroupStart. Operations on the HIP stream depending on the RCCL operations
   need to be called after ncclGroupEnd.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: Result code. See ``rccl_result_code`` for more details.

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclGroupEnd()


.. py:function:: ncclGroupSimulateEnd(simInfo)

   Group Simulate End

   Simulate a ncclGroupEnd() call and return NCCL's simulation info in a struct.

   Args:
       simInfo (:py:obj:`~.ncclSimInfo_v22200`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t * simInfo)


.. py:class:: ncclParamHandle(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: ncclParamHandle_t

.. py:function:: ncclParamBind()

   Look up the parameter identified by key and store a handle to it in
   out.

   The returned handle is owned by the parameter system and must not
   be freed by the caller.

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: (undocumented)
       * out (:py:obj:`~.ncclParamHandle`):
           (undocumented)
       * key (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclParamBind(ncclParamHandle_t ** out, const char * key)


.. py:function:: ncclParamGetI8(h)

   Read the value of the parameter bound to h as the type of out.

   Function names are suffixed with I/U and 8/16/32/64 for 8-, 16-, 32- and 64-bit
   signed and unsigned integers.

   Args:
       h (:py:obj:`~.ncclParamHandle`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: (undocumented)
       * out (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclParamGetI8(ncclParamHandle_t * h, int8_t * out)


.. py:function:: ncclParamGetI16(h)

   (No short description, might be part of a group.)

   Args:
       h (:py:obj:`~.ncclParamHandle`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: (undocumented)
       * out (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclParamGetI16(ncclParamHandle_t * h, int16_t * out)


.. py:function:: ncclParamGetI32(h)

   (No short description, might be part of a group.)

   Args:
       h (:py:obj:`~.ncclParamHandle`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: (undocumented)
       * out (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclParamGetI32(ncclParamHandle_t * h, int32_t * out)


.. py:function:: ncclParamGetI64(h)

   (No short description, might be part of a group.)

   Args:
       h (:py:obj:`~.ncclParamHandle`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: (undocumented)
       * out (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclParamGetI64(ncclParamHandle_t * h, int64_t * out)


.. py:function:: ncclParamGetU8(h)

   (No short description, might be part of a group.)

   Args:
       h (:py:obj:`~.ncclParamHandle`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: (undocumented)
       * out (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclParamGetU8(ncclParamHandle_t * h, uint8_t * out)


.. py:function:: ncclParamGetU16(h)

   (No short description, might be part of a group.)

   Args:
       h (:py:obj:`~.ncclParamHandle`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: (undocumented)
       * out (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclParamGetU16(ncclParamHandle_t * h, uint16_t * out)


.. py:function:: ncclParamGetU32(h)

   (No short description, might be part of a group.)

   Args:
       h (:py:obj:`~.ncclParamHandle`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: (undocumented)
       * out (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclParamGetU32(ncclParamHandle_t * h, uint32_t * out)


.. py:function:: ncclParamGetU64(h)

   (No short description, might be part of a group.)

   Args:
       h (:py:obj:`~.ncclParamHandle`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: (undocumented)
       * out (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclParamGetU64(ncclParamHandle_t * h, uint64_t * out)


.. py:function:: ncclParamGetStr(h, out)

   Read the value of the parameter bound to h as a string.

   Returned pointer is owned by the parameter system and is valid until the
   next ncclParamGetStr() call on the same thread.

   Args:
       h (:py:obj:`~.ncclParamHandle`/:py:obj:`~.object`):
           (undocumented)

       out (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclParamGetStr(ncclParamHandle_t * h, const char ** out)


.. py:function:: ncclParamGet(h, out, maxLen)

   Read the value of the parameter bound to h as raw binary data.

   The user needs to allocate a buffer for the result and the parameter value is copied
   into user buffer as bytes.

   Args:
       h (:py:obj:`~.ncclParamHandle`/:py:obj:`~.object`):
           (undocumented)

       out (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       maxLen (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: (undocumented)
       * len (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclParamGet(ncclParamHandle_t * h, void * out, int maxLen, int * len)


.. py:function:: ncclParamGetParameter(value)

   Get parameter value as string by key.

   Returned pointer is owned by the
   parameter system and is valid until the next ncclParamGetParameter() call
   on the same thread.

   Args:
       value (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: (undocumented)
       * key (:py:obj:`~.int`):
           (undocumented)
       * valueLen (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclParamGetParameter(const char * key, const char ** value, int * valueLen)


.. py:function:: ncclParamGetAllParameterKeys(table)

   Get all registered parameter keys.

   Returned pointer table is owned by the
   parameter system and is valid until the next ncclParamGetAllParameterKeys()
   call on the same thread. By default, the results include only parameters published
   in NCCL documentation. Setting NCCL_PARAM_DUMP_ALL=true will include all parameters.

   Args:
       table (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.ncclResult_t`: (undocumented)
       * tableLen (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       ncclResult_t ncclParamGetAllParameterKeys(const char *** table, int * tableLen)


.. py:function:: ncclParamDumpAll()

   Dump all parameters to log output.

   By default, the result includes only parameters published
   in NCCL documentation. Setting NCCL_PARAM_DUMP_ALL=true will include all parameters.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.ncclResult_t`:
               Always returns `~.ncclResult_t.ncclSuccess`.

   .. rubric:: C signature

   .. code-block:: c

       void ncclParamDumpAll()


