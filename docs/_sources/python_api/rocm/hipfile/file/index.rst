rocm.hipfile.file
=================

.. py:module:: rocm.hipfile.file


Classes
-------

.. autoapisummary::

   rocm.hipfile.file.FileHandle


Module Contents
---------------

.. py:class:: FileHandle(path, flags, mode=DEFAULT_MODE, handle_type=FileHandleType.OPAQUE_FD)

   Lifecycle manager for a hipFile-registered open file.

   Wraps `~.hipFileHandleRegister` / `~.hipFileHandleDeregister` plus
   synchronous `~.hipFileRead` / `~.hipFileWrite`.

   Use as a context manager:

       with FileHandle(path, os.O_RDWR | os.O_DIRECT) as fh:
           fh.read(buf, size, file_offset, buffer_offset)
           fh.write(buf, size, file_offset, buffer_offset)


   .. py:attribute:: DEFAULT_MODE
      :value: 420



   .. py:property:: handle_type


   .. py:property:: flags


   .. py:property:: handle


   .. py:property:: mode


   .. py:property:: path


   .. py:method:: open()


   .. py:method:: close()


   .. py:method:: read(buffer, size, file_offset, buffer_offset)

      Synchronous read into a registered `~.buffer.Buffer`.

      Returns the number of bytes read on success. Raises
      `~.error.HipFileException` (with the parsed
      `~.enums.OpError` and HIP driver error) on a hipFile-level
      error, or ``OSError`` (with the real ``errno``) on a
      POSIX-level error.



   .. py:method:: write(buffer, size, file_offset, buffer_offset)

      Synchronous write from a registered `~.buffer.Buffer`.

      Returns the number of bytes written on success. Same error
      semantics as `~.FileHandle.read`.



