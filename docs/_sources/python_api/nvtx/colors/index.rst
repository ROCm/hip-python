nvtx.colors
===========

.. py:module:: nvtx.colors

.. autoapi-nested-parse::

   Color helpers for the ``nvtx`` compatibility shim.

   This mirrors the ``nvtx.colors`` submodule of the upstream (Apache-2.0)
   ``nvtx`` package so that code importing ``nvtx.colors`` keeps working. It is a
   fresh, MIT-licensed re-implementation.

   .. important::

      Colors have **no runtime effect** in this shim. The underlying
      ``rocm.bindings.roctx`` bindings are message-only and expose no color
      channel, so any color computed here is dropped before the event reaches
      roctx. These helpers exist purely for source compatibility.



Functions
---------

.. autoapisummary::

   nvtx.colors.color_to_hex


Module Contents
---------------

.. py:function:: color_to_hex(color=None)

   Convert a color to its ARGB hex value.

   Accepts an integer (returned unchanged), one of the built-in color names,
   or - when ``matplotlib`` is installed - any matplotlib color spec.

   Note:
       The returned value is not forwarded to roctx (which has no color
       channel); this function is provided for API compatibility only.


