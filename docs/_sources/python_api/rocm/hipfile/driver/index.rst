rocm.hipfile.driver
===================

.. py:module:: rocm.hipfile.driver


Classes
-------

.. autoapisummary::

   rocm.hipfile.driver.Driver


Module Contents
---------------

.. py:class:: Driver

   Lifecycle manager for the hipFile driver.

   Each instance brackets one ``hipFileDriverOpen`` /
   ``hipFileDriverClose`` pair. The driver is reference-counted by the
   library; multiple ``Driver`` instances coexist safely.

   Use as a context manager:

       with Driver():
           ...

   or call `~.Driver.open` / `~.Driver.close` explicitly.


   .. py:method:: use_count()
      :staticmethod:


      Return the current driver reference count.



   .. py:method:: close()


   .. py:method:: open()


