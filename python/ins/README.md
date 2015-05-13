To run the unit tests of the C implementation of the INS run

   python setup.py build_ext --inplace
   python test.py

this will compile a cython wrapper and then run a series of
unit tests on convergence and convergence rates.
