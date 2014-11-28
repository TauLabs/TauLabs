from distutils.core import setup, Extension, Command
import numpy

module1 = Extension('ins',
	sources = ['insmodule.c', '../../flight/Libraries/insgps13state.c'],
	include_dirs=['../../flight/Libraries/inc','../../shared/api',numpy.get_include()])
 
setup (name = 'PackageName',
        version = '1.0',
        description = 'INS C module',
        ext_modules = [module1])
