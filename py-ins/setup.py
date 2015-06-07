from distutils.core import setup, Extension, Command
import numpy

module1 = Extension('ins',
	sources = ['insmodule.c', '../../flight/Libraries/insgps14state.c'],
	            include_dirs=['../../flight/Libraries/inc','../../shared/api',numpy.get_include()],
                    extra_compile_args=['-std=gnu99'],)
 
setup (name = 'PackageName',
        version = '1.0',
        description = 'INS C module',
        ext_modules = [module1])
