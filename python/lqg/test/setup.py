#/usr/bin/evn python

from distutils.core import setup, Extension, Command
import numpy

module1 = Extension('rtkf',
	sources = ['rtkf_module.c', 'pios_heap.c', '../../../flight/Libraries/lqg_rate/rate_torque_kf.c'],
	            include_dirs=['../../../flight/Libraries/lqg_rate','../../../flight/PiOS/inc',numpy.get_include()],
                    extra_compile_args=['-std=gnu99'],)
 
setup (name = 'PackageName',
        version = '1.0',
        description = 'Rate Torque KF C Module',
        ext_modules = [module1])
