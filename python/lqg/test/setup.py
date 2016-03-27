#/usr/bin/evn python

from distutils.core import setup, Extension, Command
import numpy

module1 = Extension('rtkf',
	sources = ['rtkf_module.c', 'pios_heap.c', '../../../flight/Libraries/lqg_rate/rate_torque_kf.c'],
	            include_dirs=['../../../flight/Libraries/lqg_rate','../../../flight/PiOS/inc',numpy.get_include()],
                    extra_compile_args=['-std=gnu99'],)

module2 = Extension('rtsi',
	sources = ['rtsi_module.c', 'pios_heap.c', '../../../flight/Libraries/lqg_rate/rate_torque_si.c'],
	            include_dirs=['../../../flight/Libraries/lqg_rate','../../../flight/PiOS/inc',numpy.get_include()],
                    extra_compile_args=['-std=gnu99'],)
 
setup (name = 'RateTorqueLQG',
        version = '1.0',
        description = 'Rate Torque LQG python test code',
        ext_modules = [module1, module2])
