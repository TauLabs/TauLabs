
def mag_calibration(mag,gyros=None,LH=200,LV=500):
	""" Calibrates the magnetometer data by fitting it to a sphere,
	ideally when constantly turning to spread the data around that
	sphere somewhat evenly (or at least in a horizontal plane)"""

	import numpy
	from scipy.optimize import minimize
	from numpy.core.multiarray import arange

	def find_spinning(mag,gyros):
		""" return the indicies in the magnetometer data when
		the gyro indicates it is spinning on the z axis """

		import numpy
		import scipy.signal
		from matplotlib.mlab import find

		threshold = 40
		spinning = scipy.signal.medfilt(abs(gyros['z'][:,0]),kernel_size=5) > threshold

		# make sure to always find end elements
		spinning = numpy.concatenate((numpy.array([False]),spinning,numpy.array([False])))
		start = find(spinning[1:] & ~spinning[0:-1])
		stop = find(~spinning[1:] & spinning[0:-1])-1

		tstart = gyros['time'][start]
		tstop = gyros['time'][stop]

		idx = numpy.zeros((0),dtype=int)
		for i in arange(tstart.size):

			i1 = abs(mag['time']-tstart[i]).argmin()
			i2 = abs(mag['time']-tstop[i]).argmin()
			
			idx = numpy.concatenate((idx,arange(i1,i2,dtype=int)))

		return idx

	if gyros is not None:
		idx = find_spinning(mag,gyros)
	else:
		idx = arange(mag['time'].size)

	mag_x = mag['x'][idx,0]
	mag_y = mag['y'][idx,0]
	mag_z = mag['z'][idx,0]

	rx = max(mag_x) - min(mag_x)
	ry = max(mag_y) - min(mag_y)

	mx = rx / 2 + min(mag_x)
	my = ry / 2 + min(mag_y)

	def distortion(x,mag_x=mag_x,mag_y=mag_y,mag_z=mag_z,LH=LH,LV=LV):
		""" loss function for distortion from spherical data """
		from numpy import sqrt, mean
		cor_x = mag_x * x[0] - x[3]
		cor_y = mag_y * x[1] - x[4]
		cor_z = mag_z * x[2] - x[5]
		l = sqrt(cor_x**2 + cor_y**2 + cor_z**2)
		L0 = sqrt(LH**2 + LV**2)
		spherical_error = numpy.mean((l - L0)**2)

		# note that ideally the horizontal error would be calculated
		# after correcting for attitude but that requires high temporal
		# accuracy from attitude which we don't want to requires. this
		# works well in practice.
		lh = sqrt(cor_x**2 + cor_y**2)
		err = (lh - LH)**2
		horizontal_error = numpy.mean(err)

		# weight both the spherical error and the horizontal error
		# components equally
		return spherical_error+horizontal_error

	cons = ({'type': 'ineq', 'fun' : lambda x: numpy.array([x[0] - 0.5])},
	        {'type': 'ineq', 'fun' : lambda x: numpy.array([x[1] - 0.5])},
	        {'type': 'ineq', 'fun' : lambda x: numpy.array([x[2] - 0.5])})
	opts = {'xtol': 1e-8, 'disp': False, 'maxiter': 10000}

	# method of COBYLA also works well
	x0 = numpy.array([1, 1, 1, numpy.mean(mag_x), numpy.mean(mag_y), numpy.mean(mag_z)])
	res = minimize(distortion, x0, method='COBYLA', options=opts, constraints=cons)

	x = res.x
	cor_x = mag_x * x[0] - x[3]
	cor_y = mag_y * x[1] - x[4]
	cor_z = mag_z * x[2] - x[5]

	import matplotlib
	from numpy import sqrt
	matplotlib.pyplot.subplot(1,2,1)
	matplotlib.pyplot.plot(cor_x,cor_y,'.',cor_x,cor_z,'.',cor_z,cor_y,'.')
	#matplotlib.pyplot.xlim(-1,1)
	#matplotlib.pyplot.ylim(-1,1)
	matplotlib.pyplot.subplot(1,2,2)
	matplotlib.pyplot.plot(sqrt(cor_x**2+cor_y**2+cor_z**2))

	return res, cor_x, cor_y, cor_z