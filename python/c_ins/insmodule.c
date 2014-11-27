#include <Python.h>
#include "math.h"

#define NPY_NO_DEPRECATED_API 7
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"

#include <insgps.h>

/**
 * parseFloatVec3(vec_in, vec_out)
 *
 * @param[in] vec_in the python array to extract elements from
 * @param[out] vec_out float array of the numbers
 * @return true if successful, false if not
 *
 * Convert a python array type to a 3 element float
 * vector.
 */
static bool parseFloatVecN(PyArrayObject *vec_in, float *vec_out, int N)
{
	NpyIter *iter;
	NpyIter_IterNextFunc *iternext;
  
  /*  create the iterators */
	iter = NpyIter_New(vec_in, NPY_ITER_READONLY, NPY_KEEPORDER,
							 NPY_NO_CASTING, NULL);
	if (iter == NULL)
		goto fail;

  iternext = NpyIter_GetIterNext(iter, NULL);
  if (iternext == NULL) {
		NpyIter_Deallocate(iter);
		goto fail;
	}

	double ** dataptr = (double **) NpyIter_GetDataPtrArray(iter);

	/*  iterate over the arrays */
	int i = 0;
	do {
		vec_out[i++] = **dataptr;
	} while(iternext(iter) && (i < N));

  NpyIter_Deallocate(iter);

  return true;

fail:
  return false;
}

/**
 * parseFloatVec3(vec_in, vec_out)
 *
 * @param[in] vec_in the python array to extract elements from
 * @param[out] vec_out float array of the numbers
 * @return true if successful, false if not
 *
 * Convert a python array type to a 3 element float
 * vector.
 */
static bool parseFloatVec3(PyArrayObject *vec_in, float *vec_out)
{
  return parseFloatVecN(vec_in, vec_out, 3);
}

/**
 * pack_state put the state information into an array
 */
static PyObject*
pack_state(PyObject* self)
{
	float pos[3], vel[3], q[4], gyro_bias[3], accel_bias[3];        
	INSGetState(pos, vel, q, gyro_bias, accel_bias);

	const int N = 16;
	int nd = 1;
	int dims[1];
	dims[0] = N;

	PyArrayObject *state;
	state = (PyArrayObject*) PyArray_FromDims(nd, dims, NPY_DOUBLE);
	double *s = (double *) PyArray_DATA(state);

	s[0] = pos[0];
	s[1] = pos[1];
	s[2] = pos[2];
	s[3] = vel[0];
	s[4] = vel[1];
	s[5] = vel[2];
	s[6] = q[0];
	s[7] = q[1];
	s[8] = q[2];
	s[9] = q[3];
	s[10] = gyro_bias[0];
	s[11] = gyro_bias[1];
	s[12] = gyro_bias[2];
	s[13] = accel_bias[0];
	s[14] = accel_bias[1];
	s[15] = accel_bias[2];

	return Py_BuildValue("O", state);
}

/**
 * prediction - perform a state prediction step
 * @params[in] self
 * @params[in] args
 *  - gyro
 *  - accel
 *  - dT
 * @return state
 */
static PyObject*
prediction(PyObject* self, PyObject* args)
{
	PyArrayObject *vec_gyro, *vec_accel;
	float gyro_data[3], accel_data[3];
	float dT;

	if (!PyArg_ParseTuple(args, "O!O!f", &PyArray_Type, &vec_gyro,
				   &PyArray_Type, &vec_accel, &dT))  return NULL;
	if (NULL == vec_gyro)  return NULL;
	if (NULL == vec_accel)  return NULL;

	parseFloatVec3(vec_gyro, gyro_data);
	parseFloatVec3(vec_accel, accel_data);

	INSStatePrediction(gyro_data, accel_data, dT);
	INSCovariancePrediction(dT);

	if (false) {
		const float zeros[3] = {0,0,0};
		INSSetGyroBias(zeros);
		INSSetAccelBias(zeros);
	}

	return pack_state(self);
}
 
/**
 * correction - perform a correction of the EKF
 * @params[in] self
 * @params[in] args
 *  - Z - vector of measurements (position, velocity, mag, baro)
 *  - sensors - binary flags for which sensors should be used
 * @return state
 */
 static PyObject*
correction(PyObject* self, PyObject* args)
{
	PyArrayObject *vec_z;
	float z[10];
	int sensors;

	if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &vec_z,
				   &sensors))  return NULL;
	if (NULL == vec_z)  return NULL;

	parseFloatVecN(vec_z, z, 10);

	INSCorrection(&z[6], &z[0], &z[3], z[9], sensors);

	return pack_state(self);
}

static PyObject*
init(PyObject* self, PyObject* args)
{
	const float mag_var[3] = {0.1f, 0.1f, 10.0f};
	const float accel_var[3] = {3e-3f, 3e-3f, 3e-3f};
	const float gyro_var[3] = {1e-5f, 1e-5f, 1e-4f};
	const float baro_var = 0.01f;
	const float gps_pos_var = 1e-5f;
	const float gps_vel_var = 1e-5f;
	const float gps_vert_var = 10.0f;
	INSSetMagVar(mag_var);
	INSSetAccelVar(accel_var);
	INSSetGyroVar(gyro_var);
	INSSetBaroVar(baro_var);
	INSSetPosVelVar(gps_pos_var, gps_vel_var, gps_vert_var);

	return pack_state(self);	
}

static PyMethodDef InsMethods[] =
{
	{"init", init, METH_VARARGS, "Reset INS state."},
	{"prediction", prediction, METH_VARARGS, "Advance state 1 time step."},
	{"correction", correction, METH_VARARGS, "Apply state correction based on measured sensors."},
	{NULL, NULL, 0, NULL}
};
 
PyMODINIT_FUNC
initins(void)
{
	(void) Py_InitModule("ins", InsMethods);
	import_array();
	init(NULL, NULL);
	INSGPSInit();
}
