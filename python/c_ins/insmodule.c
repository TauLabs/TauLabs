#include <Python.h>
#include "math.h"

#define NPY_NO_DEPRECATED_API 7
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"

#include <insgps.h>

int not_doublevector(PyArrayObject *vec)
{
	if (PyArray_TYPE(vec) != NPY_DOUBLE) {
		PyErr_SetString(PyExc_ValueError,
              "Vector is not a float or double vector.");
		return 1;
	}
	if (PyArray_NDIM(vec) != 1)  {
		PyErr_Format(PyExc_ValueError,
              "Vector is not a 1 dimensional vector (%d).", PyArray_NDIM(vec));
		return 1;  
	}
	return 0;
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
static bool parseFloatVecN(PyArrayObject *vec_in, float *vec_out, int N)
{
	/* verify it is a valid vector */
	if (not_doublevector(vec_in))
		return false;

	if (PyArray_DIM(vec_in,0) != N) {
		PyErr_Format(PyExc_ValueError, "Vector length incorrect. Received %ld and expected %d", PyArray_DIM(vec_in,0), N);
		return false;
	}

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
	fprintf(stderr, "Parse fail\r\n");
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

	if (!parseFloatVec3(vec_gyro, gyro_data))
		return NULL;
	if (!parseFloatVec3(vec_accel, accel_data))
		return NULL;

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

	if (!parseFloatVecN(vec_z, z, 10))
		return NULL;

	INSCorrection(&z[6], &z[0], &z[3], z[9], sensors);

	return pack_state(self);
}

/**
 * configure the EKF paramters (e.g. variances)
 * @params[in] self
 * @params[in] args
 *  - mag_var
 *  - accel_var
 *  - gyro_var
 *  - baro_var
 * @return nothing
 */
static PyObject*
configure(PyObject* self, PyObject* args, PyObject *kwarg)
{
	static char *kwlist[] = {"mag_var", "accel_var", "gyro_var", "baro_var", "gps_var", NULL};

	PyArrayObject *mag_var = NULL, *accel_var = NULL, *gyro_var = NULL, *gps_var = NULL;
	float baro_var = 0.0f;

	if (!PyArg_ParseTupleAndKeywords(args, kwarg, "|OOOfO", kwlist,
		 &mag_var, &accel_var, &gyro_var, &baro_var, &gps_var)) {
		return NULL;
	}

	if (mag_var) {
		float mag[3];
		if (!parseFloatVec3(mag_var, mag))
			return NULL;
		INSSetMagVar(mag);
	}

	if (accel_var) {
		float accel[3];
		if (!parseFloatVec3(accel_var, accel))
			return NULL;
		INSSetAccelVar(accel);
	}

	if (gyro_var) {
		float gyro[3];
		if (!parseFloatVec3(gyro_var, gyro))
			return NULL;
		INSSetGyroVar(gyro);
	}

	if (baro_var != 0.0f) {
		INSSetBaroVar(baro_var);
	}

	if (gps_var) {
		float gps[3];
		if (!parseFloatVec3(gps_var, gps))
			return NULL;
		INSSetPosVelVar(gps[0], gps[1], gps[2]);
	}

	return Py_None;
}

static PyObject*
set_state(PyObject* self, PyObject* args, PyObject *kwarg)
{
	static char *kwlist[] = {"pos", "vel", "q", "gyro_bias", "accel_bias", NULL};

	PyArrayObject *vec_pos = NULL, *vec_vel = NULL, *vec_q = NULL, *vec_gyro_bias = NULL, *vec_accel_bias = NULL;

	if (!PyArg_ParseTupleAndKeywords(args, kwarg, "|OOOOO", kwlist,
		 &vec_pos, &vec_vel, &vec_q, &vec_gyro_bias, &vec_accel_bias)) {
		return NULL;
	}

	float pos[3], vel[3], q[4], gyro_bias[3], accel_bias[3];
	INSGetState(pos, vel, q, gyro_bias, accel_bias);

	// Overwrite state with any that were passed in
	if (vec_pos) {
		if (!parseFloatVec3(vec_pos, pos))
			return NULL;
	}
	if (vec_vel) {
		if (!parseFloatVec3(vec_vel, vel))
			return NULL;
	}
	if (vec_q) {
		if (!parseFloatVecN(vec_q, q, 4))
			return NULL;
	}
	if (vec_gyro_bias) {
		if (!parseFloatVec3(vec_gyro_bias, gyro_bias))
			return NULL;
	}
	if (vec_accel_bias) {
		if (!parseFloatVec3(vec_accel_bias, accel_bias))
			return NULL;
	}

	INSSetState(pos, vel, q, gyro_bias, accel_bias);

	return Py_None;
}


static PyObject*
init(PyObject* self, PyObject* args)
{
	INSGPSInit();

	const float Be[] = {400, 0, 1600};
	INSSetMagNorth(Be);

	return pack_state(self);	
}

static PyMethodDef InsMethods[] =
{
	{"init", init, METH_VARARGS, "Reset INS state."},
	{"prediction", prediction, METH_VARARGS, "Advance state 1 time step."},
	{"correction", correction, METH_VARARGS, "Apply state correction based on measured sensors."},
	{"configure", (PyCFunction)configure, METH_VARARGS|METH_KEYWORDS, "Configure EKF parameters."},
	{"set_state", (PyCFunction)set_state, METH_VARARGS|METH_KEYWORDS, "Set the EKF state."},
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
